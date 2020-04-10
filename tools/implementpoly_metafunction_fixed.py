# -*- coding: utf-8 -*-
import sys

import sollya, polynomial2tree
import os
from sollya import Interval, sup, inf

# constant for sollya's version of number 2
S2 = sollya.SollyaObject(2)
sollya.settings.display = sollya.decimal

# local module to trace function easily
from function_trace import function_trace, function_trace_hex_2
import function_trace as ft_module

from metalibm_core.core.ml_formats import (
    get_sollya_from_long,
    ML_Custom_FixedPoint_Format,
)
from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.core.ml_operations import (
    Return, Conversion, Statement
)

from metalibm_core.utility.log_report import Log
from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)

from metalibm_core.code_generation.code_constant import C_Code

from metalibm_core.targets.common import FixedPointBackend

import metalibm_core.opt.runtime_error_eval as runtime_error_eval
from metalibm_core.opt.opt_utils import depth_node_ordering


import ml_fixed_wrapper

sollya.settings.verbosity = 2

# importing sollya scripts
try:
    sollya_scripts_path = os.environ["GENERICIMPLEMENTPOLY_ROOT"]
except KeyError:
    sollya_scripts_path = ""

SOLLYA_SCRIPT_POLYNOMIAL_PATH = os.path.join(sollya_scripts_path, "polynomials.sol")

print("Imporintg genericimplepoly sollya script: {}".format(SOLLYA_SCRIPT_POLYNOMIAL_PATH))

sollya.execute(SOLLYA_SCRIPT_POLYNOMIAL_PATH)

# importing into python, the function generateGappa
# loaded from polynomials.sol
generateGappa = sollya.parse("generateGappa")

# importing into python, the function computeErrorBounds
# loaded from polynomials.sol
computeErrorBounds = sollya.parse("computeErrorBounds")

# importing into python, the function containsCancellations
# loaded from polynomials.sol
containsCancellations = sollya.parse("containsCancellations")

def convert_value_to_bit_weigth(value):
    return int(sollya.floor(sollya.log2(abs(value))))

def get_value_indexes(value):
    """ return the MSB and LSB indexes to encode value without error"""
    signed = value < 0
    value = abs(value)
    msb_index = convert_value_to_bit_weigth(value) + (1 if signed else 0)
    # determining lsb_index
    lsb_index = msb_index
    while get_sollya_from_long(int(value * S2**-lsb_index)) != value * S2**-lsb_index:
        lsb_index -= 1
    return msb_index, lsb_index

def get_accuracy_from_epsilon(epsilon):
    """ convert a numerical relative error into
        a number of accuracy bits """
    return sollya.floor(-sollya.log2(abs(epsilon)))

class MLL_FixedFormat:
    """ Container for all information required to
        describe a node format as infered by MLL_Context """
    def __init__(self, fixed_format, interval, delta_target=None, eps_target=None,exact=False):
        # Metalibm's fixed-point format to be used
        self.fixed_format = fixed_format
        # interval: range of the node values
        self.interval = interval
        # initial requirement for delta (absolute) target
        self.delta_target = delta_target
        # initial requirement for epsilon (relative) target
        self.eps_target = eps_target
        # if the operation is exact
        self.exact = exact

    @property
    def rel_error(self):
        if self.exact:
            return 0.0
        else:
            max_abs_error = S2**-self.fixed_format.get_frac_size()
            return max_abs_error / inf(abs(self.interval))

    @property
    def abs_error(self):
        if self.exact:
            return 0.0
        else:
            max_abs_error = S2**-self.fixed_format.get_frac_size()
            return max_abs_error

    def _sollya_(self):
        """ convert to SollyaObject (by wrapping in an externaldata
            container) """
        return sollya.externaldata(self)

    def __str__(self):
        """ convert to string (used for comprehensive content display) """
        return "(fixed_format={}, interval={}, delta_target={}, exact={})".format(
            str(self.fixed_format), str(self.interval), str(self.delta_target), self.exact
        )

    @property
    def struct(self):
        """ Handler for compatibility with sollya's record """
        return self


class MLL_FixedContext:
    """ Class to wrapper the opaque type which bridges computeErrorBounds
        and metalibm Lugdunum """
    def __init__(self, var_format, var_interval):
        # dummy 1000-bit accuracy
        self.variableFormat = MLL_FixedFormat(
            var_format,
            var_interval,
            delta_target=0,
            eps_target=0,
            exact=True
        )
        # dictionnary of power -> MLL_Format
        self.power_map = {
            #1: [(0, self.variableFormat)]
        }
        # dict k -> (i, j) which indicates that X^k must be compute as
        #                  X^i * X^j
        self.power_schemes = {}
        # maximum number of word in the largest multi-word
        self.LIMB_NUM_THRESHOLD = 3
        # minimal difference factor betwen hi and medium limb which triggers
        # the insertion of a renormalization operation
        self.LIMB_DIFF_RENORM_THRESHOLD = S2**-15

    def __str__(self):
        return "MLL_Context"

    def get_ml_fixed_format_from_abs_error(self, interval, abs_error):
        """ return a fixed-point format able to contain @p interval's max value
            and with sufficient bit to ensure LSB weigth is lower than
            @p abs_error """
        assert abs_error > 0
        signed = inf(interval) < 0
        integer_size = int(sollya.ceil(sollya.log2(sup(abs(interval)))))
        frac_size = -int(sollya.floor(sollya.log2(abs_error)))
        return fixed_point_format(integer_size, frac_size, signed=signed)

    def get_ml_fixed_format_from_rel_error(self, interval, rel_error):
        assert rel_error > 0
        # relative_error = absolute_error / min(value)
        min_value = inf(abs(interval))
        absolute_error = abs(rel_error) * min_value
        fixed_format = self.get_ml_fixed_format_from_abs_error(interval, absolute_error)
        return fixed_format


    def get_format_from_abs_error(self, abs_error, interval=None):
        """ Generate a format which ensure that a given value within interval will
            be converted without absolute error exceeding abs_error

            :param abs_error:
            :type abs_error: numerical value [float, sollya.SollyaObject, ...]
            :param interval: value range
            :type interval: sollya.Interval
        """
        low_level_format = self.get_ml_fixed_format_from_abs_error(interval, abs_error)
        return MLL_FixedFormat(low_level_format, interval, delta_target=abs_error, exact=(abs_error==0.0))

    def get_format_from_rel_error(self, rel_error, eps_target=None, interval=None):
        low_level_format = self.get_ml_fixed_format_from_rel_error(interval, rel_error)
        return MLL_FixedFormat(low_level_format, interval, eps_target=eps_target, exact=(rel_error==0.0))


    @function_trace(4)
    def computeBoundAddition(self, out_format, input_format_lhs, input_format_rhs):
        return out_format.rel_error

    @function_trace(4)
    def computeBoundMultiplication(self, out_format, input_format_lhs, input_format_rhs):
        return out_format.rel_error

    @function_trace(4)
    def computeBoundPower(self, k, out_format, var_format):
        # TODO: fix
        epsilon = out_format.struct.rel_error
        eps_target = out_format.struct.eps_target
        # to avoid derivating to larger and larger format when post-processing
        # powerings, we over-estimate the error while matching eps_target
        if eps_target > epsilon and epsilon > 0:
            # limiting error to limit precision explosion
            l_eps_target = sollya.log2(eps_target)
            l_epsilon = sollya.log2(epsilon)

            virtual_error_log = (l_eps_target + l_epsilon) / 2.0
            virtual_error = sollya.evaluate(sollya.SollyaObject(S2**(virtual_error_log)), 1)
            print("lying on power_error target=2^{}, epsilon=2^{}, virtual_error=2^{} / {}".format(
                l_eps_target, l_epsilon, virtual_error_log, virtual_error))
            return virtual_error
        else:
            return sollya.SollyaObject(epsilon)

    @function_trace(3)
    def compareFormats(self, format_a, format_b):
        if format_a.struct.rel_error < format_b.struct.rel_error:
            return -1
        elif format_a.struct.rel_error == format_b.struct.rel_error:
            return 0
        else:
            return 1

    @function_trace(3)
    def computeBoundVariableRounding(self, format_a, format_b):
        """ Returns a bound on the relative error implied by "rounding" _x_,
           which is naturally stored as an @p format_a variable,
           to a format_b variable.

           If bf is "larger" (i.e. more precise) than af,
           the error is most probably zero as this just means
           "upconverting" a variable.

           The inputs af and bf are of the "opaque format type".

           The result is a positive or zero real number.
           """
        if format_a.struct.rel_error >= format_b.struct.rel_error:
            return 0
        else:
            return format_b.struct.rel_error


    @function_trace(4)
    def computeOutputFormatAddition(self, epsTarget, inputFormatA, inputFormatB):
        """
           Returns the output format of an addition  that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        # we suppose the operation computed exactly and then truncated
        lhs_format = inputFormatA.struct
        rhs_format = inputFormatB.struct
        add_interval = lhs_format.interval + rhs_format.interval
        integer_size = max(lhs_format.fixed_format.integer_size, rhs_format.fixed_format.integer_size) + 1
        signed = lhs_format.fixed_format.signed or rhs_format.fixed_format.signed
        frac_size = max(lhs_format.fixed_format.frac_size, rhs_format.fixed_format.frac_size)
        input_driven_format = MLL_FixedFormat(
            fixed_point_format(integer_size, frac_size, signed=signed),
            add_interval,
            epsTarget=epsTarget,
            exact=True
        )
        if epsTarget == 0 or 0 in add_interval:
            # TODO: format computation could be optimized
            return input_driven_format

        else:
            target_format = self.get_format_from_rel_error(epsTarget, epsTarget, add_interval)
            if target_format.fixed_format.frac_size > integer_size:
                # the target format exceeds the fraction size reachable by the inputs
                return input_driven_format
            return target_format


    @function_trace(4)
    def computeOutputFormatMultiplication(self, epsTarget, inputFormatA, inputFormatB):
        """ Returns the output format of a multiplication that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        # we suppose the operation computed exactly and then truncated
        lhs_format = inputFormatA.struct
        rhs_format = inputFormatB.struct
        mul_interval = lhs_format.interval * rhs_format.interval
        if epsTarget == 0 or 0 in mul_interval:
            # TODO: format computation could be optimized
            signed = lhs_format.fixed_format.signed ^ rhs_format.fixed_format.signed
            integer_size = lhs_format.fixed_format.integer_size + rhs_format.fixed_format.integer_size
            frac_size = lhs_format.fixed_format.frac_size + rhs_format.fixed_format.frac_size
            return MLL_FixedFormat(
                fixed_point_format(integer_size, frac_size, signed=signed),
                mul_interval,
                eps_target=epsTarget,
                exact=True
            )

        else:
            return self.get_format_from_rel_error(epsTarget, epsTarget, mul_interval)

    @function_trace_hex_2
    def computeConstantFormat(self, c):
        """ This function takes a constant that has been obtained
           using the function roundConstantToTargetPrecFormatSpecific
           and returns the format of this constant.

           The input is a real number.
           The output is of the "opaque format" type.
        """
        if c == 0:
            # default to single bit value precision
            return MLL_FixedFormat(
                fixed_point_format(1, 0, signed=False),
                Interval(0),
                delta_target=0,
                eps_target=0,
                exact=True
            )
        else:
            signed = c < 0
            msb_index, lsb_index = get_value_indexes(abs(c))
            return MLL_FixedFormat(
                fixed_point_format(msb_index + 1, -lsb_index, signed=signed),
                Interval(c),
                delta_target=0,
                eps_target=0,
                exact=True
            )

    @function_trace(4)
    def computeNeededVariableFormat(self, I, epsTarget, variableFormat):
        if epsTarget == 0 or 0 in variableFormat.interval:
            return variableFormat
        elif epsTarget > 0:
            if epsTarget < variableFormat.rel_error:
                # if expected error is less that what the variable format
                # can provide, we can return it directly
                return variableFormat
            else:
                return self.get_format_from_rel_error(epsTarget, epsTarget, I)
        else:
            raise NotImplementedError

    @function_trace(4)
    def computeOutputFormatPower(self, k, epsTarget, variableFormat):
        # convert k to python integer
        if not k is None:
            k = int(k)
        if epsTarget > 0:
            if k == 1:
                # if k is 1, then the format is the one of Variable verifying
                # epsTarget constraint
                return self.computeNeededVariableFormat(None, epsTarget, variableFormat)
            else:
                final_format = None
                if k in self.power_map:
                    for error_bound, power_format_list in self.power_map[k]:
                        # if record error_bound is less than epsTarget
                        # TODO: implement search for tighter bound
                        if epsTarget > error_bound:
                            # final_format is last element of power_format_list
                            final_format = power_format_list[-1]
                            if final_format.rel_error < epsTarget:
                                # there is possibly a smaller format than final_format which
                                # could match epsTarget constraint
                                final_format = None # GR: to be checked ??
                                continue
                            # updating final_format eps_target to make sure it
                            # always keep track of the most constraining
                            # eps_target
                            final_format.eps_target = min(final_format.eps_target, epsTarget)
                            return final_format
                        # as error_bound are ordered from larger to smaller
                if final_format is None:
                    if k == 2:
                        lhs_k, rhs_k = 1, 1
                    else:
                        lhs_k, rhs_k = self.power_schemes[k]
                    # TODO: implement more complex error budget
                    lhs_error_budget = epsTarget / 4
                    rhs_error_budget = epsTarget / 4
                    lhs_format = self.computeOutputFormatPower(lhs_k, lhs_error_budget, variableFormat)
                    rhs_format = self.computeOutputFormatPower(rhs_k, rhs_error_budget, variableFormat)
                    mult_format = self.computeOutputFormatMultiplication(epsTarget / 2, lhs_format, rhs_format)

                    lhs_error = self.computeBoundPower(lhs_k, lhs_format, variableFormat)
                    rhs_error = self.computeBoundPower(rhs_k, rhs_format, variableFormat)
                    mult_error = self.computeBoundMultiplication(mult_format, lhs_format, rhs_format)


                    # TODO: should take into account second order error
                    final_error = lhs_error + rhs_error + mult_error
                    if final_error > epsTarget:
                        print("possible ERROR in computeOutputFormatPower: failed to meet epsTarget")

                    final_format = mult_format
                    final_format.eps_target = epsTarget

                    record = (final_error, [lhs_format, rhs_format, final_format])

                    if not k in self.power_map:
                        self.power_map[k] = []
                    self.power_map[k].append(record)
                    # sorting list from large error to small errors
                    self.power_map[k].sort(key=(lambda v: v[0]), reverse=True)

                return final_format
        else:
            return sollya.error

    @function_trace(3)
    def roundConstant(self, c, epsTarget):
        """ Rounds a given coefficient c into a format that guarantees
           that the rounding error is less than epsTarget. The function
           does not return the retained format but the rounded number.

           epsTarget is a positive OR ZERO number.

           If epsTarget is zero, the function is supposed to check
           whether there exists a format such that the constant can be
           represented exactly.

           The function returns a structure with at least two fields

           *   .okay indicating that the rounding was able to be performed
           *   .c    the rounded constant

        """
        if epsTarget >= 0:
                if c == 0:
                    # 0 value can always fit exactly in a format
                    return sollya.SollyaObject({"okay": True, "c": c})
                else:
                    if epsTarget == 0:
                        msb_index, lsb_index = get_value_indexes(c)
                        return sollya.SollyaObject({"okay": True, "c": c})

                    else:
                        abs_error = abs(c) * epsTarget
                        lsb_index = convert_value_to_bit_weigth(abs_error)
                        cR = int(c * S2**-lsb_index) * S2**lsb_index
                        print("abs_error={}, lsb_index={}, cR={}".format(abs_error, lsb_index, cR))
                        return sollya.SollyaObject({"okay": True, "c": cR})
        else:
            return sollya.SollyaObject({"okay": False})

    @function_trace(4)
    def computeActualAdditionBoundAbsolute(self, out_format, format_lhs, format_rhs):
        """ Return a bound on the absolute error for an addition
           adding an inputFormatA variable by an inputFormatB variable,
           returning an outputFormatVariable.

           The input types are opaque format types.
           The output type is a positive or zero real number
        """
        return out_format.abs_error

    @function_trace(4)
    def computeActualMultiplicationBoundAbsolute(self, out_format, format_lhs, format_rhs):
        """ Return a bound on the absolute error for a multiplication
           multiplying an inputFormatA variable by an inputFormatB variable,
           returning an outputFormatVariable.

           The input types are opaque format types.
           The output type is a positive or zero real number
        """
        return out_format.abs_error

    @function_trace(4)
    def computeBoundPowerAbsolute(self, k, out_format, var_format):
        """ Return a bound on the absolute error for powering the _x_ variable by k,
           which is on a variableFormat format, producing the power result
           on an outputFormat format.

           The input types is an integer k of the power to compute and opaque format types.
           The output type is a positive or zero real number
        """
        # TODO: fix
        delta = out_format.struct.abs_error
        return delta

    @function_trace(3)
    def computeErrorDueToVariableRoundingAbsolute(self, a_f, b_f):
        """ Returns a bound on the absolute error implied by "rounding" _x_,
           which is naturally stored as an af-format variable,
           to a bf-format variable.

           If bf is "larger" (i.e. more precise) than af,
           the error is most probably zero as this just means
           "upconverting" a variable.

           The inputs af and bf are of the "opaque format type".

           The result is a positive or zero real number.
        """
        if a_f.struct.abs_error >= b_f.struct.abs_error:
            return 0
        else:
            delta = b_f.struct.abs_error
            return delta

    @function_trace(4)
    def computeOutputFormatAdditionAbsolute(self, deltaTarget, inputFormatA, inputFormatB):
        """ Returns the output format of an addition that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           absolute error less than or equal to deltaTarget.

           The input and output formats are of type "opaque format type".
           The deltaTarget input is a positive or zero real number.
        """
        # we suppose the operation computed exactly and then truncated
        lhs_format = inputFormatA.struct
        rhs_format = inputFormatB.struct
        add_interval = lhs_format.interval + rhs_format.interval
        signed = lhs_format.fixed_format.signed or rhs_format.fixed_format.signed
        # maximizing integer_size to prevent overflows
        # considering addition interval rather than summing operands' integer size
        integer_size = convert_value_to_bit_weigth(sup(abs(add_interval))) + 1 + (1 if signed else 0)

        # computing digit weight to enforce deltaTarget
        # e.g. deltaTarget=0.125 => delta_digit_index = -3
        if deltaTarget == 0:
            frac_size = max(lhs_format.fixed_format.frac_size, rhs_format.fixed_format.frac_size)
            return MLL_FixedFormat(
                fixed_point_format(integer_size, frac_size, signed=signed),
                add_interval,
                delta_target=deltaTarget,
                exact=True
            )

        else:
            delta_digit_index = int(sollya.floor(sollya.log2(deltaTarget)))
            frac_size = min(max(lhs_format.fixed_format.frac_size, rhs_format.fixed_format.frac_size),
                            -delta_digit_index)
            exact = frac_size >= lhs_format.fixed_format.frac_size and frac_size >= rhs_format.fixed_format.frac_size
            return MLL_FixedFormat(
                fixed_point_format(integer_size, frac_size, signed=signed),
                add_interval,
                delta_target=deltaTarget,
                exact=exact
            )

    @function_trace(4)
    def computeOutputFormatMultiplicationAbsolute(self, deltaTarget, inputFormatA, inputFormatB):
        """ Returns the output format of a multiplication that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           absolute error less than or equal to deltaTarget.

           The input and output formats are of type "opaque format type".
           The deltaTarget input is a positive or zero real number.
        """
        # we suppose the operation computed exactly and then truncated
        lhs_format = inputFormatA.struct
        rhs_format = inputFormatB.struct
        mul_interval = lhs_format.interval * rhs_format.interval
        signed = lhs_format.fixed_format.signed or rhs_format.fixed_format.signed
        # maximizing integer_size to prevent overflows
        # considering multiplication
        # interval rather than summing operands' integer size
        integer_size = convert_value_to_bit_weigth(sup(abs(mul_interval))) + 1 + (1 if signed else 0)

        # computing digit weight to enforce deltaTarget
        # e.g. deltaTarget=0.125 => delta_digit_index = -3
        if deltaTarget == 0:
            frac_size = lhs_format.fixed_format.frac_size + rhs_format.fixed_format.frac_size
            return MLL_FixedFormat(
                fixed_point_format(integer_size, frac_size, signed=signed),
                mul_interval,
                delta_target=deltaTarget,
                exact=True
            )

        else:
            delta_digit_index = int(sollya.floor(sollya.log2(deltaTarget)))
            frac_size = min(lhs_format.fixed_format.frac_size + rhs_format.fixed_format.frac_size,
                            -delta_digit_index)
            exact = frac_size >= (lhs_format.fixed_format.frac_size + rhs_format.fixed_format.frac_size)
            return MLL_FixedFormat(
                fixed_point_format(integer_size, frac_size, signed=signed),
                mul_interval,
                delta_target=deltaTarget,
                exact=exact
            )

    @function_trace(4)
    def computeOutputFormatPowerAbsolute(self, k, deltaTarget, variableFormat):
        """ Returns the output format of a variable _x_ powering by k, knowing
           that the _x_ variable is on a variableFormat format and we
           need to achieve an absolute error of less than deltaTarget.

           The input and output formats are of type "opaque format type".
           The deltaTarget input is a positive or zero real number.
           The k input is an integer.
        """
        # convert k to python integer
        k = int(k) if not k is None else k

        if deltaTarget > 0:
            if k == 1:
                # if k is 1, then the format is the one of Variable verifying
                # epsTarget constraint
                return self.computeNeededVariableFormatAbsolute(None, deltaTarget, variableFormat)
            else:
                final_format = None
                if k in self.power_map:
                    for error_bound, power_format_list in self.power_map[k]:
                        # if record error_bound is less than epsTarget
                        # TODO: implement search for tighter bound
                        if deltaTarget > error_bound:
                            # final_format is last element of power_format_list
                            final_format = power_format_list[-1]
                            if final_format.abs_error < deltaTarget:
                                # there is possibly a smaller format than final_format which
                                # could match epsTarget constraint
                                final_format = None # GR: to be checked ??
                                continue
                            # updating final_format eps_target to make sure it
                            # always keep track of the most constraining
                            # eps_target
                            final_format.delta_target = min(final_format.delta_target, deltaTarget)
                            return final_format
                        # as error_bound are ordered from larger to smaller
                if final_format is None:
                    if k == 2:
                        lhs_k, rhs_k = 1, 1
                    else:
                        lhs_k, rhs_k = self.power_schemes[k]
                    # FIXME: implement dummy format assuming that error target is met
                    #        and delaying to after the resolution of how to compute the power
                    #        effectively
                    # FIXME: refine absolute error budget computation
                    # lhs_error_budget = deltaTarget / (4 * sup(abs(variableFormat.struct.interval)))
                    # rhs_error_budget = deltaTarget / (4 * sup(abs(variableFormat.struct.interval)))
                    # mult_error_budget = deltaTarget / 2

                    # final absolute error
                    # lhs_format = self.computeOutputFormatPowerAbsolute(lhs_k, lhs_error_budget, variableFormat)
                    # rhs_format = self.computeOutputFormatPowerAbsolute(rhs_k, rhs_error_budget, variableFormat)
                    # mult_format = self.computeOutputFormatMultiplicationAbsolute(mult_error_budget, lhs_format, rhs_format)

                    # final error is mult_error + lhs_error * rhs + rhs_error * lhs
                    # lhs_error = self.computeBoundPowerAbsolute(lhs_k, lhs_format, variableFormat)
                    # rhs_error = self.computeBoundPowerAbsolute(rhs_k, rhs_format, variableFormat)
                    # ult_error = self.computeBoundMultiplicationAbsolute(mult_format, lhs_format, rhs_format)


                    # # TODO: should take into account second order error
                    # final_error = lhs_error + rhs_error + mult_error
                    # if final_error > deltaTarget:
                    #     print("possible ERROR in computeOutputFormatPower: failed to meet epsTarget")

                    # final_format = mult_format
                    # final_format.delta_target = deltaTarget

                    # record = (final_error, [lhs_format, rhs_format, final_format])
                    power_interval = variableFormat.struct.interval**k 
                    signed = inf(power_interval) < 0
                    int_size = convert_value_to_bit_weigth(sup(abs(power_interval))) + 1 if signed else 0 
                    frac_size = 1 - convert_value_to_bit_weigth(deltaTarget)

                    power_format = MLL_FixedFormat(
                        fixed_point_format(int_size, frac_size, signed=signed),
                        power_interval,
                        delta_target=deltaTarget
                    )
                    record = (deltaTarget, [None, None, power_format])
                    final_format = power_format

                    if not k in self.power_map:
                        self.power_map[k] = []
                    self.power_map[k].append(record)
                    # sorting list from large error to small errors
                    self.power_map[k].sort(key=(lambda v: v[0]), reverse=True)

                return final_format
        else:
            return sollya.error

    @function_trace(4)
    def computeNeededVariableFormatAbsolute(self, I, deltaTarget, variableFormat):
        """ The function computes the format of the _x_ variable, which
           comes as a variableFormat naturally, when it needs to be known
           up to an absolute error deltaTarget over I.

           I is an interval.
           deltaTarget is a positive or zero real number.
           variableFormat is of type "opaque format type"

           The result is of type "opaque format type".

           When deltaTarget is zero, it makes sense to return variableFormat
           as "rounding" a variable _x_ that is naturally stored as
           a variableFormat-variable to variableFormat does not imply any
           error.
        """
        if deltaTarget == 0:
            return variableFormat
        elif deltaTarget > 0:
            if deltaTarget < variableFormat.abs_error:
                # if expected error is less that what the variable format
                # can provide, we can return it directly
                return variableFormat
            else:
                return self.get_format_from_abs_error(deltaTarget, I)
        else:
            raise NotImplementedError

def build_mll_fixed_context(precision, var_interval):
    # building sollya's structure to be used as opaque type for
    # computeErrorBounds and generateGappa
    pre_mll_context = MLL_FixedContext(precision, var_interval)
    mll_context = sollya.SollyaObject({
        "variableFormat": sollya.SollyaObject(pre_mll_context.variableFormat),

        # legacy methods for relative error
        "computeBoundPower": sollya.SollyaObject(pre_mll_context.computeBoundPower),
        "compareFormats": sollya.SollyaObject(pre_mll_context.compareFormats),
        "computeBoundAddition": sollya.SollyaObject(pre_mll_context.computeBoundAddition),
        "computeBoundMultiplication": sollya.SollyaObject(pre_mll_context.computeBoundMultiplication),
        "computeBoundVariableRounding": sollya.SollyaObject(pre_mll_context.computeBoundVariableRounding),
        "computeOutputFormatAddition": sollya.SollyaObject(pre_mll_context.computeOutputFormatAddition),
        "computeOutputFormatMultiplication": sollya.SollyaObject(pre_mll_context.computeOutputFormatMultiplication),
        "computeConstantFormat": sollya.SollyaObject(pre_mll_context.computeConstantFormat),
        "computeVariableFormat": sollya.SollyaObject(pre_mll_context.computeNeededVariableFormat),
        "computeBoundPower": sollya.SollyaObject(pre_mll_context.computeBoundPower),
        "computeOutputFormatPower": sollya.SollyaObject(pre_mll_context.computeOutputFormatPower),
        "roundConstant": sollya.SollyaObject(pre_mll_context.roundConstant),

        # new methods for absolute error
        "computeBoundAdditionAbsolute": sollya.SollyaObject(pre_mll_context.computeActualAdditionBoundAbsolute),
        "computeBoundMultiplicationAbsolute": sollya.SollyaObject(pre_mll_context.computeActualMultiplicationBoundAbsolute),
        "computeBoundPowerAbsolute": sollya.SollyaObject(pre_mll_context.computeBoundPowerAbsolute),
        "computeBoundVariableRoundingAbsolute": sollya.SollyaObject(pre_mll_context.computeErrorDueToVariableRoundingAbsolute),
        "computeOutputFormatAdditionAbsolute": sollya.SollyaObject(pre_mll_context.computeOutputFormatAdditionAbsolute),
        "computeOutputFormatMultiplicationAbsolute": sollya.SollyaObject(pre_mll_context.computeOutputFormatMultiplicationAbsolute),
        "computeOutputFormatPowerAbsolute": sollya.SollyaObject(pre_mll_context.computeOutputFormatPowerAbsolute),
        "computeVariableFormatAbsolute": sollya.SollyaObject(pre_mll_context.computeNeededVariableFormatAbsolute)
    })
    return pre_mll_context, mll_context

# upgrading ML_Custom_FixedPoint_Format max size
ML_Custom_FixedPoint_Format.MAX_BIT_SIZE = 256
ML_Custom_FixedPoint_Format.POSSIBLES_SIZES = [8, 16, 32, 64, 128, 256]

def fixed_point_format(int_size, frac_size, signed=True):
    return ML_Custom_FixedPoint_Format(int_size, frac_size, signed=signed)


class ImplementPolyMetaFunctionFixed(ML_FunctionBasis):
    """ Metalibm lugdunum meta-function to wrap implementpoly """
    function_name = "implement_poly_fixed"
    def __init__(self,
                 polynomial, epsTarget, interval, function,
                 arg_template=DefaultArgTemplate):
        # initializing I/O precision
        precision = arg_template.precision
        io_precisions = arg_template.input_precisions

        self.polynomial = polynomial
        self.epsTarget = epsTarget
        self.interval = interval
        self.function = function
        self.debug = arg_template.debug

        # initializing base class
        ML_FunctionBasis.__init__(self,
            args=arg_template
        )

        # main precision (used for product operand and default for accumulator)
        self.precision = args.precision

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        default_dict = {
            "precision": fixed_point_format(3, 29, signed=True),
            "input_precisions": [fixed_point_format(1, 31)],
            "target": FixedPointBackend(),
            "accuracy": ML_CorrectlyRounded,
            "output_file": "implement_poly_fixed.c",
            "function_name": "implement_poly_fixed",
            "language": C_Code,
            "passes": [
                ],
        }
        default_dict.update(kw)
        return DefaultArgTemplate(
            **default_dict
        )

    def get_power_eps(self, tree):
        """ list pairs of (k, epsTarget) required for powering """
        if tree.struct.type == "power":
            print(tree.struct)
            return [(int(tree.struct.k), tree.struct.epsTarget)]
        elif tree.struct.type not in ["variable", "constant"]:
            return self.get_power_eps(tree.struct.q) + self.get_power_eps(tree.struct.r)
        else:
            return []

    def get_power_delta(self, tree):
        """ list pairs of (k, delta_target) required for powering """
        if tree.struct.type == "power":
            print(tree.struct)
            return [(int(tree.struct.k), tree.struct.delta)]
        elif tree.struct.type not in ["variable", "constant"]:
            return self.get_power_delta(tree.struct.q) + self.get_power_eps(tree.struct.r)
        else:
            return []

    def get_power_eps_from_powerings(self, powerings, absolute=True):
        """ Generate a map k -> format -> PowerNode which contains a mapping
            between power's index and description of target precision and accuracy
            for this index

            :param powerings: Sollya's powering structure
            :type powerings: SollyaObject
            :param absolute: flag to select absolute / relative error
            :type absolute: bool
        """
        class PowerNode:
            """ descriptor of q required power node """
            def __init__(self, k, out_format, in_format, epsilon=None, delta=None):
                self.k = k
                self.out_format = out_format
                self.in_format = in_format
                self.epsilon = epsilon
                self.delta = delta
        result = {}
        for power_k in powerings:
            k = int(power_k.struct.k)
            result[k] = {}
            for variant in power_k.struct.variants:
                out_format = variant.struct.format.python()
                out_precision = out_format.fixed_format
                if variant.struct.errorBound == sollya.absolute:
                    power_node = PowerNode(
                        k,
                        out_format,
                        variant.struct.inputformat.python(),
                        delta=variant.struct.delta
                    )
                else:
                    power_node = PowerNode(
                        k,
                        out_format,
                        variant.struct.inputformat.python(),
                        epsilon=variant.struct.eps
                    )
                # TODO: add check on in_format equality
                if not out_precision in result[k]:
                    result[k][out_precision] = power_node
                elif (absolute and  result[k][out_precision].delta > out_format.abs_error) or \
                     (not(absolute) and result[k][out_precision].epsilon > out_format.rel_error):
                    # compare if new node improves on accuracy of old node
                    # sharing same k (index) and precision
                    result[k][out_precision] = power_node
                    # else keep unchanged
        return result

    def compute_polytree(self, p, I, power, deltaTarget, mll_context, pre_mll_context):
        """ from abstract polynomial p build a poly-tree
            (a.k.a evaluation scheme)
            which match absolute error constraint deltaTarget """
        found = False
        classic_value = None
        if sollya.degree(p) > 12:
            classic_value = 'estrin'
        print("***** CGPE ... buildTree will be called with p = ", p)
        while not found and classic_value != 'unknown':
            found = True
            Log.report(Log.Verbose, " building Tree with classic_value = {}".format(classic_value))
            PR, powers = polynomial2tree.buildTree(p, classic=classic_value, power=True)
            Log.report(Log.Verbose, "powers = {}".format(powers))
            Log.report(Log.Verbose, " done ")

            # for each power build a table indicating from which pair of
            # sub-power in should be computed
            power_schemes = {}
            for k in powers:
                k_q = 1 if powers[k].q.type == 'variable' else powers[k].q.k
                k_r = 1 if powers[k].r.type == 'variable' else powers[k].r.k
                power_schemes[k] = (k_q, k_r)
            pre_mll_context.power_schemes = power_schemes
            Log.report(Log.Verbose, "power_schemes: {}".format(power_schemes))
            # input()
            PRsollya = PR.convert2SollyaObject()

            P = PRsollya["poly"]

            Log.report(Log.Verbose, " computing Error bounds, P={}, I={}, deltaTarget={}, mll_context={}".format(P, I, deltaTarget, mll_context))
            R = computeErrorBounds(P, I, deltaTarget, sollya.absolute, mll_context)
            Log.report(Log.Verbose, " end of computing Error bounds, R={}".format(R))
            Rdict = dict(R)
            if 'reason' in Rdict:
                print("computeErrorBounds failed with output {}".format(Rdict))
                found = False
            # ...
            FALLBACK_MAP = {
                None: "estrin",
                "estrin": "horner4",
                "horner4": "horner2",
                "horner2": "horner",
                "horner": "unknown"
            }
            if not found:
                if classic_value in FALLBACK_MAP:
                    classic_value = FALLBACK_MAP[classic_value]
                    Log.report(Log.Verbose, " ... fall back to '{}' scheme".format(classic_value))
                else:
                    classic_value = "unknown"
                    PR = None
                    powers = None


        if found and classic_value != None:
            Log.report(Log.Verbose, "  >>> scheme '{}' can be sucessfully used".format(classic_value))
        elif found:
            Log.report(Log.Verbose, "  >>> 'best' scheme can be sucessfully used")
        else:
            Log.report(Log.Verbose, "  >>> not found and classic_value = {} ".format(classic_value))
        return R, PR, powers


    def dump(self, p, level = 0):
        """ display genericimplementpoly polynomial p """
        if 'epsTarget' in dict(p).keys():
            if p.struct.epsTarget != 0:
                print("**"*(level+1)+" "+str(p.struct.type)+"= "+str(sollya.evaluate(sollya.ceil(sollya.log2(p.struct.epsTarget)),1))+" relative" \
                      +" and "+str(p.struct.format.python().fixed_format))
            else:
                print("**"*(level+1)+" "+str(p.struct.type)+"= 0 and "+str(p.struct.format.python().fixed_format))
            if p.struct.type == "multiplication" or p.struct.type == "addition":
                self.dump(p.struct.q, level+1)
                self.dump(p.struct.r, level+1)
        else:
            if p.struct.deltaTarget != 0:
                print("**"*(level+1)+" "+str(p.struct.type)+"= "+str(sollya.evaluate(sollya.ceil(sollya.log2(p.struct.deltaTarget)),1))+" absolute" \
                      +" and "+str(p.struct.format.python().fixed_format))
            else:
                print("**"*(level+1)+" "+str(p.struct.type)+"= 0 and "+str(p.struct.format.python().fixed_format))
            if p.struct.type == "multiplication" or p.struct.type == "addition":
                self.dump(p.struct.q, level+1)
                self.dump(p.struct.r, level+1)


    def generate_scheme(self):
        # extracting main arguments
        p = self.polynomial
        epsTarget = self.epsTarget
        I = self.interval

        # input variable
        input_precision = self.get_input_precision()
        vx = self.implementation.add_input_variable("var_x", input_precision)

        if not isinstance(input_precision, ML_Custom_FixedPoint_Format):
            signed = inf(I) < 0
            int_size = convert_value_to_bit_weigth(sup(abs(I))) + (1 if signed else 0)
            # we keep the same number of bits for float and fixed inputs
            frac_size = input_precision.get_bit_size() - int_size
            fixed_precision = fixed_point_format(int_size, frac_size, signed=signed)

            vx = Conversion(vx, precision=fixed_precision)
        else:
            fixed_precision = input_precision


        pre_mll_context, mll_context = build_mll_fixed_context(fixed_precision, I)

        R, PR, powers = self.compute_polytree(p, I, True, epsTarget, mll_context, pre_mll_context)
        Log.report(Log.Verbose, "powers = {}".format(powers))

        # debug display of poly structure
        #self.dump(R.struct.poly)

        if PR != None and PR.convert2SollyaObject()["okay"] == True:
            self.metalibm_p = R.struct.poly.struct.p
            Log.report(Log.Verbose, "R.struct.powerings = {}", R.struct.powerings)

            # list of pairs (k, eps target) used in R.struct.poly
            #eps_power_list = self.get_power_eps(R.struct.poly)

            # map of k -> list of (epsValue, ml_format for power output)
            # power map has also been simplified: for each k and each format only keep the most
            # accurate node
            # epsTargets are really absolute errors => deltaTargets
            epsTargets = self.get_power_eps_from_powerings(R.struct.powerings)
            Log.report(Log.Verbose, "eps from powerings = {}".format(epsTargets))

            map_X_k = []
            map_X_k_okay = True
            for k in epsTargets:
                # looking for a poly struct to implement x^k for p
                for idx, power_precision in enumerate(epsTargets[k]):
                    e = epsTargets[k][power_precision].delta
                    label = "pow_k{}_{}".format(k, idx)
                    # possible supernumeary computation of X^k to extract its poly object
                    # could not it be extract from main computeErrorBounds ?
                    power_poly = powers[k].convert2SollyaObject()["poly"]
                    Log.report(Log.Verbose, "power_poly={}, I={}, e={}", power_poly, I, e)
                    power_R = computeErrorBounds(sollya.SollyaObject(power_poly), I, e, sollya.absolute, mll_context)
                    Log.report(Log.Verbose, "processing {}, e={}, epsTarget={}, result precision is {}".format(label, e, epsTargets[k][power_precision].out_format.eps_target, power_R.struct.poly.struct.format.python()))

                    if str(power_R.struct.okay) == "false":
                        print("FAILED to compute error bound for power graph k={}, idx={}, epsilon={}, power_precision={}, I={}".format(k, idx, e, power_precision, I))
                        print(str(power_R.struct.reason))
                        raise Exception()
                    map_X_k.append((label, power_R))
                    # accumulating okay value
                    map_X_k_okay = (map_X_k_okay and power_R.struct.okay)


            # listing expression for X^k
            map_X_k_poly = [P_R.struct.poly for label, P_R in map_X_k]
            parser = ml_fixed_wrapper.SollyaFixedConverter(vx, lutetia_tag_mode=False) #not(self.debug))
            pre_poly_scheme = parser.build(R.struct.poly, map_X_k_poly)

            main_statement = Statement()

            if len(self.value_test) > 0:
                input_mapping = {vx: self.value_test[0][0]}
                error_eval_map = runtime_error_eval.generate_error_eval_graph(pre_poly_scheme, input_mapping)
                for node in depth_node_ordering(pre_poly_scheme, input_mapping)[::-1]:
                    if node in error_eval_map:
                        main_statement.add(error_eval_map[node])

            # expanding meta-blocks
            poly_scheme = pre_poly_scheme
            result = Return(Conversion(poly_scheme, precision=self.precision))

            main_statement.add(result)

            self.RESULT_TAG = "result_o"

            return main_statement
        else:
            Log.report(Log.Error, "PR return okay=False")

    def numeric_emulate(self, vx):
        """ Numerical emulation using self.function """
        return self.function(vx)

if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(
        default_arg=ImplementPolyMetaFunctionFixed.get_default_args()
    )
    arg_template.get_parser().add_argument(
         "--eps-target", dest="eps_target", default=S2**-60, type=sollya.parse,
        action="store", help="define error (epsilon) target")
    arg_template.get_parser().add_argument(
         "--function", default=sollya.exp(sollya.x), type=sollya.parse,
        action="store", help="define function to approximate")
    arg_template.get_parser().add_argument(
         "--range", default=Interval(-2**-4,2**-4), type=sollya.parse,
        action="store", help="define approximation interval")
    arg_template.get_parser().add_argument(
         "--monomials", default=8, type=eval,
        action="store", help="define list of polynomial monomials")
    arg_template.get_parser().add_argument(
         "--formats", default=[1, 70, 60, 60, 50, 50, 50, 50, 50], type=eval,
        action="store", help="define list of coefficient format")
    arg_template.get_parser().add_argument(
         "--proof", default="poly.gappa", type=str,
        action="store", help="define output Gappa proof file")
    arg_template.get_parser().add_argument(
         "--trace", default=False, const=True,
        action="store_const", help="enable verbose trace")
    arg_template.get_parser().add_argument(
         "--lutetia-mode", default=False, const=True,
        action="store_const", help="select lutetia API for output function")

    args = arg_template.arg_extraction()

    ft_module.enable_trace = args.trace

    f = args.function
    I = args.range
    monomials = args.monomials

    # overloading auto-test range to match approximation validity range
    # args.auto_test_range = I

    sollya.settings.prec = 3000;

    monomial_formats = args.formats
    monomials = args.monomials

    p = sollya.fpminimax(f, monomials, monomial_formats, I)
    print("f={}".format(f))
    print("p={}".format(p))
    sollya.settings.prec = 300;

    deltaTarget = sollya.dirtyinfnorm(p-f, I)

    print("I=%s" % I)
    print("deltaTarget={:.3e}".format(float(deltaTarget)))

    implement_poly = ImplementPolyMetaFunctionFixed(p, deltaTarget, I, f, args)
    implement_poly.gen_implementation()

