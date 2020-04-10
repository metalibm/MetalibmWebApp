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
    get_sollya_from_long, ML_Binary64, ML_Void
)
from metalibm_core.core.ml_complex_formats import ML_Pointer_Format
from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.ml_operations import (
    Return, Conversion, Addition, Multiplication,
    BuildFromComponent, ReferenceAssign, Dereference, Statement
)
from metalibm_core.utility.log_report import Log

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)


from metalibm_core.code_generation.vhdl_backend import VHDLBackend
from metalibm_core.core.ml_entity import ML_EntityBasis, DefaultEntityArgTemplate

from metalibm_core.utility.ml_template import (
    ML_EntityArgTemplate
)
from metalibm_core.utility.log_report import Log

from metalibm_core.core.precisions import ML_Faithful, ML_CorrectlyRounded
from metalibm_core.core.ml_formats import ML_Bool


from metalibm_core.core.ml_hdl_format import (
    ML_StdLogicVectorFormat, ML_StdLogic, fixed_point,
)
from metalibm_core.code_generation.code_constant import VHDL_Code
from metalibm_core.core.ml_hdl_operations import (
    BitSelection, Signal, PlaceHolder,
    equal_to, Concatenation, SubSignalSelection
)

from metalibm_core.utility.rtl_debug_utils import (
        debug_fixed, debug_std,
)
from metalibm_core.utility.ml_template import hdl_precision_parser


import ml_hdl_wrapper

sollya.settings.verbosity = 2

try:
    sollya_scripts_path = os.environ["GENERICIMPLEMENTPOLY_ROOT"]
except KeyError:
    sollya_scripts_path = "./"

print(sollya_scripts_path + "/polynomials.sol")

sollya.execute(sollya_scripts_path + "/polynomials.sol")

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
    """ return the MSB and LSB indexes to encore value without error"""
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
    def __init__(self, fixed_format, interval, eps_target, exact=False):
        # Metalibm's fixed-point format to be used
        self.fixed_format = fixed_format = fixed_format
        # interval: range of the node values
        self.interval = interval
        # initial requirement for epsilon target
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

    def _sollya_(self):
        """ convert to SollyaObject (by wrapping in an externaldata
            container) """
        return sollya.externaldata(self)

    def __str__(self):
        """ convert to string (used for comprehensive content display) """
        return "(fixed_format={}, interval={}, eps_target={}, exact={})".format(
            str(self.fixed_format), str(self.interval), str(self.eps_target), self.exact
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
            0,
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
        """ return a fixed-point format able to contains max_value, with
            at least accuracy @p accuracy """
        assert abs_error > 0
        signed = inf(interval) < 0
        integer_size = int(sollya.ceil(sollya.log2(sup(abs(interval)))))
        frac_size = -int(sollya.floor(sollya.log2(abs_error)))
        return fixed_point(integer_size, frac_size, signed=signed)

    def get_ml_fixed_format_from_rel_error(self, interval, rel_error):
        assert rel_error > 0
        # relative_error = absolute_error / min(value)
        min_value = inf(abs(interval))
        absolute_error = abs(rel_error) * min_value
        fixed_format = self.get_ml_fixed_format_from_abs_error(interval, absolute_error)
        return fixed_format


    def get_format_from_abs_error(self, abs_error, eps_target=None, interval=None):
        low_level_format = self.get_ml_fixed_format_from_abs_error(interval, abs_error)
        return MLL_FixedFormat(low_level_format, interval, eps_target, exact=(abs_error==0.0))

    def get_format_from_rel_error(self, rel_error, eps_target=None, interval=None):
        low_level_format = self.get_ml_fixed_format_from_rel_error(interval, rel_error)
        return MLL_FixedFormat(low_level_format, interval, eps_target, exact=(rel_error==0.0))


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
        if epsTarget == 0 or 0 in add_interval:
            # TODO: format computation could be optimized
            signed = lhs_format.fixed_format.signed or rhs_format.fixed_format.signed
            integer_size = max(lhs_format.fixed_format.integer_size, rhs_format.fixed_format.integer_size) + 1
            frac_size = max(lhs_format.fixed_format.frac_size, rhs_format.fixed_format.frac_size)
            return MLL_FixedFormat(
                fixed_point(integer_size, frac_size, signed=signed),
                add_interval,
                epsTarget,
                exact=True
            )

        else:
            return self.get_format_from_rel_error(epsTarget, epsTarget, add_interval)



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
                fixed_point(integer_size, frac_size, signed=signed),
                mul_interval,
                epsTarget,
                exact=True
            )

        else:
            return self.get_format_from_rel_error(epsTarget, epsTarget, mul_interval)

    @function_trace_hex_2
    def computeConstantFormat(self, c):
        if c == 0:
            # default to single bit value precision
            return MLL_FixedFormat(
                fixed_point(1, 0, signed=False),
                Interval(0),
                0,
                exact=True
            )
        else:
            signed = c < 0
            msb_index, lsb_index = get_value_indexes(abs(c))
            print("computeConstantFormat c={}, msb_index={}, lsb_index={}".format(c, msb_index, lsb_index))
            return MLL_FixedFormat(
                fixed_point(msb_index + 1, -lsb_index, signed=signed),
                Interval(c),
                0,
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
                    lhs_format = self.computeOutputFormatPower(lhs_k, epsTarget / 4, variableFormat)
                    rhs_format = self.computeOutputFormatPower(rhs_k, epsTarget / 4, variableFormat)
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


def build_mll_fixed_context(precision, var_interval):
    # building sollya's structure to be used as opaque type for
    # computeErrorBounds and generateGappa
    pre_mll_context = MLL_FixedContext(precision, var_interval)
    mll_context = sollya.SollyaObject({
        "computeBoundPower": sollya.SollyaObject(pre_mll_context.computeBoundPower),
        "variableFormat": sollya.SollyaObject(pre_mll_context.variableFormat),
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
    })
    return pre_mll_context, mll_context


class ImplementPolyMetaEntity(ML_EntityBasis):
    """ Metalibm lugdunum meta-function to wrap implementpoly """
    entity_name = "implement_poly"
    def __init__(self,
                 polynomial, epsTarget, interval, function,
                 arg_template = DefaultEntityArgTemplate,
                 precision = fixed_point(3, 29, signed=True),
                 accuracy    = ML_Faithful,
                 debug_flag = False,
                 target = VHDLBackend(),
                 output_file = "implement_poly.vhd",
                 entity_name = "implement_poly",
                 language = VHDL_Code,
                 acc_prec = None,
                 pipelined = False):
        # initializing I/O precision
        precision = arg_template.precision
        io_precisions = arg_template.input_precisions

        self.polynomial = polynomial
        self.epsTarget = epsTarget
        self.interval = interval
        self.function = function
        self.debug = arg_template.debug

        # initializing base class
        ML_EntityBasis.__init__(self,
            base_name = "implement_poly",
            entity_name = entity_name,
            output_file = output_file,

            io_precisions = io_precisions,
            abs_accuracy = None,

            backend = target,

            debug_flag = debug_flag,
            language = language,
            arg_template = arg_template
        )

        self.accuracy    = accuracy
        # main precision (used for product operand and default for accumulator)
        self.precision = precision
        # enable operator pipelining
        self.pipelined = pipelined

    ## default argument template generation
    @staticmethod
    def get_default_args(**kw):
        default_dict = {
            "precision": fixed_point(3, 29, signed=True),
            "input_precisions": [fixed_point(1, 31)],
            "target": VHDLBackend(),
            "accuracy": ML_CorrectlyRounded,
            "output_file": "implement_poly.vhd",
            "entity_name": "implement_poly",
            "language": VHDL_Code,
            "pipelined": False,
            "dummy_mode": False,
            "passes": [
                ("beforepipelining:size_datapath"),
                ("beforepipelining:rtl_legalize"),
                ("beforepipelining:unify_pipeline_stages"),
                ],
        }
        default_dict.update(kw)
        return DefaultEntityArgTemplate(
            **default_dict
        )

    def get_power_eps(self, tree):
        """ list pairs of (k, epsTarget) required for powering """
        if tree.struct.type == "power":
            return [(int(tree.struct.k), tree.struct.epsTarget)]
        elif tree.struct.type not in ["variable", "constant"]:
            return self.get_power_eps(tree.struct.q) + self.get_power_eps(tree.struct.r)
        else:
            return []

    def get_power_eps_from_powerings(self, powerings):
        class PowerNode:
            """ descriptor of q required power node """
            def __init__(self, k, out_format, in_format, epsilon):
                self.k = k
                self.out_format = out_format
                self.in_format = in_format
                self.epsilon = epsilon
        result = {}
        for power_k in powerings:
            k = int(power_k.struct.k)
            result[k] = {}
            for variant in power_k.struct.variants:
                out_format = variant.struct.format.python()
                out_precision = out_format.fixed_format
                power_node = PowerNode(
                    k,
                    out_format,
                    variant.struct.inputformat.python(),
                    variant.struct.eps
                )
                # TODO: add check on in_format equality
                if not out_precision in result[k]:
                    result[k][out_precision] = power_node
                elif result[k][out_precision].epsilon > out_format.rel_error:
                    # compare if new node improves on accuracy of old node
                    # sharing same k (index) and precision
                    result[k][out_precision] = power_node
                    # else keep unchanged
        return result

    def compute_polytree(self, p, I, power, epsTarget, mll_context, pre_mll_context):
        found = False
        classic_value = None
        if sollya.degree(p) > 12:
            classic_value = 'estrin'
        print("***** CGPE ... buildTree will be called with p = ", p)
        while not found and classic_value != 'unknown':
            found = True
            print(" building Tree with classic_value = {}".format(classic_value))
            PR, powers = polynomial2tree.buildTree(p, classic=classic_value, power=True)
            print("powers = {}".format(powers))
            print(" done ")

            # for each power build a table indicating from which pair of
            # sub-power in should be computed
            power_schemes = {}
            for k in powers:
                k_q = 1 if powers[k].q.type == 'variable' else powers[k].q.k
                k_r = 1 if powers[k].r.type == 'variable' else powers[k].r.k
                power_schemes[k] = (k_q, k_r)
            pre_mll_context.power_schemes = power_schemes
            print("power_schemes: {}".format(power_schemes))
            # input()
            PRsollya = PR.convert2SollyaObject()
            if PRsollya["okay"] == True:
                P = PRsollya["poly"]
                print(" trying to detect possible cancellations containsCancellations({}, {})".format(P, I))
                CL = dict(containsCancellations(P, I))
                print("    = {}".format(CL))

                if CL["okay"] == True:
                    if CL["res"] == True:
                        print("  >>> the given tree contains cancellations: ", CL["cancellations"])
                        found = False
                else:
                    print("  >>> could not determine if the given tree contains cancellations")
            print(" computing Error bounds")
            R = computeErrorBounds(P, I, epsTarget, mll_context)
            print(" end of computing Error bounds")
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
                    print(" ... fall back to '{}' scheme".format(classic_value))
                else:
                    classic_value = "unknown"
                    PR = None
                    powers = None


        if found and classic_value != None:
            print("  >>> scheme '{}' can be sucessfully used".format(classic_value))
        elif found:
            print("  >>> 'best' scheme can be sucessfully used")
        else:
            print("  >>> not found and classic_value = {} ".format(classic_value))
        return R, PR, powers


    def dump(self, p, level = 0):
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
        # input variable
        vx = self.implementation.add_input_variable("var_x", self.get_input_precision())

        # extracting main arguments
        p = self.polynomial
        epsTarget = self.epsTarget
        I = self.interval

        pre_mll_context, mll_context = build_mll_fixed_context(self.get_input_precision(), I)

        R, PR, powers = self.compute_polytree(p, I, True, epsTarget, mll_context, pre_mll_context)
        print("powers = {}".format(powers))

        self.dump(R.struct.poly)

        if PR != None and PR.convert2SollyaObject()["okay"] == True:
            self.metalibm_p = R.struct.poly.struct.p
            print("R.struct.powerings = {}".format(R.struct.powerings))

            # list of pairs (k, eps target) used in R.struct.poly
            eps_power_list = self.get_power_eps(R.struct.poly)

            # map of k -> list of (epsValue, ml_format for power output)
            # power map has also been simplified: for each k and each format only keep the most
            # accurate node
            epsTargets = self.get_power_eps_from_powerings(R.struct.powerings)
            print("eps from powerings = {}".format(epsTargets))

            map_X_k = []
            map_X_k_okay = True
            for k in epsTargets:
                # looking for a poly struct to implement x^k for p
                for idx, power_precision in enumerate(epsTargets[k]):
                    e = epsTargets[k][power_precision].epsilon
                    label = "pow_k{}_{}".format(k, idx)
                    # possible supernumeary computation of X^k to extract its poly object
                    # could not it be extract from main computeErrorBounds ?
                    power_poly = powers[k].convert2SollyaObject()["poly"]
                    power_R = computeErrorBounds(sollya.SollyaObject(power_poly), I, e, mll_context)
                    print("processing {}, e={}, epsTarget={}, result precision is {}".format(label, e, epsTargets[k][power_precision].out_format.eps_target, power_R.struct.poly.struct.format.python()))

                    if str(power_R.struct.okay) == "false":
                        print("FAILED to compute error bound for power graph k={}, idx={}, epsilon={}, power_precision={}, I={}".format(k, idx, e, power_precision, I))
                        print(str(power_R.struct.reason))
                        raise Exception()
                    map_X_k.append((label, power_R))
                    # accumulating okay value
                    map_X_k_okay = (map_X_k_okay and power_R.struct.okay)


            # listing expression for X^k
            map_X_k_poly = [P_R.struct.poly for label, P_R in map_X_k]
            parser = ml_hdl_wrapper.SollyaHDLConverter(vx, lutetia_tag_mode=False)
            pre_poly_scheme = parser.build(R.struct.poly, map_X_k_poly)

            # expanding meta-blocks
            poly_scheme = pre_poly_scheme
            result = Conversion(poly_scheme, precision=self.precision)

            self.RESULT_TAG = "result_o"

            self.implementation.add_output_signal(
                self.RESULT_TAG, result
            )

            return [self.implementation]
        else:
            Log.report(Log.Error, "PR return okay=False")

    def numeric_emulate(self, io_map):
        """ Numerical emulation using self.function """
        print(io_map)
        vx = io_map["var_x"]
        return {self.RESULT_TAG: self.function(vx)}

if __name__ == "__main__":
    arg_template = ML_EntityArgTemplate(
        default_entity_name="implement_poly",
        default_output_file="implement_poly.vhd",
        default_arg=ImplementPolyMetaEntity.get_default_args()
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
         "--monomials", default=[0, 1, 2, 3, 4, 5], type=sollya.parse,
        action="store", help="define list of polynomial monomials")
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
    Itmp = Interval(2**-100, sup(I))
    monomials = args.monomials

    sollya.settings.prec = 3000;
    F = [1, sollya.doubledouble, sollya.doubledouble]+[sollya.binary64]*3;

    p = sollya.fpminimax(f, monomials, F, Itmp);
    sollya.settings.prec = 300;

    epsTarget = sollya.dirtyinfnorm(p/f-1, I)

    print("I=%s" % I)
    print("epsTarget=%s" % epsTarget)

    implement_poly = ImplementPolyMetaEntity(p, epsTarget, I, f, args)
    implement_poly.gen_implementation()

