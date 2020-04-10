# -*- coding: utf-8 -*-
import sys

import sollya, polynomial2tree
import os
from sollya import Interval, sup

# constant for sollya's version of number 2
S2 = sollya.SollyaObject(2)
sollya.settings.display = sollya.decimal

# local module to trace function easily
from function_trace import function_trace, function_trace_hex_2
import function_trace as ft_module

from metalibm_core.core.ml_formats import (
    ML_DoubleDouble, ML_TripleDouble, ML_Binary64, ML_Void, ML_FP_MultiElementFormat
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

from metalibm_core.opt.ml_blocks import (
    MP_Node, get_Addition_MB_compatible_list,
    get_Multiplication_MB_compatible_list,
    get_MB_cost, MB_Normalize_33_td, MB_Identity,
    MB_Wrapper_2Op, MB_PostWrapper_2Op, is_tri_limb_precision,
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

import ml_wrapper

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


def get_accuracy_from_epsilon(epsilon):
    """ convert a numerical relative error into
        a number of accuracy bits """
    return sollya.floor(-sollya.log2(abs(epsilon)))

class MLL_Format:
    """ Container for all information required to
        describe a node format as infered by MLL_Context """
    def __init__(self, mp_node, meta_block, eps_target):
        # support format
        self.mp_node = mp_node
        # upper bound on the relative error, eps <= 2^-self.accuracy
        self.accuracy = get_accuracy_from_epsilon(mp_node.epsilon)
        # meta_block used to generate this node
        self.meta_block = meta_block
        # initial requirement for epsilon target
        self.eps_target = eps_target

    def _sollya_(self):
        """ convert to SollyaObject (by wrapping in an externaldata
            container) """
        return sollya.externaldata(self)

    def __str__(self):
        """ convert to string (used for comprehensive content display) """
        return "(mp_node={}, accuracy={}, meta_block={}, eps_target={})".format(
            str(self.mp_node), str(self.accuracy), str(self.meta_block), str(self.eps_target)
        )

    @property
    def struct(self):
        """ Handler for compatibility with sollya's record """
        return self

def get_MB_compatible_list(OpClass, lhs, rhs):
    """ return a list of metablock instance implementing an operation of
        type OpClass and compatible with format descriptor @p lhs and @p rhs
        """
    fct_map = {
        Addition: get_Addition_MB_compatible_list,
        Multiplication: get_Multiplication_MB_compatible_list
    }
    return fct_map[OpClass](lhs, rhs)

class MLL_Context:
    """ Class to wrapper the opaque type which bridges computeErrorBounds
        and metalibm Lugdunum """
    def __init__(self, var_format, var_interval):
        # dummy 1000-bit accuracy
        if isinstance(var_format, ML_FP_MultiElementFormat):
                # no overlap
            overlaps = [S2**-var_format.get_limb_precision(0).get_mantissa_size()] * (var_format.limb_num-1)
        else:
            overlaps = []
        self.variableFormat = MLL_Format(
            MP_Node(
                var_format,
                0,
                overlaps,
                interval=var_interval
            ),
            None,
            0
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

    def get_ml_format_from_accuracy(self, accuracy):
        """ return a tuple (ml_format, limb_diff_factors) which
            best fit accuracy requirement """
        if accuracy <= 53:
            return ML_Binary64, []
        elif accuracy <= 106:
            return ML_DoubleDouble, [S2**-53]
        elif accuracy <= 159:
            return ML_TripleDouble, [S2**-53, S2**-53]
        else:
            return None, []

    def get_smaller_format_min_error(self, ml_format):
        """ return the maximal accuracy / minimal error
            of the format just before @p ml_format in term
            of size """
        MIN_ERROR_MAP = {
            ML_Binary64: -sollya.log2(0), # no format smaller than ML_Binary64
            ML_DoubleDouble: S2**-53,
            ML_TripleDouble: S2**-106
        }
        return MIN_ERROR_MAP[ml_format]


    def get_format_from_accuracy(self, accuracy, eps_target=None, interval=None):
        """ return an MLL_Format object with """ 
        # TODO: manage ML_Binary32
        ml_format, limb_diff_factors = self.get_ml_format_from_accuracy(accuracy)
        if ml_format is None:
            return sollya.error
        else:
            eps_target = S2**-accuracy if eps_target is None else eps_target
            return MLL_Format(MP_Node(ml_format, eps_target, limb_diff_factors, interval), None, eps_target)

    @function_trace(4)
    def computeBoundAddition(self, out_format, input_format_lhs, input_format_rhs):
        eps = sollya.SollyaObject(out_format.struct.meta_block.local_relative_error_eval(
            input_format_lhs.struct.mp_node, input_format_rhs.struct.mp_node
        ))
        return eps

    @function_trace(4)
    def computeBoundMultiplication(self, out_format, input_format_lhs, input_format_rhs):
        eps = sollya.SollyaObject(out_format.struct.meta_block.local_relative_error_eval(
            input_format_lhs.struct.mp_node, input_format_rhs.struct.mp_node
        ))
        return eps

    @function_trace(4)
    def computeBoundPower(self, k, out_format, var_format):
        # TODO: fix
        epsilon = out_format.struct.mp_node.epsilon
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
        if format_a.struct.accuracy < format_b.struct.accuracy:
            return -1
        elif format_a.struct.accuracy == format_b.struct.accuracy:
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
        if format_a.struct.accuracy <= format_b.struct.accuracy:
            return 0
        else:
            return S2**(-format_b.struct.accuracy)# + 1)

    def compute_output_format(self, OpClass, epsTarget, lhs, rhs):
        def get_renormalize_MB(op):
            """ determine if there exists a renormalization meta-block
                compatible with @p op """
            for mb in [MB_Normalize_33_td]:
                if mb.check_input_descriptors(op) and op.limb_diff_factor[0] >= self.LIMB_DIFF_RENORM_THRESHOLD:
                    return mb
            return None
        ft_compatible_list = get_MB_compatible_list(OpClass, lhs, rhs)
        def check_mb_error_target(mb, epsTarget, lhs, rhs):
            return mb.local_relative_error_eval(lhs, rhs) <= epsTarget
        valid_eps_list = [mb for mb in ft_compatible_list if check_mb_error_target(mb, epsTarget, lhs, rhs)]
        renormalize_rhs = False
        renormalize_lhs = False
        if not len(valid_eps_list):
            # trying to insert a normalize to see if a compatible meta-block appears
            # left-hand side
            renorm_MB_lhs = get_renormalize_MB(lhs)
            if renorm_MB_lhs:
                ft_module.trace_print("Renormalizing from {} ".format(lhs))
                lhs = renorm_MB_lhs.get_output_descriptor(lhs)
                ft_module.trace_print("  to {}".format(lhs))
                renormalize_lhs = True
            # right hand side
            renorm_MB_rhs = get_renormalize_MB(rhs)
            if renorm_MB_rhs:
                ft_module.trace_print("Renormalizing {} to".format(rhs))
                rhs = renorm_MB_rhs.get_output_descriptor(rhs)
                ft_module.trace_print("  to {}".format(rhs))
                renormalize_rhs = True

            ft_compatible_list = get_MB_compatible_list(OpClass, lhs, rhs)
            valid_eps_list = [mb for mb in ft_compatible_list if check_mb_error_target(mb, epsTarget, lhs, rhs)]

            if not len(valid_eps_list):
                return sollya.error
        meta_block = min(valid_eps_list, key=get_MB_cost)
        out_format = meta_block.get_output_descriptor(lhs, rhs, global_error=False)
        if renormalize_lhs or renormalize_rhs:
            lhs_block = renorm_MB_lhs if renormalize_lhs else MB_Identity
            rhs_block = renorm_MB_rhs if renormalize_rhs else MB_Identity
            return MLL_Format(out_format, MB_Wrapper_2Op(meta_block, lhs_block, rhs_block), epsTarget)
        else:
            if is_tri_limb_precision(out_format.precision) and out_format.limb_diff_factor[0] >= self.LIMB_DIFF_RENORM_THRESHOLD:
                # reduction in precision is too big => Normalize insertion
                renorm_MB = get_renormalize_MB(out_format)
                ft_module.trace_print("Renormalizing {} to".format(out_format))
                out_format = renorm_MB.get_output_descriptor(lhs)
                ft_module.trace_print(" {}".format(out_format))
                meta_block = MB_PostWrapper_2Op(meta_block, renorm_MB)
            return MLL_Format(out_format, meta_block, epsTarget)


    @function_trace(4)
    def computeOutputFormatAddition(self, epsTarget, inputFormatA, inputFormatB):
        """
           Returns the output format of an addition  that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """

        lhs = inputFormatA.struct.mp_node
        rhs = inputFormatB.struct.mp_node
        return self.compute_output_format(Addition, epsTarget, lhs, rhs)


    @function_trace(4)
    def computeOutputFormatMultiplication(self, epsTarget, inputFormatA, inputFormatB):
        """ Returns the output format of a multiplication that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        lhs = inputFormatA.struct.mp_node
        rhs = inputFormatB.struct.mp_node
        return self.compute_output_format(Multiplication, epsTarget, lhs, rhs)

    @function_trace_hex_2
    def computeConstantFormat(self, c):
        if c == 0:
            # default to double precision
            return self.get_format_from_accuracy(53, eps_target=0, interval=Interval(c))
        else:
            accuracy = 0
            cN = c
            limb_num = 0
            while cN != 0 and limb_num < self.LIMB_NUM_THRESHOLD:
                cR = sollya.round(cN, sollya.binary64, sollya.RN)
                cN = cN - cR
                accuracy += 53
                limb_num += 1
            if accuracy > 159 or limb_num > self.LIMB_NUM_THRESHOLD:
                eps_target = S2**--accuracy
                accuracy = 159
            else:
                eps_target = 0 if cN == 0 else S2**-accuracy
            return self.get_format_from_accuracy(accuracy, eps_target=eps_target, interval=Interval(c))


    @function_trace(4)
    def computeNeededVariableFormat(self, I, epsTarget, variableFormat):
        if epsTarget > 0:
            # TODO: fix to support ML_Binary32
            if epsTarget >= S2**-53 or variableFormat.mp_node.precision is ML_Binary64:
                # FIXME: default to minimal precision ML_Binary64
                return variableFormat
            else:
                target_accuracy = sollya.ceil(-sollya.log2(epsTarget))
                target_format   = self.get_format_from_accuracy(target_accuracy, eps_target=epsTarget, interval=variableFormat.struct.mp_node.interval)
                if target_format.struct.mp_node.precision.get_bit_size() < variableFormat.struct.mp_node.precision.get_bit_size():
                    return target_format
                else:
                    # if variableFormat is smaller (less bits) and more accurate
                    # then we use it
                    return variableFormat
        else:
            return variableFormat

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
                            if self.get_smaller_format_min_error(final_format.mp_node.precision) < epsTarget:
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
                    cR = sollya.round(c, sollya.binary64, sollya.RN)
                    limb_num = 1
                    while abs(cR / c - 1) > epsTarget and limb_num < self.LIMB_NUM_THRESHOLD:
                        cN = sollya.round(c - cR, sollya.binary64, sollya.RN)
                        cR += cN
                        limb_num += 1
                    if limb_num > self.LIMB_NUM_THRESHOLD or abs(cR / c - 1) > epsTarget:
                        return sollya.SollyaObject({"okay": False})
                    else:
                        return sollya.SollyaObject({"okay": True, "c": cR})
        else:
            return sollya.SollyaObject({"okay": False})


def build_mll_context(precision, var_interval):
    # building sollya's structure to be used as opaque type for
    # computeErrorBounds and generateGappa
    pre_mll_context = MLL_Context(precision, var_interval)
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


class ImplementPolyMetaFunction(ML_FunctionBasis):
    """ Metalibm lugdunum meta-function to wrap implementpoly """
    function_name = "implementpoly"

    def __init__(self, polynomial, epsTarget, interval, function = None, arg_template = DefaultArgTemplate):
        ML_FunctionBasis.__init__(self, args=arg_template)

        self.polynomial = polynomial
        self.epsTarget = epsTarget
        self.interval = interval
        self.function = function
        self.proof = arg_template.proof
        #
        self.function_name = arg_template.function_name
        self.input_format = arg_template.input_precisions
        self.lutetia_mode = arg_template.lutetia_mode
        self.debug = arg_template.debug
        self.start_scheme = arg_template.start_scheme

    @staticmethod
    def get_default_args(**args):
        """ Generate a default argument structure """
        default_args = {
            "precision": ML_TripleDouble,
            "input_precisions": [ML_Binary64],
            "accuracy": ML_CorrectlyRounded,
            "target": GenericProcessor(),
            "output_file": "my_implementpoly.c",
            "function_name": "my_poly",
            "language": C_Code,
            "proof": "proof.gappa",
            "lutetia_mode": False,
            "passes": ["start:basic_legalization", "start:expand_multi_precision"],
            "vector_size": 1,
            "arity": 1,
            "start_scheme": None,
        }
        default_args.update(args)
        return DefaultArgTemplate(**default_args)

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
                out_precision = out_format.mp_node.precision
                power_node = PowerNode(
                    k,
                    out_format,
                    variant.struct.inputformat.python(),
                    variant.struct.eps
                )
                # TODO: add check on in_format equality
                if not out_precision in result[k]:
                    result[k][out_precision] = power_node
                elif result[k][out_precision].epsilon > out_format.mp_node.epsilon:
                    # compare if new node improves on accuracy of old node
                    # sharing same k (index) and precision
                    result[k][out_precision] = power_node
                    # else keep unchanged
        return result

    def compute_polytree(self, p, I, power, epsTarget, mll_context, pre_mll_context, start_scheme=None):
        found = False
        classic_value = start_scheme

        FALLBACK_MAP = {
            None: "estrin",
            "estrin": "horner4",
            "horner4": "horner2",
            "horner2": "horner",
            "horner": "unknown"
        }

        if sollya.degree(p) > 12 and classic_value is None:
            # force estrin scheme for large polynomial (to avoid combinatorial explosion)
            classic_value = 'estrin'
        print("***** CGPE ... buildTree will be called with p = ", p)
        while not found and classic_value != 'unknown':
            found = True
            print(" building Tree with classic_value = {}".format(classic_value))
            PR, powers = polynomial2tree.buildTree(p, classic=classic_value, power=True)
            print("powers = {}".format(powers))

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
            print("compute_polytree: computing Error bounds")
            R = computeErrorBounds(P, I, epsTarget, sollya.relative, mll_context)
            print(" end of computing Error bounds")
            Rdict = dict(R)
            if 'reason' in Rdict:
                print("computeErrorBounds failed with output {}".format(Rdict))
                found = False
            # ...
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
                      +" and "+str(p.struct.format.python().mp_node.precision))
            else:
                print("**"*(level+1)+" "+str(p.struct.type)+"= 0 and "+str(p.struct.format.python().mp_node.precision))
            if p.struct.type == "multiplication" or p.struct.type == "addition":
                self.dump(p.struct.q, level+1)
                self.dump(p.struct.r, level+1)
        else:
            if p.struct.deltaTarget != 0:
                print("**"*(level+1)+" "+str(p.struct.type)+"= "+str(sollya.evaluate(sollya.ceil(sollya.log2(p.struct.deltaTarget)),1))+" absolute" \
                      +" and "+str(p.struct.format.python().mp_node.precision))
            else:
                print("**"*(level+1)+" "+str(p.struct.type)+"= 0 and "+str(p.struct.format.python().mp_node.precision))
            if p.struct.type == "multiplication" or p.struct.type == "addition":
                self.dump(p.struct.q, level+1)
                self.dump(p.struct.r, level+1)
    def generate_scheme(self):
        # ...
        suffix = ["h", "m", "l"]
        if self.lutetia_mode and (isinstance(self.input_format[0], ML_FP_MultiElementFormat) and self.input_format[0].limb_num != 1):
            input_x = []
            for i in range(self.input_format[0].limb_num):
                input_x.append(
                    self.implementation.add_input_variable(
                        "x{}".format(suffix[i]),
                        self.input_format[0].get_limb_precision(i),
                        rel_error=0))
            vx = BuildFromComponent(*tuple(input_x), precision=self.get_input_precision(0))
        else:
            vx = self.implementation.add_input_variable("x", self.get_input_precision(0), rel_error=0)

        # extracting main arguments
        p = self.polynomial
        epsTarget = self.epsTarget
        I = self.interval

        pre_mll_context, mll_context = build_mll_context(self.get_input_precision(0), I)

        R, PR, powers = self.compute_polytree(p, I, True, epsTarget, mll_context, pre_mll_context, start_scheme=self.start_scheme)
        print("powers = {}".format(powers))

        self.dump(R.struct.poly)

        if PR != None and PR.convert2SollyaObject()["okay"] == True:
            Gappa = generateGappa("poly_", R.struct.poly, R.struct.powerings, mll_context)
            self.metalibm_p = R.struct.poly.struct.p
            print("R.struct.powerings = {}".format(R.struct.powerings))

            # list of pairs (k, eps target) used in R.struct.poly
            eps_power_list = self.get_power_eps(R.struct.poly)

            # map of k -> list of (epsValue, ml_format for power output)
            # power map has also been simplified: for each k and each format only keep the most
            # accurate node
            epsTargets = self.get_power_eps_from_powerings(R.struct.powerings)
            print("eps from powerings = {}".format(epsTargets))


            # map of (power index k) -> list of (pair, poly struct)
            Gappa_X_k = []
            Gappa_X_k_okay = True
            for k in epsTargets:
                # looking for a poly struct to implement x^k for p
                for idx, power_precision in enumerate(epsTargets[k]):
                    e = epsTargets[k][power_precision].epsilon
                    label = "pow_k{}_{}".format(k, idx)
                    # possible supernumeary computation of X^k to extract its poly object
                    # could not it be extract from main computeErrorBounds ?
                    power_poly = powers[k].convert2SollyaObject()["poly"]
                    power_R = computeErrorBounds(sollya.SollyaObject(power_poly), I, e, sollya.relative, mll_context)
                    print("processing {}, e={}, epsTarget={}, result precision is {}".format(label, e, epsTargets[k][power_precision].out_format.eps_target, power_R.struct.poly.struct.format.python()))

                    if str(power_R.struct.okay) == "false":
                        print("FAILED to compute error bound for power graph k={}, idx={}, epsilon={}, power_precision={}, I={}".format(k, idx, e, power_precision, I))
                        print(str(power_R.struct.reason))
                        raise Exception()
                    power_Gappa = generateGappa(label, power_R.struct.poly, [], mll_context)
                    Gappa_X_k.append((label, power_Gappa))
                    # accumulating okay value
                    Gappa_X_k_okay = (Gappa_X_k_okay and power_Gappa.struct.okay)

            if Gappa.struct.okay and Gappa_X_k_okay:
                # print(Gappa.struct.gappa)
                with open(self.proof, "w") as gappa_proof:
                    gappa_proof.write(str(Gappa.struct.gappa))
                for g_label, g_poly in Gappa_X_k:
                    gappa_out_name = "{}_{}.gappa".format(self.proof, g_label)
                    with open(gappa_out_name, "w") as gappa_proof:
                        gappa_proof.write(str(g_poly.struct.gappa))
            else:
                Log.report(Log.Error, "failed to generate gappa proof")

            # listing expression for X^k
            Gappa_X_k_poly = [g_R.struct.poly for label, g_R in Gappa_X_k]
            parser = ml_wrapper.SollyaConverter(vx, lutetia_tag_mode=not(self.debug))
            pre_poly_scheme = parser.build(Gappa.struct.poly, Gappa_X_k_poly)
            #print(pre_poly_scheme.get_str(depth=None, display_precision=True))

            # expanding meta-blocks
            poly_scheme = pre_poly_scheme

            # ...
            if not self.lutetia_mode:
                if len(poly_scheme) == 1:
                    poly_scheme_precision = poly_scheme[0].precision # ML_Binary64
                    poly_scheme = poly_scheme[0]
                else:
                    if len(poly_scheme) == 2 and poly_scheme[0].precision == ML_Binary64:
                        poly_scheme_precision = ML_DoubleDouble
                    elif len(poly_scheme) == 3 and poly_scheme[0].precision == ML_Binary64:
                        poly_scheme_precision = ML_TripleDouble
                    else:
                        raise NotImplementedError
                    poly_scheme = BuildFromComponent(*poly_scheme, precision=poly_scheme_precision)

            # ...
            if self.lutetia_mode:
                self.implementation.set_output_format(ML_Void)
                # ...
                r = [self.implementation.add_input_variable(self.function_name+"_res"+suffix[0], ML_Pointer_Format(poly_scheme[0].precision))]
                for i in range(1, len(poly_scheme)):
                     r += [self.implementation.add_input_variable(self.function_name+"_res"+suffix[i], ML_Pointer_Format(poly_scheme[i].precision))]
                # ...
                res = [ReferenceAssign(Dereference(r[0], precision = poly_scheme[0].precision), poly_scheme[0])]
                if len(r) == 2:
                    res += [ReferenceAssign(Dereference(r[1], precision = poly_scheme[1].precision), poly_scheme[1])]
                elif len(r) == 3:
                    res += [ReferenceAssign(Dereference(r[1], precision = poly_scheme[1].precision), poly_scheme[1])]
                    res += [ReferenceAssign(Dereference(r[2], precision = poly_scheme[2].precision), poly_scheme[2])]
                # ... reorganize inputs
                f_args = self.implementation.get_arg_list()
                lin = f_args[:len(f_args)-len(r)]
                lout = f_args[len(f_args)-len(r):]
                for i in range(len(lout)):
                    f_args[i] = lout[i]
                for i in range(len(lin)):
                    f_args[i+len(lout)] = lin[i]
                # ...
                scheme = Statement(*tuple(res))
            else:
                if poly_scheme_precision is self.precision:
                    scheme = Return(poly_scheme)
                else:
                    # inserting extra conversion if poly's result is not
                    # in the expected format
                    scheme = Return(
                        Conversion(poly_scheme, precision=self.precision)
                    )
            return scheme
        else:
            Log.report(Log.Error, "PR return okay=False")

    def numeric_emulate(self, vx):
        """ Numerical emulation using self.function """
        return self.function(vx)

if __name__ == "__main__":
    arg_template = ML_NewArgTemplate(
        default_arg=ImplementPolyMetaFunction.get_default_args()
    )
    arg_template.get_parser().add_argument(
         "--eps-target", dest="eps_target", default=S2**-60, type=sollya.parse,
        action="store", help="define error (epsilon) target")
    arg_template.get_parser().add_argument(
         "--function", default=sollya.sin(sollya.x), type=sollya.parse,
        action="store", help="define function to approximate")
    arg_template.get_parser().add_argument(
         "--range", default=Interval(-2**-4,2**-4), type=sollya.parse,
        action="store", help="define approximation interval")
    arg_template.get_parser().add_argument(
         "--degree", default=11, type=int,
         help="polynomial degree")
    arg_template.get_parser().add_argument(
         "--coeff-index", default=None, type=(lambda s: [int(v) for v in s.split(',')]),
        action="store", help="define list of polynomial monomials")
    arg_template.get_parser().add_argument(
         "--coeff-format", default=sollya.doubledouble, type=sollya.parse,
        action="store", help="define list of polynomial monomial formats")
    arg_template.get_parser().add_argument(
         "--proof", default="poly.gappa", type=str,
        action="store", help="define output Gappa proof file")
    arg_template.get_parser().add_argument(
         "--trace", default=False, const=True,
        action="store_const", help="enable verbose trace")
    arg_template.get_parser().add_argument(
         "--lutetia-mode", default=False, const=True,
        action="store_const", help="select lutetia API for output function")
    arg_template.get_parser().add_argument(
         "--start-scheme", default=None,
        action="store", help="force compute_polytree to start with a specific scheme")

    args = arg_template.arg_extraction()

    ft_module.enable_trace = args.trace

    f = args.function
    I = args.range
    Itmp = Interval(2**-100, sup(I))
    if not args.coeff_index is None:
        coeff_index = args.coeff_index
        # arg.coeffs_formats must be a list of format
        coeff_format = args.coeff_format
    else:
        coeff_index = args.degree
        # arg.coeffs_formats must be a single format
        coeff_format = [args.coeff_format] * (args.degree + 1)
    print("main function is {}, interval={}, coeffs={}, monomial_formats={}".format(f, Itmp, coeff_index, coeff_format))

    sollya.settings.prec = 3000;
    #F = [1, sollya.doubledouble, sollya.doubledouble] + [sollya.binary64]*3;

    p = sollya.fpminimax(f, coeff_index, coeff_format, Itmp);
    print("main p is {}".format(p))
    sollya.settings.prec = 300;

    epsTarget = args.eps_target#sollya.dirtyinfnorm(p/f-1, I)

    print("I=%s" % I)
    print("epsTarget=%s" % epsTarget)

    implement_poly = ImplementPolyMetaFunction(p, epsTarget, I, f, args)
    implement_poly.gen_implementation()

