# -*- coding: utf-8 -*-
import sys

import sollya, polynomial2tree
from sollya import Interval, sup

# local module to trace function easily
from function_trace import function_trace

from metalibm_core.core.ml_formats import (
    ML_DoubleDouble, ML_TripleDouble, ML_Binary32, ML_Binary64
)
from metalibm_core.core.ml_function import ML_FunctionBasis
from metalibm_core.core.precisions import ML_CorrectlyRounded
from metalibm_core.code_generation.code_constant import C_Code
from metalibm_core.core.ml_operations import (
    Return, Conversion, BuildFromComponent
)
from metalibm_core.utility.log_report import Log

from metalibm_core.utility.ml_template import (
    DefaultArgTemplate, ML_NewArgTemplate
)

from metalibm_core.code_generation.generic_processor import GenericProcessor

import ml_wrapper

sollya.settings.verbosity = 2

sollya.execute("/Users/grevy/LIRMM/genericimplementpoly.git/polynomials.sol")

# importing into python, the function generateGappa
# loaded from polynomials.sol
generateGappa = sollya.parse("generateGappa")

# importing into python, the function computeErrorBounds
# loaded from polynomials.sol
computeErrorBounds = sollya.parse("computeErrorBounds")


class MLL_Format:
    def __init__(self, ml_precision, accuracy, overlaps):
        # support format
        self.ml_precision = ml_precision
        # upper bound on the relative error, eps <= 2^-self.accuracy
        self.accuracy = accuracy
        # list of successive upper bounds on overlap (from MSB to LSB)
        self.overlaps = overlaps

    def _sollya_(self):
        return sollya.externaldata(self)

    @property
    def struct(self):
        """ Handler for compatibility with sollya's record """
        return self


class MLL_Context:
    def __init__(self, var_format):
        MLL_FORMAT_MAP = {
            ML_DoubleDouble: {"ml_precision": "dd", "accuracy": 102},
        }
        # dummy 1000-bit accuracy
        self.variableFormat = MLL_Format(var_format, 1000, [])

    def get_format_from_accuracy(self, accuracy):
        if accuracy <= 24:
            return MLL_Format(ML_Binary32, accuracy, [])
        elif accuracy <= 53:
            return MLL_Format(ML_Binary64, accuracy, [])
        elif accuracy <= 105:
            return MLL_Format(ML_DoubleDouble, accuracy, [])
        elif accuracy <= 150:
            return MLL_Format(ML_TripleDouble, accuracy, [])
        else:
            return sollya.error

    @function_trace(4)
    def computeBoundAddition(self, out_format, input_format_lhs, input_format_rhs):
        accuracy = min(
            out_format.struct.accuracy,
            input_format_lhs.struct.accuracy,
            input_format_rhs.struct.accuracy)
        accuracy = out_format.struct.accuracy
        return 2**(-accuracy)

    @function_trace(4)
    def computeBoundMultiplication(self, out_format, input_format_lhs, input_format_rhs):
        accuracy = min(
            out_format.struct.accuracy,
            input_format_lhs.struct.accuracy,
            input_format_rhs.struct.accuracy)
        accuracy = out_format.struct.accuracy
        return 2**(-accuracy)

    @function_trace(4)
    def computeBoundPower(self, k, out_format, var_format):
        accuracy = out_format.struct.accuracy
        return 2**-accuracy

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
           which is naturally stored as an af-format variable,
           to a bf-format variable.

           If bf is "larger" (i.e. more precise) than af,
           the error is most probably zero as this just means
           "upconverting" a variable.

           The inputs af and bf are of the "opaque format type".

           The result is a positive or zero real number.
           """
        if format_a.struct.accuracy <= format_b.struct.accuracy:
            return 0
        else:
            return 2**(-format_b.struct.accuracy + 1)

    @function_trace(4)
    def computeOutputFormatAddition(self, epsTarget, inputFormatA, inputFormatB): 
        """
           Returns the output format of an addition  that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        if epsTarget > 0:
            if epsTarget > 2**-53 and inputFormatA.struct.accuracy <= 53 and inputFormatB.struct.accuracy <= 53:
                return self.get_format_from_accuracy(53)
            else:
                accuracy = max(
                    inputFormatA.struct.accuracy,
                    inputFormatB.struct.accuracy,
                    sollya.ceil(max(2, sollya.ceil(-sollya.log2(epsTarget) / 51) * 51))
                )
                return self.get_format_from_accuracy(accuracy)
        else:
            accuracy = max(inputFormatA.struct.accuracy, inputFormatB.struct.accuracy)
            return self.get_format_from_accuracy(accuracy)

    @function_trace(4)
    def computeOutputFormatMultiplication(self, epsTarget, inputFormatA, inputFormatB): 
        """ Returns the output format of a multiplication that will be fed two operands
           of formats inputFormatA and inputFormatB and that is supposed to have a
           relative error less than or equal to epsTarget.

           The input and output formats are of type "opaque format type".
           The epsTarget input is a positive or zero real number.
        """
        if epsTarget > 0:
            if epsTarget > 2**-53 and inputFormatA.struct.accuracy <= 53 and inputFormatB.struct.accuracy <= 53:
                return self.get_format_from_accuracy(53)
            else:
                accuracy = max(
                    inputFormatA.struct.accuracy,
                    inputFormatB.struct.accuracy,
                    sollya.ceil(max(2, sollya.ceil(-sollya.log2(epsTarget)) / 51) * 51)
                )
                return self.get_format_from_accuracy(accuracy)
        else:
            accuracy = max(inputFormatA.struct.accuracy, inputFormatB.struct.accuracy)
            return self.get_format_from_accuracy(accuracy)

    @function_trace(2)
    def computeConstantFormat(self, c):
        if c == 0:
            return self.get_format_from_accuracy(53)
        else:
            accuracy = 0
            cN = c
            while cN != 0:
                cR = sollya.round(cN, sollya.binary64, sollya.RN)
                cN = cN - cR
                accuracy += 53
            return self.get_format_from_accuracy(53)


    @function_trace(4)
    def computeNeededVariableFormat(self, I, epsTarget, variableFormat):
        if epsTarget > 0:
            if epsTarget >= 2**-53:
                return self.get_format_from_accuracy(53)
            else:
                return self.get_format_from_accuracy(
                    sollya.ceil(max(2, sollya.ceil(-sollya.log2(epsTarget)) / 52) * 52)
                )
        else:
            return variableFormat

    @function_trace(4)
    def computeOutputFormatPower(self, k, epsTarget, variableFormat):
        if epsTarget > 0:
            if epsTarget >= 2**-53:
                return self.get_format_from_accuracy(53)
            else:
                accuracy = sollya.ceil(max(2, sollya.ceil(-sollya.log2(epsTarget)) / 51) * 51)
                return self.get_format_from_accuracy(accuracy)
        else:
            return error

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
            if epsTarget == 0:
                if c == 0:
                    return {"okay": True, "c": c}
                else:
                    # saving sollya precision
                    oldPrec = sollya.settings.prec
                    # setting prec to very high value
                    sollya.settings.prec = 10000
                    cT = Interval(c)
                    sollya.settings.prec = oldPrec
                    cR = sollya.mid(cT)
                    if cR == c:
                        return {"okay": True, "c": c}
                    else:
                        return {"okay": False}
            else:
                if c == 0:
                    return {"okay": True, "c": c}
                else:
                    cR = sollya.round(c, sollya.binary64, sollya.RN)
                    while abs(cR / c - 1) > epsTarget:
                        cN = round(c - Cr, sollya.binary64, sollya.RN)
                        cR += cN
                    return {"okay": True, "c": cR}


pre_mll_context = MLL_Context(ML_DoubleDouble)

# building sollya's structure to be used as opaque type for
# computeErrorBounds and generateGappa
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


class ImplementPolyMetaFunction(ML_FunctionBasis):
    """ Metalibm lugdunum meta-function to wrap implementpoly """
    function_name = "implementpoly"

    def __init__(self, polynomial, epsTarget, interval, function = None, proof = "proof.gappa", arg_template = DefaultArgTemplate, metalibm_lutetia_prototype = False):
        ML_FunctionBasis.__init__(self, args=arg_template)

        self.polynomial = polynomial
        self.epsTarget = epsTarget
        self.interval = interval
        self.function = function
        self.proof = proof
        self.input_format = args.input_precisions
        self.lutetia = metalibm_lutetia_prototype
        
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
            "passes": ["typing:basic_legalization", "beforecodegen:expand_multi_precision"],
            "vector_size": 1,
            "arity": 1,
        }
        default_args.update(args)
        return DefaultArgTemplate(**default_args)

    def get_power_eps(self, tree, l = []):
        if tree.struct.type == "power":
            l.append((tree.struct.k, tree.struct.epsTarget))
        elif tree.struct.type not in ["variable", "constant"]:
            l = self.get_power_eps(tree.struct.q, l)
            l = self.get_power_eps(tree.struct.r, l)
        return l

    def generate_scheme(self):
        # ...
        if self.lutetia == True or self.input_format[0].limb_num == 1:
            suffix = ["h", "m", "l"]
            input_x = []
            for i in range(self.input_format[0].limb_num):
                input_x.append(self.implementation.add_input_variable("x"+str(suffix[i]), self.input_format[0].get_limb_precision(i), rel_error=0))
            vx = BuildFromComponent(*tuple(input_x), precision=self.get_input_precision(0))
        else:
            vx = self.implementation.add_input_variable("x", self.get_input_precision(0), rel_error=0)
        print(vx, vx.precision)
        # ...
        p = self.polynomial
        epsTarget = self.epsTarget
        I = self.interval

        PR, powers = polynomial2tree.buildTree(p, classic='estrin', power=True)
        PR = PR.convert2SollyaObject()

        if PR["okay"] == True:
            P = PR["poly"]
            # print(P, I, epsTarget, mll_context)
            R = computeErrorBounds(P, I, epsTarget, mll_context)
            Gappa = generateGappa("poly_", R.struct.poly, R.struct.powerings, mll_context)
            
            # ...
            eps_power_list = self.get_power_eps(R.struct.poly)
            epsTargets = {}
            for k in R.struct.powerings:
                epsTargets[int(k.struct.k)] = []
                #for f in k.struct.variants:
                #    epsTargets[int(k.struct.k)].append(f.struct.format.python().accuracy)
                for kvalue, epsvalue in eps_power_list:
                    if kvalue == int(k.struct.k):
                        epsTargets[int(k.struct.k)].append((epsvalue,pre_mll_context.computeOutputFormatPower(None,epsvalue,None)))
            tmp = {}
            for k in epsTargets:
                for eS, fS in epsTargets[k]:
                    if k not in tmp.keys():
                        tmp[k] = [(eS, fS)]
                    else:
                        insert = False
                        for idx in range(0, len(tmp[k])):
                            eD, fD = tmp[k][idx]
                            if pre_mll_context.compareFormats(fS, fD) == 0 and eS > eD:
                                tmp[k][idx] = (eD, fD)
                                insert = True
                        if not insert and pre_mll_context.compareFormats(fS, fD) != 0:
                            tmp[k].append((eS, fS))
            epsTargets = {}
            for k in tmp:
                epsTargets[k] = []
                for e in tmp[k]:
                    epsTargets[k].append(e[0])
            #
            Gappa_X_k = []
            Gappa_X_k_okay = True
            for p in epsTargets:
                idx = 0
                for e in epsTargets[p]:
                    tmp = computeErrorBounds(powers[p].convert2SollyaObject()["poly"], I, e, mll_context)
                    tmp = generateGappa("pow_k"+str(p)+"_"+str(idx), tmp.struct.poly, [], mll_context)
                    Gappa_X_k.append(("pow_k"+str(p)+"_"+str(idx), tmp))
                    Gappa_X_k_okay = (Gappa_X_k_okay and tmp.struct.okay)
                    idx = idx + 1

            if Gappa.struct.okay and Gappa_X_k_okay:
                # print(Gappa.struct.gappa)
                with open(self.proof, "w") as gappa_proof:
                    gappa_proof.write(str(Gappa.struct.gappa))
                for g in Gappa_X_k:
                    with open(self.proof+"_"+str(g[0])+".gappa", "w") as gappa_proof:
                        gappa_proof.write(str(g[1].struct.gappa))
            else:
                Log.report(Log.Error, "failed to generate gappa proof")

            Gappa_X_k_poly = [tmp[1].struct.poly for tmp in Gappa_X_k]
            parser = ml_wrapper.SollyaConverter(vx)
            poly_scheme = parser.build(Gappa.struct.poly, Gappa_X_k_poly)
            if poly_scheme.precision is self.precision:
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
         "--eps-target", dest="eps_target", default=2**-60, type=sollya.parse,
        action="store", help="define error (epsilon) target")
    arg_template.get_parser().add_argument(
         "--function", default=sollya.sin(sollya.x), type=sollya.parse,
        action="store", help="define function to approximate")
    arg_template.get_parser().add_argument(
         "--range", default=Interval(-2**-4,2**-4), type=sollya.parse,
        action="store", help="define approximation interval")
    arg_template.get_parser().add_argument(
         "--monomials", default=[1, 3, 5, 7, 9, 11], type=sollya.parse,
        action="store", help="define list of polynomial monomials")
    arg_template.get_parser().add_argument(
         "--proof", default="poly.gappa", type=str,
        action="store", help="define output Gappa proof file")
    
    args = arg_template.arg_extraction()
    
    f = args.function
    I = args.range
    Itmp = Interval(2**-100, sup(I))
    monomials = args.monomials

    sollya.settings.prec = 3000;
    F = [sollya.doubledouble, sollya.doubledouble]+[sollya.binary64]*4;

    p = sollya.fpminimax(f, monomials, F, Itmp);
    sollya.settings.prec = 165;

    epsTarget = sollya.dirtyinfnorm(p/f-1, I)

    print("I=%s" % I)
    print("epsTarget=%s" % epsTarget)

    args.input_precisions = [ML_DoubleDouble]
    
    implement_poly = ImplementPolyMetaFunction(p, epsTarget, I, None, str(args.proof), args, True)
    implement_poly.gen_implementation()
