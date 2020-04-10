# -*- coding: utf-8 -*-
import sollya
from sollya import Interval, sup
S2 = sollya.SollyaObject(2)

from implementpoly_metafunction import ImplementPolyMetaFunction, build_mll_context

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.core.ml_formats import ML_DoubleDouble


def unit_test_0():
    cst = sollya.parse(" 4.11413408561438400844066014298373608344977785003823426773348852708088123596954730531888867e-6")
    ctx, _ = build_mll_context(ML_DoubleDouble, Interval(-S2**-3, S2**-3))
    new_cst = ctx.roundConstant(cst, S2**-150)
    print(ctx.computeConstantFormat(new_cst.struct.c))
    cst = sollya.parse("1.576569824486569195950668504183247047506519550056709129157535539778371949923618261163646428e-16")
    new_cst = ctx.roundConstant(cst, 0)
    print(ctx.computeConstantFormat(cst))

    cst = sollya.parse("1.5765698244865692191047500701353578585831716326812e-16")
    print(ctx.computeConstantFormat(cst))

def unit_test_1():
    arg_template = ImplementPolyMetaFunction.get_default_args()
    f = sollya.sin(sollya.x)
    I = Interval(-S2**-4, S2**-4)
    Itmp = Interval(2**-100, sup(I))
    monomials = [1, 3, 5, 7, 9, 11]

    sollya.settings.prec = 3000;
    F = [sollya.doubledouble, sollya.doubledouble]+[sollya.binary64]*4;

    p = sollya.fpminimax(f, monomials, F, Itmp);
    sollya.settings.prec = 300;

    epsTarget = sollya.dirtyinfnorm(p/f-1, I)

    print("I=%s" % I)
    print("epsTarget=%s" % epsTarget)

    implement_poly = ImplementPolyMetaFunction(p, epsTarget, I, f, arg_template)
    implement_poly.gen_implementation()

if __name__ == "__main__":
    TEST_LIST = {
        "unit_test_0": unit_test_0,
        "unit_test_1": unit_test_1,
    }
    for test_label in TEST_LIST:
        print("executing {}".format(test_label))
        TEST_LIST[test_label]()
