# -*- coding: utf-8 -*-
import sollya
from sollya import Interval, sup
S2 = sollya.SollyaObject(2)

from implementpoly_metafunction import ImplementPolyMetaFunction, build_mll_context
import function_trace as ft_module

from metalibm_core.utility.ml_template import ML_NewArgTemplate
from metalibm_core.core.ml_formats import ML_DoubleDouble



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
         "--monomials", default=[1, 3, 5, 7, 9, 11], type=sollya.parse,
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

    #f = sollya.parse(" 53849069281361992801008944059481b-100 + x * (988070124679083b-50 + x * (-8636562708039121b-55 + x * (-1317426832905441b-53 + x * (5757708472025691b-58 + x * (8431531730532333b-60 + x * (-6141555702248167b-63 + x * (-1606006008204893b-63 + x * (27417644945419b-61 + x * (5709946017123031b-71 + x * (-1246555663920131b-73 + x * (-6180684337531123b-78)))))))))))")
    # epsTarget = S2**-100
    # I = sollya.parse(" [-1501199875790193b-53;750599937895069b-52]")

    # f = sollya.parse("x * (9007199254739273b-53 + x^2 * (6004799503402639b-55 + x^2 * (4803839561542047b-59 + x^2 * (7320147289428393b-65 + x^2 * (3252742925169029b-70 + x * (1808160821314751b-96 + x * 7718416187061345b-78))))))")
    # p = f
    # epsTarget = S2**-41
    # I = Interval(-1, 1)

    # f= sollya.parse("1 + x^2 * (-1b-1 + x^2 * (3002399751579061b-56 + x^2 * (-3202559709017065b-61 + x * (1315715817525467b-89 + x * 7319261925338859b-68))))")
    # p = f
    # I = sollya.parse("[-6.25e-2;6.25e-2]")
    # epsTarget = sollya.parse("1.08420217248550443400745280086994171142578125e-19")

    f = sollya.parse("10164112626621887990988689126069b-103 + x * (162625802025950207855819026017087b-107 + x * (162625802025950207855819043038579b-108 + x * (6938700886440542201848472342972911b-115 + x * (108417201350633468545477044046999b-111 + x * (86733761080506351304232338910637b-113 + x * (231290031234941542591955042265063b-117 + x * 1057326389439115401022917434699937b-122))))))")
    I = sollya.parse("[-260138115340281b-59;264955349881883643879011315797352727414831658140266535523326246369409536792312548640758723739142978884799b-358]")
    p = f
    epsTarget = sollya.parse("1b-107")


    print("I=%s" % I)
    print("epsTarget=%s" % epsTarget)

    implement_poly = ImplementPolyMetaFunction(p, epsTarget, I, f, args)
    implement_poly.gen_implementation()
