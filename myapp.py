import sys
import traceback
import argparse
import os
import collections


from wsgiref.simple_server import make_server
from tg import MinimalApplicationConfigurator
from tg.configurator.components.statics import StaticsConfigurationComponent
from tg import expose, TGController

from metalibm_core.utility.ml_template import (
    precision_parser, target_parser, target_map
)
from metalibm_core.core.passes import Pass
from metalibm_core.code_generation.code_constant import (
    C_Code, LLVM_IR_Code, OpenCL_Code
)


import metalibm_core.utility.log_report as ml_log_report
import metalibm_core.code_generation.code_configuration as ml_code_configuration
from metalibm_core.code_generation.code_utility import insert_line_break
import metalibm_core.utility.version_info as ml_version_info

import metalibm_core.utility.log_report as ml_log_report

import metalibm_functions.ml_exp
import metalibm_functions.ml_expm1
import metalibm_functions.ml_exp2
import metalibm_functions.ml_cbrt
import metalibm_functions.ml_sqrt
import metalibm_functions.ml_isqrt
import metalibm_functions.ml_vectorizable_log
import metalibm_functions.ml_cosh
import metalibm_functions.ml_sinh
import metalibm_functions.ml_sincos
import metalibm_functions.ml_atan
import metalibm_functions.external_bench
import metalibm_functions.ml_tanh
import metalibm_functions.ml_div
import metalibm_functions.generic_log
import metalibm_functions.erf
import metalibm_functions.ml_tanh
import metalibm_functions.ml_cosh
import metalibm_functions.ml_sinh
import metalibm_functions.ml_div


GIT_COMMENT_TEMPLATE = """\
generated using Metalibm Web App ({localhost})
based on metalibm {version_num}
sha1 git: {sha1}({sha1_status})
"""




def custom_get_common_git_comment(localhost, url_getter):
    def func():
        git_comment = GIT_COMMENT_TEMPLATE.format(
            localhost=localhost, version_num=ml_version_info.VERSION_NUM,
            sha1=str(ml_version_info.GIT_SHA),
            sha1_status="clean" if ml_version_info.GIT_STATUS else "dirty")
        if not ml_version_info.GIT_STATUS:
            git_comment += "\nWARNING: git status was not clean when file was generated !\n\n"
            ml_log_report.Log.report(ml_log_report.Log.Warning, "git status was not clean when file was generated !")
        else:
            git_comment += "\nINFO: git status was clean when file was generated.\n\n"
        git_comment += "command used for generation:\n  %s\n" % insert_line_break(url_getter(), break_len=70, sep="&", allow_init_break=False, break_char="&\\\n    ")

        return git_comment
    return func



class MetalibmWebApp:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    TEMPLATE = os.path.join(SCRIPT_DIR, "main.xhtml")

    FUNCTION_MAP = {
        "exp": (metalibm_functions.ml_exp.ML_Exponential, {}),
        "exp2": (metalibm_functions.ml_exp2.ML_Exp2, {}),
        "cbrt": (metalibm_functions.ml_cbrt.ML_Cbrt, {}),
        "div":  (metalibm_functions.ml_div.ML_Division, {}),
        "sqrt": (metalibm_functions.ml_sqrt.MetalibmSqrt, {}),
        "log": (metalibm_functions.generic_log.ML_GenericLog, {}),
        "log2": (metalibm_functions.generic_log.ML_GenericLog, {"basis": 2}),
        "log10": (metalibm_functions.generic_log.ML_GenericLog, {"basis": 10}),
        "cos": (metalibm_functions.ml_sincos.ML_SinCos, {"sin_output": False}),
        "sin": (metalibm_functions.ml_sincos.ML_SinCos, {"sin_output": True}),
        "erf": (metalibm_functions.erf.ML_Erf, {}),
        "tanh": (metalibm_functions.ml_tanh.ML_HyperbolicTangent, {}),
        "cosh": (metalibm_functions.ml_cosh.ML_HyperbolicCosine, {}),
        "sinh": (metalibm_functions.ml_sinh.ML_HyperbolicSine, {}),
    }

    LANGUAGE_MAP = {
        "c": C_Code,
        "ll-ir": LLVM_IR_Code,
        "opencl-c": OpenCL_Code
    }

    format_list = ["binary32", "binary64"]
    vector_size_list = [1, 2, 4, 8]
    sub_vector_size_list = [1, 2, 4, 8]


    # dictionnary tag -> url of application examples
    EXAMPLE_MAP = collections.OrderedDict([
        ("4-way single precision exponential on generic vector target in C", "{localhost}/function?name=exp&io_format=binary32&vector_size=4&sub_vector_size=4&target=vector&language=c&new_pass=basic_block_simplification&registered_pass_list=vector_mask_test_legalization%2Cvirtual_vector_bool_legalization"),
        ("4-way single precision exponential on x86 AVX2 in C", "{localhost}/function?name=exp&io_format=binary64&vector_size=4&sub_vector_size=4&target=x86_avx2&language=c&new_pass=silence_fp_ops&registered_pass_list=virtual_vector_bool_legalization%2Cvector_mask_test_legalization%2Cm128_promotion%2Cm256_promotion"),
        ("single precision division in C", "{localhost}/function?name=div&io_format=binary32&vector_size=1&sub_vector_size=1&target=generic&language=c&new_pass=expand_multi_precision&registered_pass_list=basic_legalization%2Cexpand_multi_precision%2Ccheck_processor_support"),
        ("single precision exponential in LLVM-IR", "{localhost}/function?name=exp&io_format=binary32&vector_size=1&sub_vector_size=1&target=llvm&language=ll-ir&new_pass=rtl_legalize&registered_pass_list=gen_basic_block%2Cbasic_block_simplification%2Cssa_translation"),
    ])

    ALLOWED_PASS_LIST = [
        "m128_promotion",
        "m256_promotion",
        "basic_block_simplification",
        "basic_legalization",
        "check_precision",
        "check_processor_support",
        "check_target_support",
        "dump",
        "expand_multi_precision",
        "fuse_fma",
        "gen_basic_block",
        "instantiate_abstract_prec",
        "instantiate_prec",
        "m128_promotion",
        "m256_promotion",
        "silence_fp_ops",
        "ssa_translation",
        "sub_expr_sharing",
        "tag_node",
        "vector_mask_test_legalization",
        "vector_promotion",
        "virtual_vector_bool_legalization",
        # "check_generic",
        #"critical_path_eval",
        #"dump_with_stages",
        #"simplify_rtl",
        #"size_datapath",
        #"unify_pipeline_stages",
        #"quit",
        #"rtl_legalize",
    ]

    def __init__(self, localhost):
        self.LOCALHOST = localhost
        available_pass_list = [tag for tag in Pass.get_pass_tag_list() if tag in self.ALLOWED_PASS_LIST]

        self.option_dict = {
            "format_list": self.format_list,
            "vector_size_list": self.vector_size_list,
            "sub_vector_size_list": self.sub_vector_size_list,
            "function_name_list": sorted(list(self.FUNCTION_MAP.keys())),
            "target_list": sorted(list(target_map.keys())),
            "available_pass_list": sorted(available_pass_list),
            "language_list": list(self.LANGUAGE_MAP.keys()),
            "example_map": {k: self.encode_url(v) for k, v in self.EXAMPLE_MAP.items()},
            "localhost": self.LOCALHOST,
        }


    def encode_url(self, url):
        encoded_url = url.format(localhost=self.LOCALHOST)
        return encoded_url


class MyLogHandler:
    def __init__(self):
        self.log_output = ""
    def write(self, msg):
        self.log_output += msg

def gen_report_issue_url(url="https://github.com/kalray/metalibm/issues/new", **kw):
    """ Generated an url to automatically report an error in metalibm
        encountered from the metalibm web app """
    title="issue with {} reported from MWA".format(kw["name"])
    return "{}?title={}&body={}".format(url, title, ", ".join("{}={}".format(k, v) for k, v in kw.items()))

# installing custom log handler for metalibm
ml_log_report.Log.exit_on_error = False
ml_log_report.Log.log_stream = MyLogHandler()

# RootController of our web app, in charge of serving content for /
class RootController(TGController):
    def __init__(self, localhost):
        super().__init__()
        self.mwa = MetalibmWebApp(localhost)

    @expose(MetalibmWebApp.TEMPLATE)
    def index(self):
        return dict(
            code="no code generated",
            precision=self.mwa.format_list[0],
            registered_pass_list=["check_processor_support"],
            vector_size=1,
            sub_vector_size=1,
            debug=False,
            target="generic",
            language="c",
            name=self.mwa.option_dict["function_name_list"][0],
            error=None,
            **self.mwa.option_dict)


    @expose(MetalibmWebApp.TEMPLATE)
    def function(self, name, io_format="binary32", vector_size=1, target="generic", registered_pass_list="", sub_vector_size=1, debug=False, language="c"):

        input_url = "{localhost}/function?name={name}&io_format={io_format}&vector_size={vector_size}&target={target}&registered_pass_list={registered_pass_list}&debug={debug}&language={language}".format(
            localhost=self.mwa.LOCALHOST,
            name=name, io_format=io_format,
            vector_size=vector_size, target=target,
            registered_pass_list=registered_pass_list,
            sub_vector_size=sub_vector_size, debug=debug,
            language=language)
        ml_code_configuration.GLOBAL_GET_GIT_COMMENT = custom_get_common_git_comment(self.mwa.LOCALHOST, lambda : input_url)

        code = "generated {} for {} with vector_size={}".format(name, io_format, vector_size)
        registered_pass_list = registered_pass_list.split(",")
        print("registered_pass_list={}".format(registered_pass_list))
        report_issue_url = "https://github.com/metalibm/MetalibmWebApp/issues/new"
        error = None
        if not name in self.mwa.FUNCTION_MAP:
            source_code = "unknown function {}".format(name)
        else:
            # clearing logs
            ml_log_report.Log.log_stream.log_output = ""
            try:
                fct_ctor, fct_extra_args = self.mwa.FUNCTION_MAP[name]
                language_object = self.mwa.LANGUAGE_MAP[language]
                precision = precision_parser(io_format)
                vector_size = int(vector_size)
                sub_vector_size = int(sub_vector_size)
                debug = bool(debug)
                target_class = target_parser(target)
                target_inst = target_class()
                passes = ["beforecodegen:{}".format(pass_tag) for pass_tag in registered_pass_list]
                args = fct_ctor.get_default_args(
                    precision=precision,
                    vector_size=vector_size,
                    sub_vector_size=sub_vector_size,
                    passes=passes,
                    language=language_object,
                    debug=debug,
                    target=target_inst,
                    **fct_extra_args)
                fct_instance = fct_ctor(args=args)
                source_code = fct_instance.generate_full_source_code()
            except:
                e = sys.exc_info()
                error = "Output: \n{}\nException:\n {}".format(ml_log_report.Log.log_stream.log_output, "".join(traceback.format_exception(*e)))
                source_code = ""
                report_issue_url = gen_report_issue_url("https://github.com/kalray/metalibm/issues/new",
                    precision=io_format,
                    name=name,
                    target=target,
                    vector_size=vector_size,
                    debug=debug,
                    language=language,
                    sub_vector_size=sub_vector_size,
                    registered_pass_list=registered_pass_list,
                )
        return dict(
            code=source_code,
            precision=io_format,
            name=name,
            target=target,
            vector_size=vector_size,
            debug=debug,
            language=language,
            sub_vector_size=sub_vector_size,
            registered_pass_list=registered_pass_list,
            report_issue_url=report_issue_url,
            error=error,
            **self.mwa.option_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--localhost", type=str, default="localhost:8080",
                        help="website base url")
    parser.add_argument("--port", type=int, default=8080, help="server connection port")
    args = parser.parse_args()

    # Serve the newly configured web application.
    PORT = args.port
    LOCALHOST = args.localhost
    # Configure a new minimal application with our root controller.
    config = MinimalApplicationConfigurator()
    config.register(StaticsConfigurationComponent)
    config.update_blueprint({
        'root_controller': RootController(LOCALHOST),
        'renderers': ['kajiki']
    })
    config.update_blueprint({
        "serve_static": True,
        "paths": {
            "static_files": "public"
        }
    })

    print("Serving on port {}...".format(PORT))
    httpd = make_server('', PORT, config.make_wsgi_app())
    httpd.serve_forever()
