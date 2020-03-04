import sys
import traceback
import argparse
import os
import collections
import time

import sollya


from wsgiref.simple_server import make_server
from tg import MinimalApplicationConfigurator
from tg.configurator.components.statics import StaticsConfigurationComponent
from tg import expose, TGController
import kajiki

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

import metalibm_functions.function_expr as ml_function_expr

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


PRE_CONFIGURE_FLOWS = {
    #"llvm_flow": {
    #    "title": "LLVM required passes",
    #    "pass_list": ["gen_basic_block", "basic_block_simplification", "ssa_translation"],
    #},
    "vector_flow": {
        "title": "Vector recommended passes",
        "pass_list": ["vector_mask_test_legalization", "virtual_vector_bool_legalization"],
    },
    "x86_sse_flow": {
        "title": "x86-SSE recommended passes",
        "pass_list": ["m128_promotion"],
    },
    "x86_avx_flow": {
        "title": "x86-AVX recommended passes",
        "pass_list": ["m128_promotion", "m256_promotion"],
    },
}

preconf_flow_script= kajiki.TextTemplate("""\
var pre_conf_map = {
%for flow in preconf_flow
    ${'"{}": [{}],\\n\\t'.format(flow, ",".join('"%s"' % tag for tag in preconf_flow[flow]['pass_list']))}
%end
};
""")({"preconf_flow": PRE_CONFIGURE_FLOWS}).render()

class MetalibmWebApp:
    SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
    TEMPLATE = os.path.join(SCRIPT_DIR, "main.xhtml")


    LANGUAGE_MAP = {
        "c": C_Code,
        "ll-ir": LLVM_IR_Code,
        "opencl-c": OpenCL_Code
    }

    format_list = ["binary32", "binary64"]
    vector_size_list = [1, 2, 4, 8]
    sub_vector_size_list = ["default", 1, 2, 4, 8]


    # dictionnary tag -> url of application examples
    EXAMPLE_MAP = collections.OrderedDict([
        ("4-way single precision exponential on generic vector target in C", "{localhost}/function?fct_expr=exp(x)&io_format=binary32&vector_size=4&sub_vector_size=4&target=vector&language=c"),
        ("4-way single precision exponential on x86 AVX2 in C", "{localhost}/function?fct_expr=exp(x)&io_format=binary64&vector_size=4&sub_vector_size=4&target=x86_avx2&language=c&registered_pass_list=virtual_vector_bool_legalization%2Cvector_mask_test_legalization%2Cm128_promotion%2Cm256_promotion"),
        ("single precision division in C", "{localhost}/function?fct_expr=div(x,y)&io_format=binary32&vector_size=1&sub_vector_size=1&target=generic&language=c&registered_pass_list=basic_legalization%2Cexpand_multi_precision%2Ccheck_processor_support"),
        ("single precision exponential in LLVM-IR", "{localhost}/function?fct_expr=exp(x)&io_format=binary32&vector_size=1&sub_vector_size=1&target=llvm&language=ll-ir"),
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
        "evaluate_range",
        "fuse_fma",
        "gen_basic_block",
        "instantiate_abstract_prec",
        "instantiate_prec",
        "m128_promotion",
        "m256_promotion",
        "numerical_simplification",
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

    def __init__(self, localhost, version_info):
        self.LOCALHOST = localhost
        available_pass_list = [tag for tag in Pass.get_pass_tag_list() if tag in self.ALLOWED_PASS_LIST]

        self.option_dict = {
            "format_list": self.format_list,
            "vector_size_list": self.vector_size_list,
            "sub_vector_size_list": self.sub_vector_size_list,
            "target_list": sorted(list(target_map.keys())),
            "available_pass_list": sorted(available_pass_list),
            "language_list": list(self.LANGUAGE_MAP.keys()),
            "example_map": {k: self.encode_url(v) for k, v in self.EXAMPLE_MAP.items()},
            "localhost": self.LOCALHOST,
            "preconf_flow": PRE_CONFIGURE_FLOWS,
            "preconf_flow_script": preconf_flow_script,
            "version_info": version_info,
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
    title="issue with expression {} reported from MWA".format(kw["fct_expr"])
    return "{}?title={}&body={}".format(url, title, ", ".join("{}={}".format(k, v) for k, v in kw.items()))

# installing custom log handler for metalibm
ml_log_report.Log.exit_on_error = True
ml_log_report.Log.log_stream = MyLogHandler()

# RootController of our web app, in charge of serving content for /
class RootController(TGController):
    def __init__(self, localhost, version_info):
        super().__init__()
        self.mwa = MetalibmWebApp(localhost, version_info)

    @expose(MetalibmWebApp.TEMPLATE, content_type="text/html")
    def index(self):
        return dict(
            code="no code generated",
            precision=self.mwa.format_list[0],
            registered_pass_list=["check_processor_support"],
            vector_size=1,
            sub_vector_size="default",
            debug=False,
            target="generic",
            language="c",
            fct_expr="exp(x)",
            error=None,
            range_lo="-infty",
            range_hi="+infty",
            range_nan=True,
            total_time=None,
            **self.mwa.option_dict)


    @expose(MetalibmWebApp.TEMPLATE, content_type="text/html")
    def function(self, fct_expr="exp(x)", io_format="binary32", vector_size=1, target="generic", registered_pass_list="", sub_vector_size="default", debug=False, language="c", range_nan="false", range_lo="-infty", range_hi="+infty"):

        total_time = None
        input_url = "{localhost}/function?fct_expr={fct_expr}&io_format={io_format}&vector_size={vector_size}&target={target}&registered_pass_list={registered_pass_list}&debug={debug}&language={language}".format(
            localhost=self.mwa.LOCALHOST,
            fct_expr=fct_expr, io_format=io_format,
            vector_size=vector_size, target=target,
            registered_pass_list=registered_pass_list,
            sub_vector_size=sub_vector_size, debug=debug,
            language=language)
        ml_code_configuration.GLOBAL_GET_GIT_COMMENT = custom_get_common_git_comment(self.mwa.LOCALHOST, lambda : input_url)

        registered_pass_list = [tag for tag in registered_pass_list.split(",") if tag != ""]

        report_issue_url = "https://github.com/metalibm/MetalibmWebApp/issues/new"
        error = None
        # checking inputs
        try:
            if not ml_function_expr.check_fct_expr(fct_expr):
                source_code = "invalid function expression {}".format(fct_expr)
            elif not all((pass_tag in self.mwa.ALLOWED_PASS_LIST) for pass_tag in registered_pass_list): 
                source_code = "unknown pass in {}".format([pass_tag for pass_tag in registered_pass_list if not pass_tag in self.mwa.ALLOWED_PASS_LIST])
                print(source_code)
            # no allowed target list for now
            elif not io_format in self.mwa.format_list:
                source_code = ("forbidden format {}".format(io_format))
                print(source_code)
            elif not int(vector_size) in  self.mwa.vector_size_list:
                source_code = ("forbidden vector_size {}".format(vector_size))
                print(source_code)
            elif sub_vector_size != "default" and not int(sub_vector_size) in self.mwa.sub_vector_size_list:
                source_code = ("forbidden sub_vector_size {}".format(sub_vector_size))
                print(source_code)
            elif not language in self.mwa.LANGUAGE_MAP:
                source_code = ("forbidden language {}".format(language))
                print(source_code)
            elif not range_nan.lower() in ["true", "false"]:
                source_code = ("invalid range NaN  flag {}".format(range_nan))
                print(source_code)
        except:
            e = sys.exc_info()
            error = "Exception:\n {}".format("".join(traceback.format_exception(*e))).replace('\n', '<br/>')
            source_code = ""
        else:
            # clearing logs
            ml_log_report.Log.log_stream.log_output = ""
            try:
                start_time = time.perf_counter()
                fct_ctor = ml_function_expr.FunctionExpression
                arity=ml_function_expr.count_expr_arity(fct_expr)
                fct_extra_args = {}
                language_object = self.mwa.LANGUAGE_MAP[language]
                precision = precision_parser(io_format)
                vector_size = int(vector_size)
                sub_vector_size = None if sub_vector_size == "default" else int(sub_vector_size)
                #range_nan = bool(range_nan)
                range_nan = range_nan.lower() in ["true"]
                if range_nan:
                    input_interval = None
                else:
                    input_interval = sollya.Interval(sollya.parse(range_lo), sollya.parse(range_hi))
                debug = bool(debug)
                target_class = target_parser(target)
                target_inst = target_class()
                passes = ["beforecodegen:{}".format(pass_tag) for pass_tag in registered_pass_list if pass_tag in self.mwa.ALLOWED_PASS_LIST]
                args = fct_ctor.get_default_args(
                    function_expr_str=[fct_expr],
                    precision=precision,
                    input_precisions=(precision,)*arity,
                    input_intervals=(input_interval,)*arity,
                    vector_size=vector_size,
                    sub_vector_size=sub_vector_size,
                    passes=passes,
                    language=language_object,
                    debug=debug,
                    target=target_inst,
                    **fct_extra_args)
                fct_instance = fct_ctor(args=args)
                source_code = fct_instance.generate_full_source_code()
                total_time = time.perf_counter() - start_time
            except:
                e = sys.exc_info()
                error = "Output: \n{}\nException:\n {}".format(ml_log_report.Log.log_stream.log_output, "".join(traceback.format_exception(*e))).replace('\n', '<br/>')
                source_code = ""
                report_issue_url = gen_report_issue_url("https://github.com/kalray/metalibm/issues/new",
                    precision=io_format,
                    fct_expr=fct_expr,
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
            fct_expr=fct_expr,
            target=target,
            vector_size=vector_size,
            debug=debug,
            language=language,
            sub_vector_size=sub_vector_size,
            registered_pass_list=registered_pass_list,
            report_issue_url=report_issue_url,
            error=error,
            range_lo=range_lo,
            range_hi=range_hi,
            range_nan=range_nan,
            total_time=total_time,
            **self.mwa.option_dict)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--localhost", type=str, default="localhost:8080",
                        help="website base url")
    parser.add_argument("--port", type=int, default=8080, help="server connection port")
    parser.add_argument("--version-info", type=str, default="0.0.0", help="version info to display")
    args = parser.parse_args()

    # Serve the newly configured web application.
    PORT = args.port
    LOCALHOST = args.localhost
    VERSION_INFO = args.version_info
    # Configure a new minimal application with our root controller.
    config = MinimalApplicationConfigurator()
    config.register(StaticsConfigurationComponent)
    config.update_blueprint({
        'root_controller': RootController(LOCALHOST, VERSION_INFO),
        'renderers': ['kajiki'],
        'templating.kajiki.template_extension': '.xhtml',
        'templating.kajiki.force_mode': 'html5',
    })
    config.update_blueprint({
        "serve_static": True,
        "paths": {
            "static_files": os.path.join(MetalibmWebApp.SCRIPT_DIR, "public")
        }
    })

    print("Serving on port {}...".format(PORT))
    httpd = make_server('', PORT, config.make_wsgi_app())
    httpd.serve_forever()
