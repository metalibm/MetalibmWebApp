from wsgiref.simple_server import make_server
from tg import MinimalApplicationConfigurator
from tg import expose, TGController

from metalibm_core.utility.ml_template import (
    precision_parser, target_parser, target_map
)
from metalibm_core.core.passes import Pass

import metalibm_functions.ml_exp as ml_exp
import metalibm_functions.generic_log as genlog

import metalibm_core.utility.log_report as ml_log_report

FUNCTION_MAP = {
    "exp": ml_exp.ML_Exponential,
    "log": genlog.ML_GenericLog
}

format_list = ["binary32", "binary64"]
vector_size_list = [1, 2, 4, 8]
available_pass_list = [tag for tag in Pass.get_pass_tag_list()]


option_dict = {
    "format_list": format_list,
    "vector_size_list": vector_size_list,
    "function_name_list": list(FUNCTION_MAP.keys()),
    "target_list": list(target_map.keys()),
    "available_pass_list": available_pass_list,
}



class MyLogHandler:
    def __init__(self):
        self.log_output = ""
    def write(self, msg):
        self.log_output += msg

# installing custom log handler for metalibm
ml_log_report.Log.exit_on_error = False
ml_log_report.Log.log_stream = MyLogHandler()

# RootController of our web app, in charge of serving content for /
class RootController(TGController):
    @expose("main.xhtml") #content_type="text/html")
    def index(self):
        return dict(
            code="no code generated",
            precision=format_list[0],
            registered_pass_list=["check_processor_support"],
            vector_size=1,
            target="generic",
            name=option_dict["function_name_list"][0],
            **option_dict)

    @expose("main.xhtml") #content_type="text/html")
    def function(self, name, io_format, vector_size=1, target="generic", registered_pass_list=""):
        code = "generated {} for {} with vector_size={}".format(name, io_format, vector_size)
        registered_pass_list = registered_pass_list.split(",")
        print("registered_pass_list={}".format(registered_pass_list))
        if not name in FUNCTION_MAP:
            source_code = "unknown function {}".format(name)
        else:
            # clearing logs
            ml_log_report.Log.log_stream.log_output = ""
            try:
                fct_ctor = FUNCTION_MAP[name]
                precision = precision_parser(io_format)
                vector_size = int(vector_size)
                target_class = target_parser(target)
                target_inst = target_class()
                passes = ["beforecodegen:{}".format(pass_tag) for pass_tag in registered_pass_list]
                args = fct_ctor.get_default_args(
                    precision=precision,
                    vector_size=vector_size,
                    passes=passes,
                    target=target_inst)
                fct_instance = fct_ctor(args=args)
                source_code = fct_instance.generate_full_source_code()
            except Exception as e:
                source_code = "Error encountered:\n{}\n\n{}".format(e, ml_log_report.Log.log_stream.log_output)
        return dict(
            code=source_code,
            precision=io_format,
            name=name,
            target=target,
            vector_size=vector_size,
            registered_pass_list=registered_pass_list,
            **option_dict)

# Configure a new minimal application with our root controller.
config = MinimalApplicationConfigurator()
config.update_blueprint({
    'root_controller': RootController(),
    'renderers': ['kajiki']
})

# Serve the newly configured web application.
print("Serving on port 8080...")
httpd = make_server('', 8080, config.make_wsgi_app())
httpd.serve_forever()
