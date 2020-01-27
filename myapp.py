from wsgiref.simple_server import make_server
from tg import MinimalApplicationConfigurator
from tg import expose, TGController

from metalibm_core.utility.ml_template import precision_parser

import metalibm_functions.ml_exp as ml_exp

FUNCTION_MAP = {
    "exp": ml_exp.ML_Exponential
}

format_list = ["binary32", "binary64"]
vector_size_list = [1, 2, 4, 8]


# RootController of our web app, in charge of serving content for /
class RootController(TGController):
    @expose("main.xhtml") #content_type="text/html")
    def index(self):
        return dict(code="no code generated", format_list=format_list, vector_size_list=vector_size_list)

    @expose("main.xhtml") #content_type="text/html")
    def function(self, name, io_format, vector_size=1):
        code = "generated {} for {} with vector_size={}".format(name, io_format, vector_size)
        if not name in FUNCTION_MAP:
            source_code = "unknown function {}".format(name)
        else:
            fct_ctor = FUNCTION_MAP[name]
            precision = precision_parser(io_format)
            vector_size = int(vector_size)
            args = fct_ctor.get_default_args(precision=precision, vector_size=vector_size)
            fct_instance = fct_ctor(args=args)
            source_code = fct_instance.generate_full_source_code()
        return dict(code=source_code, format_list=format_list, vector_size_list=vector_size_list)

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
