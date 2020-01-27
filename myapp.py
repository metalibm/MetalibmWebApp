from wsgiref.simple_server import make_server
from tg import MinimalApplicationConfigurator
from tg import expose, TGController


FORM_TEMPLATE = """\
<html>
	<body>
		<table>
		  <tr>
		  <td width="50%">
			<form action="function" method="get" >
				<label>Function Name:</label>
				<input type="text" name="name" value="name"></input> <br/>
				<input type="submit" value="Generate!">
			</form>
		  </td>
		  <td width="50%">
			{code}
		  </td>
		</tr>
	  </table>
	<body>
</html>

"""

# RootController of our web app, in charge of serving content for /
class RootController(TGController):
    @expose(content_type="text/html")
    def index(self):
        return FORM_TEMPLATE.format(code="")

    @expose()
    def function(self, name):
        return FORM_TEMPLATE.format(code=name)

# Configure a new minimal application with our root controller.
config = MinimalApplicationConfigurator()
config.update_blueprint({
    'root_controller': RootController()
})

# Serve the newly configured web application.
print("Serving on port 8080...")
httpd = make_server('', 8080, config.make_wsgi_app())
httpd.serve_forever()
