import os

from werkzeug.middleware.proxy_fix import ProxyFix
from run import app as app_module

application = app_module.app

if os.getenv("TRUST_PROXY_HEADERS") == "true":
    application.wsgi_app = ProxyFix(
        application.wsgi_app,
        x_for=0,
        x_proto=1,
        x_host=0,
        x_port=0,
        x_prefix=0,
    )