import app
from app.config.env_config import FLASK_RUN_PORT
import app.config.oauth2_config
import app.config.persistance_config
import app.routes.api
import app.routes.admin

if __name__ == '__main__':
    # from pyngrok import ngrok
    # from werkzeug.middleware.proxy_fix import ProxyFix
    # app.app.wsgi_app = ProxyFix(app.app.wsgi_app, x_proto=1, x_host=1)
    # public_url = ngrok.connect(FLASK_RUN_PORT)
    # print(" * ngrok tunnel:", public_url)
    app.socketio.run(app.app, host='0.0.0.0', port=FLASK_RUN_PORT)
