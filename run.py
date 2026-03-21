import app
from app.config.env_config import FLASK_RUN_PORT
import app.config.oauth2_config
import app.config.persistance_config
import app.routes.api
import app.routes.admin

from pyngrok import ngrok
if __name__ == '__main__':
    # public_url = ngrok.connect(FLASK_RUN_PORT)
    # print(" * ngrok tunnel:", public_url)
    app.socketio.run(app.app, host='0.0.0.0', port=FLASK_RUN_PORT)