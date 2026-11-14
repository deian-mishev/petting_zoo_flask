import os
import app
from functools import wraps
from flask import jsonify, redirect, render_template, request, session, url_for
from authlib.integrations.base_client.errors import MismatchingStateError
from authlib.integrations.flask_client import OAuth
from jose import jwt
import requests
from urllib.parse import urlencode

from dotenv import load_dotenv

load_dotenv()

OAUTH2_CLIENT_ID = os.getenv("OAUTH2_CLIENT_ID")
OAUTH2_CLIENT_SECRET = os.getenv("OAUTH2_CLIENT_SECRET")
OAUTH2_SERVER_URL = os.getenv("OAUTH2_SERVER_URL")
OAUTH2_REALM = os.getenv("OAUTH2_REALM")
OAUTH2_LOGOUT_URL = f"{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/logout"

OAUTH2_JWKS_URL = f"{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/certs"
_jwks_keys = None

def _get_jwks_keys(force_refresh=False):
    global _jwks_keys
    if _jwks_keys is None or force_refresh:
        response = requests.get(OAUTH2_JWKS_URL, timeout=5)
        response.raise_for_status()
        _jwks_keys = response.json()['keys']
    return _jwks_keys

app = app.app
oauth = OAuth(app)
oauth.register(
    name='api',
    client_id=OAUTH2_CLIENT_ID,
    client_secret=OAUTH2_CLIENT_SECRET,
    server_url=OAUTH2_SERVER_URL,
    server_metadata_url=f'{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/.well-known/openid-configuration',
    access_token_url=f'{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/token',
    authorize_url=f'{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/auth',
    userinfo_endpoint=f'{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/userinfo',
    client_kwargs={'scope': 'openid profile email', 'timeout': 5},
)

def get_key(token):
    headers = jwt.get_unverified_header(token)
    kid = headers['kid']
    for key in _get_jwks_keys():
        if key['kid'] == kid:
            return key
    for key in _get_jwks_keys(force_refresh=True):
        if key['kid'] == kid:
            return key
    raise ValueError("Public key not found after refreshing Keycloak signing keys")

def extract_roles(decoded):
    roles = []
    if "realm_access" in decoded:
        roles.extend(decoded["realm_access"].get("roles", []))
    if "resource_access" in decoded:
        for client_roles in decoded["resource_access"].values():
            roles.extend(client_roles.get("roles", []))
    return roles

def token_required_api(roles=None):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith("Bearer "):
                return jsonify({"error": "Missing or invalid token"}), 401

            token = auth_header.split(" ")[1]
            try:
                key = get_key(token)
                decoded = jwt.decode(
                    token,
                    key,
                    algorithms=["RS256"],
                    audience=OAUTH2_CLIENT_ID
                )
                user_roles = extract_roles(decoded)

                if roles and not any(r in user_roles for r in roles):
                    return jsonify({"error": "Missing required role"}), 403

                request.user = decoded
                request.user_roles = user_roles
            except Exception:
                app.logger.warning("Bearer token validation failed.")
                return jsonify({"error": f"Invalid token"}), 401

            return f(*args, **kwargs)
        return wrapper
    return decorator

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            if request.accept_mimetypes.accept_json and not request.accept_mimetypes.accept_html:
                return jsonify({"error": "Unauthorized"}), 401
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated


def roles_required(*roles):
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            user_roles = session.get('roles', [])
            if not any(role in user_roles for role in roles):
                return jsonify({"error": "Forbidden"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


@app.route('/auth')
def auth():
    # Inspect before Authlib consumes the state, without logging cookies,
    # authorization codes, tokens, or state values.
    cookie_present = app.config['SESSION_COOKIE_NAME'] in request.cookies
    session_present = bool(session)
    try:
        token = oauth.api.authorize_access_token()
    except MismatchingStateError:
        app.logger.warning(
            "OAuth callback rejected: host=%s scheme=%s cookie_present=%s "
            "session_present=%s state_present=%s. Check cookie delivery, "
            "login/callback hosts, and FLASK_KEY consistency across replicas/revisions.",
            request.host, request.scheme, cookie_present, session_present,
            bool(request.args.get('state')),
        )
        # Do not automatically restart OAuth: persistent cookie/configuration
        # problems would otherwise cause an endless redirect loop.
        return render_template('auth_error.html'), 400
    userinfo = oauth.api.userinfo(token=token)

    roles = []
    access_token = token.get('access_token')
    if access_token:
        try:
            key = get_key(access_token)
            decoded = jwt.decode(access_token, key, algorithms=["RS256"], audience=OAUTH2_CLIENT_ID)
            roles = extract_roles(decoded)
        except Exception:
            app.logger.warning("Access token role validation failed during login.")

    session['id_token'] = token.get('id_token')
    session['user'] = userinfo
    session['roles'] = roles

    next_url = session.pop('next', url_for('index'))
    return redirect(next_url)


@app.route('/logout')
def logout():
    id_token = session.get('id_token')
    session.clear()

    redirect_uri = url_for('index', _external=True)
    logout_params = {
        'client_id': OAUTH2_CLIENT_ID,
        'post_logout_redirect_uri': redirect_uri,
    }
    if id_token:
        logout_params['id_token_hint'] = id_token
    return redirect(f"{OAUTH2_LOGOUT_URL}?{urlencode(logout_params)}")


@app.route('/login')
def login():
    session['next'] = request.args.get('next', url_for('index'))
    redirect_uri = url_for('auth', _external=True)
    return oauth.api.authorize_redirect(redirect_uri=redirect_uri)


@app.after_request
def prevent_auth_response_caching(response):
    if request.endpoint in {'login', 'auth', 'logout'}:
        response.headers['Cache-Control'] = 'no-store'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Referrer-Policy'] = 'no-referrer'
    return response
