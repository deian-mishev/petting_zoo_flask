import os
import app
from functools import wraps
from flask import jsonify, redirect, request, session, url_for
from authlib.integrations.flask_client import OAuth
from jose import jwt
import requests
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

OAUTH2_CLIENT_ID = os.getenv("OAUTH2_CLIENT_ID")
OAUTH2_CLIENT_SECRET = os.getenv("OAUTH2_CLIENT_SECRET")
OAUTH2_SERVER_URL = os.getenv("OAUTH2_SERVER_URL")
OAUTH2_REALM = os.getenv("OAUTH2_REALM")
OAUTH2_LOGOUT_URL = f"{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/logout"

OAUTH2_JWKS_URL = f"{OAUTH2_SERVER_URL}/realms/{OAUTH2_REALM}/protocol/openid-connect/certs"
jwks_keys = requests.get(OAUTH2_JWKS_URL).json()['keys']

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
    client_kwargs={'scope': 'openid profile email'},
)

def get_key(token):
    headers = jwt.get_unverified_header(token)
    kid = headers['kid']
    for key in jwks_keys:
        if key['kid'] == kid:
            return key
    raise Exception("Public key not found")

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
    token = oauth.api.authorize_access_token()
    userinfo = oauth.api.userinfo(token=token)

    roles = []
    if 'realm_access' in userinfo:
        roles.extend(userinfo['realm_access'].get('roles', []))
    if 'resource_access' in userinfo:
        for client_roles in userinfo['resource_access'].values():
            roles.extend(client_roles.get('roles', []))
    session['id_token'] = token.get('id_token')
    session['user'] = userinfo
    session['roles'] = roles

    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    id_token = session.get('id_token')
    session.clear()

    redirect_uri = url_for('index', _external=True)
    logout_url = f"{OAUTH2_LOGOUT_URL}?post_logout_redirect_uri={quote(redirect_uri, safe='')}"
    if id_token:
        logout_url += f"&id_token_hint={quote(id_token)}"
    return redirect(logout_url)


@app.route('/login')
def login():
    next_url = request.args.get('next', url_for('index'))
    redirect_uri = url_for('auth', _external=True, next=next_url)
    return oauth.api.authorize_redirect(redirect_uri=redirect_uri)
