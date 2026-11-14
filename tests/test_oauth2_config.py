import os
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from flask import Flask
from jose import jwk, jwt
import requests


CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "app" / "config" / "oauth2_config.py"
)


class OAuthKeyRefreshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.signing_keys = {}
        cls.public_keys = {}
        for kid in ("old-key", "new-key"):
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            cls.signing_keys[kid] = key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
            public_pem = key.public_key().public_bytes(
                serialization.Encoding.PEM,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
            cls.public_keys[kid] = {
                **jwk.construct(public_pem, algorithm="RS256").to_dict(),
                "kid": kid,
            }

    def setUp(self):
        self.app = Flask(__name__)
        app_module = ModuleType("app")
        app_module.app = self.app
        environment = {
            "OAUTH2_SERVER_URL": "https://auth.example.invalid",
            "OAUTH2_REALM": "test",
            "OAUTH2_CLIENT_ID": "test-client",
        }
        with patch.dict(sys.modules, {"app": app_module}):
            with patch.dict(os.environ, environment, clear=True):
                with patch("dotenv.load_dotenv"):
                    self.config = runpy.run_path(str(CONFIG_PATH))

        @self.app.route("/protected")
        @self.config["token_required_api"](roles=["Admin"])
        def protected():
            return {"success": True}

        self.client = self.app.test_client()
        get = patch("requests.get")
        self.get = get.start()
        self.addCleanup(get.stop)

    def jwks_response(self, kid):
        response = Mock()
        response.json.return_value = {"keys": [self.public_keys[kid]]}
        return response

    def token(self, signing_kid, header_kid=None):
        return jwt.encode(
            {
                "sub": "test-user",
                "aud": "test-client",
                "realm_access": {"roles": ["Admin"]},
            },
            self.signing_keys[signing_kid],
            algorithm="RS256",
            headers={"kid": header_kid or signing_kid},
        )

    def request(self, token):
        return self.client.get(
            "/protected", headers={"Authorization": f"Bearer {token}"}
        )

    def test_rotated_key_is_refetched_and_new_token_signature_is_verified(self):
        self.get.side_effect = [
            self.jwks_response("old-key"),
            self.jwks_response("new-key"),
        ]
        self.assertEqual(self.request(self.token("old-key")).status_code, 200)
        self.assertEqual(self.request(self.token("new-key")).status_code, 200)
        self.assertEqual(self.request(self.token("new-key")).status_code, 200)
        self.assertEqual(self.get.call_count, 2)
        self.get.assert_called_with(self.config["OAUTH2_JWKS_URL"], timeout=5)

    def test_unknown_key_is_rejected_after_one_refresh(self):
        self.get.return_value = self.jwks_response("old-key")
        self.assertEqual(self.request(self.token("old-key")).status_code, 200)
        self.assertEqual(
            self.request(self.token("new-key", "unknown-key")).status_code, 401
        )
        self.assertEqual(self.get.call_count, 2)

    def test_known_key_with_wrong_signature_is_still_rejected(self):
        self.get.return_value = self.jwks_response("old-key")
        self.assertEqual(self.request(self.token("old-key")).status_code, 200)
        self.assertEqual(
            self.request(self.token("new-key", "old-key")).status_code, 401
        )
        self.assertEqual(self.get.call_count, 1)

    def test_http_failure_preserves_cache_and_does_not_accept_new_token(self):
        failed_response = self.jwks_response("new-key")
        failed_response.raise_for_status.side_effect = requests.HTTPError(
            "JWKS endpoint unavailable"
        )
        self.get.side_effect = [
            self.jwks_response("old-key"),
            failed_response,
        ]
        self.assertEqual(self.request(self.token("old-key")).status_code, 200)
        self.assertEqual(self.request(self.token("new-key")).status_code, 401)
        self.assertEqual(self.request(self.token("old-key")).status_code, 200)
        self.assertEqual(self.get.call_count, 2)
        failed_response.json.assert_not_called()


class OAuthBrowserSessionTests(unittest.TestCase):
    ORIGIN = "https://ataripettingzoo.com"

    def setUp(self):
        self.app = Flask(
            __name__, template_folder=str(CONFIG_PATH.parents[1] / "templates")
        )
        self.app.config.update(TESTING=True, SECRET_KEY="stable-test-key")
        app_module = ModuleType("app")
        app_module.app = self.app
        with patch.dict(sys.modules, {"app": app_module}):
            with patch.dict(os.environ, {
                "OAUTH2_SERVER_URL": "https://auth.example.invalid",
                "OAUTH2_REALM": "test",
                "OAUTH2_CLIENT_ID": "test-client",
            }, clear=True):
                with patch("dotenv.load_dotenv"):
                    self.config = runpy.run_path(str(CONFIG_PATH))

        @self.app.route("/")
        def index():
            return "home"

        self.remote = self.config["oauth"].api
        # Keep real Authlib state generation, storage, and validation. Only
        # replace calls that would contact the identity provider.
        for name, value in (
            ("load_server_metadata", {}),
            ("fetch_access_token", {}),
            ("userinfo", {"sub": "test-user", "name": "Test Player"}),
        ):
            mock = patch.object(self.remote, name, return_value=value)
            setattr(self, name, mock.start())
            self.addCleanup(mock.stop)
        self.client = self.app.test_client()

    def start_login(self, client=None, origin=None):
        response = (client or self.client).get(
            "/login?next=/play", base_url=origin or self.ORIGIN,
        )
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        params = parse_qs(urlsplit(response.location).query)
        self.assertEqual(params["redirect_uri"], [(origin or self.ORIGIN) + "/auth"])
        return params["state"][0]

    def callback(self, state, client=None, origin=None):
        return (client or self.client).get(
            "/auth", query_string={"state": state, "code": "private-test-code"},
            base_url=origin or self.ORIGIN,
        )

    def assert_rejected(self, response):
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"Try signing in again", response.data)
        self.assertNotIn("Location", response.headers)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
        self.fetch_access_token.assert_not_called()
        self.userinfo.assert_not_called()

    def test_matching_state_completes_login_and_cannot_be_reused(self):
        state = self.start_login()
        response = self.callback(state)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.location, "/play")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.fetch_access_token.assert_called_once()
        cookie = self.client.get_cookie("session", domain="ataripettingzoo.com")
        data = self.app.session_interface.get_signing_serializer(self.app).loads(cookie.value)
        self.assertEqual(data["user"]["sub"], "test-user")
        self.fetch_access_token.reset_mock()
        self.userinfo.reset_mock()
        self.assert_rejected(self.callback(state))

    def test_missing_cookie_is_rejected_without_token_exchange(self):
        state = self.start_login()
        with self.assertLogs(self.app.logger, level="WARNING") as logs:
            self.assert_rejected(self.callback(state, client=self.app.test_client()))
        message = " ".join(logs.output)
        self.assertIn("cookie_present=False session_present=False", message)
        self.assertNotIn(state, message)
        self.assertNotIn("private-test-code", message)

    def test_changed_signing_key_is_reported_as_unreadable_session(self):
        state = self.start_login()
        self.app.secret_key = "different-replica-key"
        with self.assertLogs(self.app.logger, level="WARNING") as logs:
            self.assert_rejected(self.callback(state))
        self.assertIn("cookie_present=True session_present=False", " ".join(logs.output))

    def test_wrong_state_is_rejected_and_original_attempt_remains_valid(self):
        state = self.start_login()
        self.assert_rejected(self.callback("wrong-state"))
        self.assertEqual(self.callback(state).status_code, 302)

    def test_callback_on_different_hostname_cannot_use_login_cookie(self):
        state = self.start_login(origin="https://www.ataripettingzoo.com")
        self.assert_rejected(self.callback(state))

    def test_both_public_hostnames_can_complete_their_own_login(self):
        for origin in (self.ORIGIN, "https://www.ataripettingzoo.com"):
            with self.subTest(origin=origin):
                client = self.app.test_client()
                state = self.start_login(client=client, origin=origin)
                self.assertEqual(self.callback(state, client=client, origin=origin).status_code, 302)

    def test_direct_callback_without_state_offers_retry(self):
        self.assert_rejected(self.client.get("/auth", base_url=self.ORIGIN))

    def test_logout_without_session_identifies_client_on_both_hostnames(self):
        for origin in (self.ORIGIN, "https://www.ataripettingzoo.com"):
            with self.subTest(origin=origin):
                client = self.app.test_client()
                response = client.get("/logout", base_url=origin)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(response.headers["Referrer-Policy"], "no-referrer")
                target = urlsplit(response.location)
                self.assertEqual(target.scheme, "https")
                self.assertEqual(target.netloc, "auth.example.invalid")
                self.assertEqual(target.path, "/realms/test/protocol/openid-connect/logout")
                self.assertEqual(parse_qs(target.query), {
                    "client_id": ["test-client"],
                    "post_logout_redirect_uri": [origin + "/"],
                })

    def test_logout_clears_session_sends_id_token_and_allows_second_login(self):
        for origin in (self.ORIGIN, "https://www.ataripettingzoo.com"):
            with self.subTest(origin=origin):
                client = self.app.test_client()
                state = self.start_login(client=client, origin=origin)
                self.assertEqual(self.callback(state, client=client, origin=origin).status_code, 302)
                with client.session_transaction(base_url=origin) as session:
                    self.assertEqual(session["user"]["sub"], "test-user")
                    session["id_token"] = "test.header.signature"
                    session["roles"] = ["User"]

                response = client.get("/logout", base_url=origin)
                self.assertEqual(response.status_code, 302)
                self.assertEqual(response.headers["Cache-Control"], "no-store")
                self.assertEqual(parse_qs(urlsplit(response.location).query), {
                    "client_id": ["test-client"],
                    "post_logout_redirect_uri": [origin + "/"],
                    "id_token_hint": ["test.header.signature"],
                })
                self.assertIsNone(client.get_cookie("session", domain=urlsplit(origin).hostname))
                with client.session_transaction(base_url=origin) as session:
                    self.assertEqual(dict(session), {})

                next_state = self.start_login(client=client, origin=origin)
                self.assertNotEqual(next_state, state)
                self.assertEqual(self.callback(next_state, client=client, origin=origin).status_code, 302)


if __name__ == "__main__":
    unittest.main()
