import os
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

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


if __name__ == "__main__":
    unittest.main()
