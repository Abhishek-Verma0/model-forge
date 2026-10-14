"""Accounts over real HTTP: register, duplicate, wrong password, login, /me, no
token; password reset by email (single use); Google sign-in refused without a
client id; admin role from ADMIN_EMAILS.
Run from backend/: .venv\\Scripts\\python -B -m tests.test_auth"""
import re

from tests.server import Client, account, serve  # first: isolates DATA_DIR + database

import app as app_module  # noqa: E402
from api import auth  # noqa: E402
from core import config  # noqa: E402

base = serve(app_module.app)
anon = Client(base)
email, password = "tester_forge@example.com", "SecurePassword123!"

# 1. register
reg = anon.post("/api/auth/register", json={"email": email, "password": password, "full_name": "Forge Tester"})
assert reg.status_code == 200, reg.text
token = reg.json()["access_token"]
assert reg.json()["user"]["email"] == email and reg.json()["user"]["role"] == "user"
# 2. duplicate, weak password, wrong password
assert anon.post("/api/auth/register", json={"email": email, "password": password}).status_code == 400
assert anon.post("/api/auth/register", json={"email": "x@example.com", "password": "weak"}).status_code == 422
assert anon.post("/api/auth/login", json={"email": email, "password": "WrongPassword123"}).status_code == 401
# 3. login + /me, and /me without a token
assert anon.post("/api/auth/login", json={"email": email, "password": password}).status_code == 200
me = Client(base, token).get("/api/auth/me")
assert me.status_code == 200 and me.json()["full_name"] == "Forge Tester"
assert anon.get("/api/auth/me").status_code == 401

# 4. password reset: 503 until SMTP is configured, then a single-use link by email
assert anon.post("/api/auth/forgot", json={"email": email}).status_code == 503
sent = []


class FakeSMTP:  # stands in for the mail server: records what would be sent
    def __init__(self, host, port, timeout=None):
        self.host = host

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, user, password):
        pass

    def send_message(self, msg):
        sent.append(msg)


auth.smtplib.SMTP = FakeSMTP
config.SMTP_HOST, config.SMTP_FROM = "smtp.test", "noreply@test"
r = anon.post("/api/auth/forgot", json={"email": email})
assert r.status_code == 200 and len(sent) == 1 and sent[0]["To"] == email, r.text
assert anon.post("/api/auth/forgot", json={"email": "nobody@example.com"}).json() == r.json()  # same answer
assert len(sent) == 1                                                       # ...but no email
reset_token = re.search(r"\?reset=(\S+)", sent[0].get_content()).group(1)
new_pw = "N3w!Password-2026"
assert anon.post("/api/auth/reset", json={"token": reset_token, "password": "weak"}).status_code == 422
ok = anon.post("/api/auth/reset", json={"token": reset_token, "password": new_pw})
assert ok.status_code == 200 and ok.json()["access_token"], ok.text
assert anon.post("/api/auth/reset", json={"token": reset_token, "password": "An0ther!Password"}).status_code == 400
assert anon.post("/api/auth/login", json={"email": email, "password": new_pw}).status_code == 200
assert anon.post("/api/auth/login", json={"email": email, "password": password}).status_code == 401
assert Client(base, reset_token).get("/api/auth/me").status_code == 401   # a reset link is not a session

# 5. Google: no client id configured -> refused (audience could not be checked)
config.GOOGLE_CLIENT_ID = ""
g = anon.post("/api/auth/google", json={"credential": "x.y.z"})
assert g.status_code == 503 and "GOOGLE_CLIENT_ID" in g.json()["detail"], g.text

# 6. admin role comes from ADMIN_EMAILS; admins manage accounts, others can't
user = Client(base, ok.json()["access_token"])
admin = account(base, "boss@example.com")
assert admin.get("/api/admin/users").status_code == 403
config.ADMIN_EMAILS = {"boss@example.com"}
users = admin.get("/api/admin/users")
assert users.status_code == 200 and {u["email"] for u in users.json()} == {email, "boss@example.com"}
uid = next(u["id"] for u in users.json() if u["email"] == email)
assert admin.post(f"/api/admin/users/{uid}/active", json={"active": False}).json()["is_active"] is False
assert user.get("/api/auth/me").status_code == 403                    # deactivated
admin.post(f"/api/admin/users/{uid}/active", json={"active": True})
assert user.get("/api/auth/me").status_code == 200
assert user.get("/api/admin/users").status_code == 403
print("auth check passed")
