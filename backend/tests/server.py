"""The real app over HTTP, for checks that must pass through FastAPI itself (the
owner guard, the JSON encoder, multipart parsing) -- calling a route function
directly skips all three. Uses `requests` (already a dependency); starlette's
TestClient needs httpx2, which is not installed.

Import this module FIRST in a test: it points DATA_DIR and the database at a fresh
temporary folder before the app (and its config) load, so nothing real is touched."""

import atexit
import os
import socket
import tempfile
import threading
import time
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="mf-test-"))
os.environ["DATA_DIR"] = str(TMP)
os.environ["DATABASE_URL"] = f"sqlite:///{(TMP / 'test.db').as_posix()}"

import requests  # noqa: E402
import uvicorn  # noqa: E402


def serve(app):
    """Start `app` on a free local port in a daemon thread; returns the base URL."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    def stop():
        # a daemon thread still logging while Python shuts down crashed the interpreter at exit
        # ("could not acquire lock for <stderr> at interpreter shutdown"): stop it first
        server.should_exit = True
        thread.join(timeout=10)
    atexit.register(stop)
    base = f"http://127.0.0.1:{port}"
    for _ in range(200):
        try:
            requests.get(base + "/health", timeout=1)
            return base
        except requests.ConnectionError:
            time.sleep(0.05)
    raise RuntimeError("test server did not start")


class Client:
    """requests with a base URL and an optional bearer token."""

    def __init__(self, base, token=None):
        self.base, self.token = base, token

    def __call__(self, method, path, **kw):
        headers = kw.pop("headers", {})
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return requests.request(method, self.base + path, headers=headers, timeout=120, **kw)

    def get(self, path, **kw):
        return self("GET", path, **kw)

    def post(self, path, **kw):
        return self("POST", path, **kw)

    def delete(self, path, **kw):
        return self("DELETE", path, **kw)

    def patch(self, path, **kw):
        return self("PATCH", path, **kw)


def account(base, email, password="Str0ng!Passw0rd", name=None):
    """Register (or sign in) a user; returns a Client carrying their token."""
    r = requests.post(base + "/api/auth/register", json={"email": email, "password": password,
                                                         "full_name": name}, timeout=30)
    if r.status_code == 400:
        r = requests.post(base + "/api/auth/login", json={"email": email, "password": password}, timeout=30)
    r.raise_for_status()
    return Client(base, r.json()["access_token"])
