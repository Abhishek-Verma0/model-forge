"""Security utilities: password hashing (bcrypt) and JWT tokens."""

import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import bcrypt
import jwt

from core import config

ALGORITHM = config.JWT_ALGORITHM
RESET_MINUTES = 30   # our starting point: how long a password-reset link stays valid
MIN_SECRET = 32      # HS256 key length: at least the 256-bit hash size (RFC 7518 section 3.2)


def _secret():
    """JWT_SECRET_KEY when set; otherwise one random key generated for this install
    and kept in DATA_DIR (never in the repo -- a key anyone can read lets anyone
    sign a token for any user)."""
    if len(config.JWT_SECRET_KEY) >= MIN_SECRET:
        return config.JWT_SECRET_KEY
    path = config.DATA_DIR / ".jwt_secret"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    path.parent.mkdir(parents=True, exist_ok=True)
    key = secrets.token_urlsafe(48)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(key)
    return key


SECRET_KEY = _secret()


def get_password_hash(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    pw_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(pw_bytes, salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plaintext password against its bcrypt hash."""
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Encode payload dict into a signed JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=config.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and validate a signed JWT access token. A reset token is not a session."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        return None
    return None if payload.get("purpose") else payload


def _password_mark(hashed_password):
    # changes whenever the password changes, so a used reset link stops working
    return hashlib.sha256((hashed_password or "").encode("utf-8")).hexdigest()[:16]


def create_reset_token(user) -> str:
    return create_access_token({"sub": str(user.id), "purpose": "reset",
                                "pwd": _password_mark(user.hashed_password)},
                               timedelta(minutes=RESET_MINUTES))


def reset_token_user_id(token: str, lookup) -> Optional[int]:
    """The user id a reset token is for, or None when it is invalid, expired or
    already used. `lookup(id)` returns the user (to compare the password mark)."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user = lookup(int(payload["sub"])) if payload.get("purpose") == "reset" else None
    except (jwt.PyJWTError, KeyError, ValueError):
        return None
    if user is None or payload.get("pwd") != _password_mark(user.hashed_password):
        return None
    return user.id
