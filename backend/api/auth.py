"""Authentication API routes: Email/Password, Google OAuth, session validation."""

import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlalchemy.orm import Session

from core import config
from core.database import get_db
from core.security import create_access_token, decode_access_token, get_password_hash, verify_password
from models.user import User
from schemas.auth import GoogleAuthRequest, TokenResponse, UserLoginRequest, UserRegisterRequest, UserResponse

router = APIRouter(prefix="/api/auth", tags=["Auth"])
security_bearer = HTTPBearer(auto_error=False)

EMAIL_REGEX = re.compile(r"^[\w\.-]+@[\w\.-]+\.\w+$")


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security_bearer),
    db: Session = Depends(get_db),
) -> User:
    """Dependency that authenticates JWT bearer token and retrieves user."""
    if not credentials or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token. Please sign in again.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = payload.get("sub")
    try:
        user_id_int = int(user_id)
    except (ValueError, TypeError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed session token.")

    user = db.query(User).filter(User.id == user_id_int).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")
    return user


@router.get("/config")
def get_auth_config():
    """Return public client auth configuration (e.g. Google Client ID)."""
    return {
        "google_client_id": config.GOOGLE_CLIENT_ID,
        "auth_enabled": True,
    }


@router.post("/register", response_model=TokenResponse)
def register(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user with email and password, validated to NIST/OWASP standards."""
    email = req.email

    # Check for existing email
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email already exists. Please sign in instead."
        )

    hashed = get_password_hash(req.password)
    display_name = req.full_name.strip() if (req.full_name and req.full_name.strip()) else email.split("@")[0]

    user = User(
        email=email,
        hashed_password=hashed,
        full_name=display_name,
        auth_provider="local",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(req: UserLoginRequest, db: Session = Depends(get_db)):
    """Log in with existing email and password."""
    email = req.email.strip().lower()
    user = db.query(User).filter(User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    # If user registered solely with Google OAuth, they don't have a local password
    if not user.hashed_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This account uses Google Sign-In. Please click 'Sign in with Google'."
        )

    if not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password."
        )

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.post("/google", response_model=TokenResponse)
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate or register via Google Identity Services ID token."""
    if not req.credential:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Google credential token.")

    # Audience check: If GOOGLE_CLIENT_ID is configured, enforce it.
    # If not yet set in backend/.env, verify signature against Google's public certs.
    target_audience = config.GOOGLE_CLIENT_ID if config.GOOGLE_CLIENT_ID else None

    try:
        id_info = id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            target_audience,
            clock_skew_in_seconds=15,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google token verification failed: {exc}"
        )

    google_sub = id_info.get("sub")
    email = id_info.get("email", "").strip().lower()
    full_name = id_info.get("name")
    picture = id_info.get("picture")

    if not email:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Google token missing verified email.")

    # Check if user already exists
    user = db.query(User).filter((User.google_id == google_sub) | (User.email == email)).first()
    if user:
        # Link Google ID or update avatar if needed
        updated = False
        if not user.google_id and google_sub:
            user.google_id = google_sub
            updated = True
        if picture and (not user.avatar_url or user.avatar_url != picture):
            user.avatar_url = picture
            updated = True
        if full_name and (not user.full_name):
            user.full_name = full_name
            updated = True
        if updated:
            db.commit()
            db.refresh(user)
    else:
        # New Google user registration
        user = User(
            email=email,
            full_name=full_name or email.split("@")[0],
            avatar_url=picture,
            auth_provider="google",
            google_id=google_sub,
        )
        db.add(user)
        db.commit()
        db.refresh(user)

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account is deactivated.")

    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(user: User = Depends(get_current_user)):
    """Fetch profile of currently authenticated user."""
    return UserResponse.model_validate(user)
