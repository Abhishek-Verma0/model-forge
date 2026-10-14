"""Authentication API routes: Email/Password, Google OAuth, session validation,
password recovery, and the guard every data route sits behind."""

import smtplib
from email.message import EmailMessage
from typing import Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from sqlalchemy.orm import Session

from core import config, store
from core.database import get_db
from core.security import (RESET_MINUTES, create_access_token, create_reset_token, decode_access_token,
                           get_password_hash, reset_token_user_id, verify_password)
from models.user import User
from schemas.auth import (ForgotPasswordRequest, GoogleAuthRequest, ResetPasswordRequest, TokenResponse,
                          UserLoginRequest, UserRegisterRequest, UserResponse)

router = APIRouter(prefix="/api/auth", tags=["Auth"])
admin_router = APIRouter(prefix="/api/admin", tags=["Admin"])
security_bearer = HTTPBearer(auto_error=False)


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


async def owner_guard(request: Request, user: User = Depends(get_current_user)) -> User:
    """Router-wide dependency for every data route: signed in, and when the request
    names a dataset -- `id` in the path, query, JSON body or form -- it must be this
    user's. Someone else's dataset answers 404, exactly like an unknown id, so ids
    can't be probed. Deny by default: a new route on a guarded router is covered
    without remembering to add a check. FastAPI has already read the body before
    dependencies run; Starlette caches it, so reading it here costs nothing."""
    ds_id = request.path_params.get("id") or request.query_params.get("id")
    if ds_id is None and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            try:
                body = await request.json()
            except ValueError:
                body = None
            ds_id = body.get("id") if isinstance(body, dict) else None
        elif ctype.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
            ds_id = (await request.form()).get("id")
    if ds_id is not None and not store.owned_by(str(ds_id), user.id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Dataset not found.")
    return user


def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Administrators only.")
    return user


def _session(user):
    token = create_access_token({"sub": str(user.id), "email": user.email})
    return TokenResponse(access_token=token, token_type="bearer", user=UserResponse.model_validate(user))


@router.get("/config")
def get_auth_config():
    """Public client auth configuration. The Google button is shown only when the
    server can verify Google tokens (GOOGLE_CLIENT_ID set here)."""
    return {
        "google_client_id": config.GOOGLE_CLIENT_ID,
        "auth_enabled": True,
        "password_reset": bool(config.SMTP_HOST),
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
    return _session(user)


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
    return _session(user)


@router.post("/google", response_model=TokenResponse)
def google_auth(req: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Authenticate or register via Google Identity Services ID token."""
    if not req.credential:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing Google credential token.")
    if not config.GOOGLE_CLIENT_ID:
        # Google's guidance: verify that the token's audience is your own client id.
        # Without one to compare, a token issued to any other site would be accepted.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="Google sign-in is not set up on this server (GOOGLE_CLIENT_ID in backend/.env).")

    try:
        id_info = id_token.verify_oauth2_token(
            req.credential,
            google_requests.Request(),
            config.GOOGLE_CLIENT_ID,
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

    if not email or id_info.get("email_verified") is False:
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
    return _session(user)


@router.get("/me", response_model=UserResponse)
def get_current_user_profile(user: User = Depends(get_current_user)):
    """Fetch profile of currently authenticated user."""
    return UserResponse.model_validate(user)


def _send_mail(to, subject, body):
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = config.SMTP_FROM or config.SMTP_USER, to, subject
    msg.set_content(body)
    with smtplib.SMTP(config.SMTP_HOST, config.SMTP_PORT, timeout=20) as smtp:
        if config.SMTP_TLS:
            smtp.starttls()
        if config.SMTP_USER:
            smtp.login(config.SMTP_USER, config.SMTP_PASSWORD)
        smtp.send_message(msg)


@router.post("/forgot")
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    """Email a single-use reset link. The answer is the same whether or not the
    email has an account (OWASP: don't reveal which emails are registered)."""
    if not config.SMTP_HOST:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Password reset by email is not set up on this server (SMTP_HOST in backend/.env).")
    user = db.query(User).filter(User.email == req.email).first()
    if user and user.is_active:
        link = f"{(config.FRONTEND_URL or config.FRONTEND_URL_DEFAULT).rstrip('/')}/?reset={create_reset_token(user)}"
        try:
            _send_mail(user.email, "Reset your Model Forge password",
                       f"Someone asked to reset the password for {user.email}.\n\n"
                       f"Open this link within {RESET_MINUTES} minutes to choose a new password:\n{link}\n\n"
                       "If it wasn't you, ignore this email - your password stays the same.")
        except (OSError, smtplib.SMTPException) as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"Could not send the email: {exc}") from None
    return {"sent": True, "detail": "If an account exists for that email, a reset link is on its way."}


@router.post("/reset", response_model=TokenResponse)
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """New password from a reset link; signs the user in. The link stops working
    once used (it is tied to the old password) or after RESET_MINUTES."""
    user_id = reset_token_user_id(req.token, lambda i: db.query(User).filter(User.id == i).first())
    if user_id is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "This reset link is invalid, expired or already used.")
    user = db.query(User).filter(User.id == user_id).first()
    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Account is deactivated.")
    user.hashed_password = get_password_hash(req.password)
    db.commit()
    db.refresh(user)
    return _session(user)


@admin_router.get("/users", response_model=list[UserResponse])
def list_users(_admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    """Every account (no data -- administrators manage accounts, not other people's datasets)."""
    return [UserResponse.model_validate(u) for u in db.query(User).order_by(User.id).all()]


@admin_router.post("/users/{user_id}/active", response_model=UserResponse)
def set_active(user_id: int, active: bool = Body(..., embed=True), admin: User = Depends(require_admin),
               db: Session = Depends(get_db)):
    if user_id == admin.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "You can't deactivate your own account.")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found.")
    user.is_active = bool(active)
    db.commit()
    db.refresh(user)
    return UserResponse.model_validate(user)
