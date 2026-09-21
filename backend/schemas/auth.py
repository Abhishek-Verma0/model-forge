"""Pydantic schemas for authentication requests and responses with OWASP/NIST validations."""

import re
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator

# RFC 5322 compliant regex for standard email validation
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+$"
)

# Common trivial passwords blocked by NIST recommendations
COMMON_PASSWORDS_BLOCKLIST = {
    "password", "password123", "12345678", "123456789", "qwertyuiop",
    "admin123", "welcome123", "letmein123", "iloveyou1", "abc12345",
}

# Disallow HTML/script tags or control characters in names
NAME_FORBIDDEN_CHARS = re.compile(r"[<>{}\\\x00-\x1f]")


class UserRegisterRequest(BaseModel):
    email: str = Field(..., max_length=254, description="User's email address (max 254 chars)")
    password: str = Field(..., min_length=8, max_length=128, description="Password (8-128 characters)")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name or display name")

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        cleaned = v.strip().lower()
        if not cleaned:
            raise ValueError("Email address cannot be empty.")
        if len(cleaned) > 254:
            raise ValueError("Email address cannot exceed 254 characters (RFC 5321).")
        if ".." in cleaned:
            raise ValueError("Email address contains invalid consecutive dots.")
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError("Please provide a valid email address (e.g. user@example.com).")
        return cleaned

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if len(v) > 128:
            raise ValueError("Password cannot exceed 128 characters.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter (A-Z).")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter (a-z).")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one number (0-9).")
        if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;':\",.<>/?~`\\]", v):
            raise ValueError("Password must contain at least one special character (e.g. !@#$%^&*).")
        if v.lower() in COMMON_PASSWORDS_BLOCKLIST:
            raise ValueError("This password is too common and easily guessed. Please choose a stronger password.")
        return v

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: Optional[str]) -> Optional[str]:
        if not v:
            return None
        cleaned = v.strip()
        if not cleaned:
            return None
        if len(cleaned) < 2:
            raise ValueError("Full name must be at least 2 characters long.")
        if len(cleaned) > 100:
            raise ValueError("Full name cannot exceed 100 characters.")
        if NAME_FORBIDDEN_CHARS.search(cleaned):
            raise ValueError("Full name contains invalid or potentially unsafe characters.")
        return cleaned


class UserLoginRequest(BaseModel):
    email: str = Field(..., max_length=254, description="User's email address")
    password: str = Field(..., max_length=128, description="User's password")

    @field_validator("email")
    @classmethod
    def clean_email(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password_length(cls, v: str) -> str:
        if not v:
            raise ValueError("Password is required.")
        if len(v) > 128:
            raise ValueError("Invalid credentials.")
        return v


class GoogleAuthRequest(BaseModel):
    credential: str = Field(..., description="Google ID Token issued by Google Identity Services")


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: Optional[str] = None
    avatar_url: Optional[str] = None
    auth_provider: str
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse

