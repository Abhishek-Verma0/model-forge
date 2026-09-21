"""Pydantic schemas for authentication requests and responses."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class UserRegisterRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., min_length=6, description="Password (at least 6 characters)")
    full_name: Optional[str] = Field(None, description="Full name or display name")


class UserLoginRequest(BaseModel):
    email: str = Field(..., description="User's email address")
    password: str = Field(..., description="User's password")


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
