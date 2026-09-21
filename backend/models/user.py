"""SQLAlchemy User Model for Authentication."""

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.sql import func

from core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)
    full_name = Column(String(150), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    auth_provider = Column(String(50), default="local", nullable=False)  # "local" | "google"
    google_id = Column(String(255), unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def __repr__(self):
        return f"<User id={self.id} email='{self.email}' provider='{self.auth_provider}'>"
