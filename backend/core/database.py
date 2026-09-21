"""SQL Database session management and engine initialization."""

import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from core import config

logger = logging.getLogger("modelforge.database")

# PostgreSQL engine creation with pool recycling and pre-ping
# Fallback to local SQLite if PostgreSQL connection fails
try:
    if config.DATABASE_URL.startswith("sqlite"):
        engine = create_engine(
            config.DATABASE_URL,
            connect_args={"check_same_thread": False},
        )
    else:
        engine = create_engine(
            config.DATABASE_URL,
            pool_pre_ping=True,
            pool_recycle=3600,
        )
        # Test connection
        with engine.connect() as conn:
            pass
        logger.info(f"Connected to SQL database at {config.DATABASE_URL.split('@')[-1]}")
except Exception as e:
    logger.warning(
        f"Failed to connect to primary DATABASE_URL ({e}). Falling back to SQLite for local development."
    )
    sqlite_path = config.DATA_DIR / "modelforge.db"
    sqlite_url = f"sqlite:///{sqlite_path.as_posix()}"
    engine = create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI dependency for yielding transactional DB sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all registered database tables."""
    # Ensure all models are imported before creating tables
    import models.user  # noqa: F401
    Base.metadata.create_all(bind=engine)
