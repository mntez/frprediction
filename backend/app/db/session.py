from collections.abc import Generator
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def get_database_url() -> str:
    configured_url = get_settings().database_url
    if not configured_url:
        raise RuntimeError(
            "DATABASE_URL is not configured. Set it in the repository-root .env file."
        )
    if configured_url.startswith("postgres://"):
        return configured_url.replace("postgres://", "postgresql+psycopg://", 1)
    if configured_url.startswith("postgresql://"):
        return configured_url.replace("postgresql://", "postgresql+psycopg://", 1)
    return configured_url


@lru_cache
def get_engine() -> Engine:
    url = get_database_url()
    options = {"pool_pre_ping": True}
    if url.startswith("sqlite") and ":memory:" in url:
        options["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **options)


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False)


def get_db_session() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
