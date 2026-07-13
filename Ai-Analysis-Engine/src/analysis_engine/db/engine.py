import os
from pathlib import Path
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from dotenv import load_dotenv
load_dotenv()

_default_db_path = Path(__file__).parent.parent.parent.parent / "verum.db"
DATABASE_URL = os.environ.get("DATABASE_URL",f"sqlite:///{_default_db_path}")

_content_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_content_args, echo = False)
_SessionFactory = sessionmaker(bind=engine, autoflush=False, autocommit=False)


@contextmanager
def get_session() -> Session:
    session = _SessionFactory()
    try:
        yield session
        session.commit()
        raise
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


