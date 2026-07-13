from analysis_engine.db.engine import engine
from analysis_engine.db.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
    print("[db] Database initialized.")
