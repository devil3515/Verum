from analysis_engine.db.engine import engine, SessionLocal
from analysis_engine.db.models import Base

def init_db():
    Base.metadata.create_all(bind=engine)
    # Seed default user if not exists
    from analysis_engine.db.models import User
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == "default-user").first()
        if not user:
            user = User(id="default-user", email="user@verum.ai", name="Verum User", plan="pro")
            db.add(user)
            db.commit()
    except Exception as e:
        db.rollback()
        print(f"[db init error] Failed to seed default user: {e}")
    finally:
        db.close()
    print("[db] Database initialized.")
