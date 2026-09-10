# region: Setup and Engine
# Defines the database connection. SQLite for the free, zero-setup demo;
# swapping to PostgreSQL later only requires changing DATABASE_URL.

from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_FILE = PROJECT_ROOT / "data" / "tickets.db"
DATABASE_URL = f"sqlite:///{DB_FILE}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
# endregion


# region: Schema Definition
# One table, `tickets`, matching the schema we agreed on.

class Ticket(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    instruction = Column(String, nullable=False)
    category = Column(String, nullable=False)
    intent = Column(String, nullable=False)
    priority = Column(String, nullable=False)
    draft_response = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
# endregion


# region: Table Creation
# Creates the tickets.db file and the table inside it, if they don't already exist.

def init_db():
    Base.metadata.create_all(engine)
# endregion


# region: Insert Function
# Saves one processed ticket (already classified, prioritized, and drafted) to the database.

def save_ticket(instruction: str, category: str, intent: str, priority: str, draft_response: str = None) -> int:
    session = SessionLocal()
    try:
        ticket = Ticket(
            instruction=instruction,
            category=category,
            intent=intent,
            priority=priority,
            draft_response=draft_response,
        )
        session.add(ticket)
        session.commit()
        session.refresh(ticket)
        return ticket.id
    finally:
        session.close()
# endregion


# region: Quick Test
# Initializes the DB and inserts one sample row to confirm everything works.

if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_FILE}")

    new_id = save_ticket(
        instruction="I want to cancel my order",
        category="ORDER",
        intent="cancel_order",
        priority="High",
        draft_response="I'll be happy to help you cancel your order...",
    )
    print(f"Inserted test ticket with id: {new_id}")
# endregion