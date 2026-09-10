# region: Setup
# Reuses the same database engine already configured in database.py.

from pathlib import Path
import sys
from sqlalchemy import text

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from database import engine
# endregion


# region: Report Queries
# A few business-relevant reports a support manager would actually want to see.

def ticket_count_by_category():
    query = text("""
        SELECT category, COUNT(*) AS ticket_count
        FROM tickets
        GROUP BY category
        ORDER BY ticket_count DESC
    """)
    with engine.connect() as conn:
        return conn.execute(query).fetchall()


def priority_breakdown_by_category():
    query = text("""
        SELECT category, priority, COUNT(*) AS ticket_count
        FROM tickets
        GROUP BY category, priority
        ORDER BY category, 
            CASE priority WHEN 'High' THEN 1 WHEN 'Medium' THEN 2 ELSE 3 END
    """)
    with engine.connect() as conn:
        return conn.execute(query).fetchall()


def top_high_priority_intents():
    query = text("""
        SELECT intent, COUNT(*) AS high_priority_count
        FROM tickets
        WHERE priority = 'High'
        GROUP BY intent
        ORDER BY high_priority_count DESC
        LIMIT 5
    """)
    with engine.connect() as conn:
        return conn.execute(query).fetchall()
# endregion


# region: Run Reports
if __name__ == "__main__":
    print("Tickets by category:")
    for row in ticket_count_by_category():
        print(f"  {row.category}: {row.ticket_count}")

    print("\nTop 5 intents by High-priority volume:")
    for row in top_high_priority_intents():
        print(f"  {row.intent}: {row.high_priority_count}")
# endregion