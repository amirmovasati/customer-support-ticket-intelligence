# region: Setup and Data Loading
# Loads the cleaned dataset that will become our historical ticket log.

from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "src"))

from database import engine, SessionLocal, Ticket, init_db
from business_decision import assign_priority

CLEAN_DATA_FILE = PROJECT_ROOT / "data" / "bitext_customer_support_clean.csv"
df = pd.read_csv(CLEAN_DATA_FILE)
# endregion


# region: Compute Priority for Every Row
# Reuses the phase-4 rule-based logic to assign a priority to each historical ticket.

df["priority"] = df.apply(lambda row: assign_priority(row["intent"], row["flags"]), axis=1)
# endregion


# region: Bulk Insert
# Loads all rows into the database in one batch, much faster than inserting one at a time.
# Skips re-importing if the table is already populated, so this script is safe to rerun.

def populate():
    init_db()
    session = SessionLocal()
    try:
        existing_count = session.query(Ticket).count()
        if existing_count > 100:
            print(f"Database already has {existing_count} tickets — skipping bulk import.")
            return

        tickets = [
            Ticket(
                instruction=row["instruction"],
                category=row["category"],
                intent=row["intent"],
                priority=row["priority"],
                draft_response=None,
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(tickets)
        session.commit()
        print(f"Inserted {len(tickets)} tickets into the database.")
    finally:
        session.close()

if __name__ == "__main__":
    populate()
# endregion