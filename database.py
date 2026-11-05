import sqlite3
from pathlib import Path


# Project database location
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "electricity.db"


def create_database():
    """Create the database and readings table."""

    DATA_DIR.mkdir(exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS meter_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reading_date TEXT UNIQUE NOT NULL,
            meter_reading REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    connection.commit()
    connection.close()


def validate_meter_reading(reading_date, meter_reading):
    """Check whether a new meter reading is valid."""

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    try:
        # Find the previous reading
        cursor.execute("""
            SELECT meter_reading
            FROM meter_readings
            WHERE reading_date < ?
            ORDER BY reading_date DESC
            LIMIT 1
        """, (reading_date,))

        previous = cursor.fetchone()

        # Find the next reading
        cursor.execute("""
            SELECT meter_reading
            FROM meter_readings
            WHERE reading_date > ?
            ORDER BY reading_date ASC
            LIMIT 1
        """, (reading_date,))

        following = cursor.fetchone()

        if previous and meter_reading < previous[0]:
            return False, "New meter reading cannot be lower than the previous reading."

        if following and meter_reading > following[0]:
            return False, "New meter reading cannot be higher than the next reading."

        if meter_reading < 0:
            return False, "Meter reading cannot be negative."

        return True, None

    finally:
        connection.close()


if __name__ == "__main__":
    create_database()
    print("Database created successfully!")