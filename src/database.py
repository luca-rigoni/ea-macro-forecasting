"""Store the downloaded series in a local SQLite database."""

import sqlite3

SCHEMA = """
    CREATE TABLE IF NOT EXISTS series (
        series_id   TEXT PRIMARY KEY,
        source      TEXT NOT NULL,
        key         TEXT NOT NULL,
        description TEXT,
        frequency   TEXT NOT NULL,
        unit        TEXT
    );
    CREATE TABLE IF NOT EXISTS observations (
        series_id   TEXT NOT NULL,
        period      TEXT NOT NULL,
        value       REAL NOT NULL,
        PRIMARY KEY (series_id, period),
        FOREIGN KEY (series_id) REFERENCES series(series_id)
    );
"""

def connect(path):
    """Connect to the SQLite database at the given path and create tables if they don't exist."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn

if __name__ == "__main__":
    conn = connect("data/processed/test.db")
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    print(tables)
    conn.close()