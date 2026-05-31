"""SQLite database manager for custom gesture-to-text mappings and recognition history."""

import os
import sqlite3
import sys
from typing import Dict, List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATABASE_CONFIG, DEFAULT_GESTURE_MESSAGES


class DatabaseManager:
    """Manage local SQLite storage for mappings and recognition history."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_CONFIG["path"]
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def init_database(self) -> None:
        """Create database tables and insert default gesture mappings."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS gesture_mappings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gesture_name TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    output_text TEXT NOT NULL,
                    output_speech TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS recognition_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_type TEXT NOT NULL,
                    input_name TEXT,
                    output_text TEXT,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            for gesture_name, message in DEFAULT_GESTURE_MESSAGES.items():
                cursor.execute(
                    """
                    INSERT OR IGNORE INTO gesture_mappings
                    (gesture_name, display_name, output_text, output_speech)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        gesture_name,
                        message.get("display_name", gesture_name.replace("_", " ").title()),
                        message["text"],
                        message["text"],
                    ),
                )
            conn.commit()
        print("Database initialized successfully.")

    def add_mapping(self, gesture_name: str, output_text: str, output_speech: Optional[str] = None,
                    display_name: Optional[str] = None) -> int:
        """Add or replace a gesture mapping."""
        output_speech = output_speech or output_text
        display_name = display_name or gesture_name.replace("_", " ").title()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO gesture_mappings (gesture_name, display_name, output_text, output_speech)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(gesture_name) DO UPDATE SET
                    display_name = excluded.display_name,
                    output_text = excluded.output_text,
                    output_speech = excluded.output_speech,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (gesture_name, display_name, output_text, output_speech),
            )
            conn.commit()
            return int(cursor.lastrowid or 0)

    def get_all_mappings(self) -> List[Dict]:
        """Return all gesture mappings."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gesture_mappings ORDER BY id")
            return [dict(row) for row in cursor.fetchall()]

    def get_mapping_by_gesture(self, gesture_name: str) -> Optional[Dict]:
        """Return a mapping by gesture name."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM gesture_mappings WHERE gesture_name = ?", (gesture_name,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_mapping(self, mapping_id: int, gesture_name: str, output_text: str,
                       output_speech: Optional[str] = None, display_name: Optional[str] = None) -> None:
        """Update an existing mapping."""
        output_speech = output_speech or output_text
        display_name = display_name or gesture_name.replace("_", " ").title()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE gesture_mappings
                SET gesture_name = ?, display_name = ?, output_text = ?, output_speech = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (gesture_name, display_name, output_text, output_speech, mapping_id),
            )
            conn.commit()

    def delete_mapping(self, mapping_id: int) -> None:
        """Delete a mapping by id."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM gesture_mappings WHERE id = ?", (mapping_id,))
            conn.commit()

    def add_history(self, result_type: str, input_name: str, output_text: str, confidence: float) -> None:
        """Add one recognition history record."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO recognition_history (result_type, input_name, output_text, confidence)
                VALUES (?, ?, ?, ?)
                """,
                (result_type, input_name, output_text, confidence),
            )
            conn.commit()
