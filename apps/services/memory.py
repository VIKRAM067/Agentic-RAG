import sqlite3
from pathlib import Path


class MemoryManager:
    def __init__(self):
        self.db_path = Path("./data/chat_memory.db")
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            name TEXT DEFAULT ''
        )
        """
        )
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (session_id)
            ON DELETE CASCADE
        )
        """
        )
        conn.commit()
        conn.close()

    def get_or_create_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,)
        )
        conn.commit()
        conn.close()

    def add_message(self, session_id: str, role: str, content: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        conn.close()

    def get_history(self, session_id: str, last_n: int = 20):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """SELECT role, content FROM messages
            WHERE session_id = ?
            ORDER BY timestamp ASC
            LIMIT ?""",
            (session_id, last_n),
        )
        rows = cursor.fetchall()
        conn.close()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def delete_all_sessions(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages")  # ← delete messages first
        cursor.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()

    def delete_session(self, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )  # ← comma
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()

    def rename_session(self, name: str, session_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET name = ? WHERE session_id = ?", (name, session_id)
        )
        conn.commit()
        conn.close()

    def list_sessions(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT session_id, name, created_at FROM sessions ORDER BY created_at DESC"
        )
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "session_id": row["session_id"],
                "name": row["name"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]


memory_manager = MemoryManager()
