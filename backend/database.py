import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "data" / "students.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _migrate_old_schema(conn):
    """Migra schema antigo (embedding em students) para multi-embedding."""
    print("[DB] Migrando schema antigo para multi-embedding...")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            photo_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute(
        "INSERT INTO students_new (id, name, created_at) "
        "SELECT id, name, created_at FROM students"
    )
    conn.execute(
        "INSERT INTO face_embeddings (student_id, embedding, photo_path, created_at) "
        "SELECT id, embedding, photo_path, created_at FROM students"
    )
    conn.executescript("""
        DROP TABLE students;
        ALTER TABLE students_new RENAME TO students;
    """)
    conn.commit()
    print("[DB] Migração concluída.")


def init_db():
    conn = get_conn()

    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if "students" in tables:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(students)").fetchall()]
        if "embedding" in cols:
            _migrate_old_schema(conn)

    conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            embedding BLOB NOT NULL,
            photo_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            confidence REAL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            UNIQUE(student_id, date)
        );
        CREATE INDEX IF NOT EXISTS idx_embeddings_student ON face_embeddings(student_id);
        CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
    """)
    conn.commit()
    return conn