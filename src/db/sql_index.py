from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from src.config import SQLITE_INDEX_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(db_path: Union[str, Path] = SQLITE_INDEX_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Union[str, Path] = SQLITE_INDEX_PATH) -> None:
    conn = _connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
              session_id TEXT PRIMARY KEY,
              created_at TEXT NOT NULL,
              last_used_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              doc_id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              source_file TEXT NOT NULL,
              uploaded_path TEXT,
              created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
              chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL,
              source_file TEXT NOT NULL,
              page INTEGER,
              chunk_index INTEGER,
              modality TEXT,
              priority TEXT,
              text TEXT NOT NULL,
              chroma_id TEXT
            )
            """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON chunks(session_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_docs_session ON documents(session_id)")
        conn.commit()
    finally:
        conn.close()


def upsert_session(session_id: str, db_path: Union[str, Path] = SQLITE_INDEX_PATH) -> None:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        now = _now_iso()
        conn.execute(
            """
            INSERT INTO sessions(session_id, created_at, last_used_at)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET last_used_at=excluded.last_used_at
            """,
            (session_id, now, now),
        )
        conn.commit()
    finally:
        conn.close()


def insert_document(
    *,
    session_id: str,
    source_file: str,
    uploaded_path: Optional[Union[str, Path]] = None,
    db_path: Union[str, Path] = SQLITE_INDEX_PATH,
) -> None:
    init_db(db_path)
    upsert_session(session_id, db_path=db_path)
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO documents(session_id, source_file, uploaded_path, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (session_id, source_file, str(uploaded_path) if uploaded_path is not None else None, _now_iso()),
        )
        conn.commit()
    finally:
        conn.close()


def insert_chunks(
    *,
    session_id: str,
    source_file: str,
    rows: Iterable[Tuple[Optional[int], Optional[int], Optional[str], Optional[str], str, Optional[str]]],
    db_path: Union[str, Path] = SQLITE_INDEX_PATH,
) -> None:
    """
    rows: iterable of (page, chunk_index, modality, priority, text, chroma_id)
    """
    init_db(db_path)
    upsert_session(session_id, db_path=db_path)
    conn = _connect(db_path)
    try:
        conn.executemany(
            """
            INSERT INTO chunks(session_id, source_file, page, chunk_index, modality, priority, text, chroma_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ((session_id, source_file, page, chunk_index, modality, priority, text, chroma_id) for (page, chunk_index, modality, priority, text, chroma_id) in rows),
        )
        conn.commit()
    finally:
        conn.close()


def get_documents_for_session(session_id: str, db_path: Union[str, Path] = SQLITE_INDEX_PATH) -> List[dict]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT source_file, uploaded_path, created_at
            FROM documents
            WHERE session_id = ?
            ORDER BY created_at DESC
            """,
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def count_chunks_by_modality(session_id: str, db_path: Union[str, Path] = SQLITE_INDEX_PATH) -> List[dict]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT COALESCE(modality, 'unknown') AS modality, COUNT(*) AS chunk_count
            FROM chunks
            WHERE session_id = ?
            GROUP BY COALESCE(modality, 'unknown')
            ORDER BY chunk_count DESC
            """,
            (session_id,),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def top_pages_by_chunk_count(
    session_id: str,
    *,
    limit: int = 10,
    db_path: Union[str, Path] = SQLITE_INDEX_PATH,
) -> List[dict]:
    init_db(db_path)
    conn = _connect(db_path)
    try:
        cur = conn.execute(
            """
            SELECT source_file, page, COUNT(*) AS chunk_count
            FROM chunks
            WHERE session_id = ?
            GROUP BY source_file, page
            ORDER BY chunk_count DESC
            LIMIT ?
            """,
            (session_id, int(limit)),
        )
        return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()

