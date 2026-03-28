import uuid
from datetime import datetime
from typing import Optional
from .connection import get_connection


def create_session(title: Optional[str] = None) -> str:
    """新しいセッションを作成し、セッション ID を返す。"""
    session_id = str(uuid.uuid4())
    session_title = title or f"チャット {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title) VALUES (?, ?)",
            (session_id, session_title),
        )
    return session_id


def list_sessions() -> list[dict]:
    """全セッションを新しい順で返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC, rowid DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def update_session_title(session_id: str, title: str) -> None:
    """セッションのタイトルを更新する。"""
    with get_connection() as conn:
        conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?",
            (title, session_id),
        )


def delete_session(session_id: str) -> None:
    """セッションと関連メッセージを削除する。"""
    with get_connection() as conn:
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
