from .connection import get_connection


def save_message(session_id: str, role: str, content: str) -> None:
    """メッセージを DB に保存する。role は 'user' または 'assistant'。"""
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )


def get_messages(session_id: str) -> list[dict]:
    """指定セッションのメッセージを時系列順で返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages "
            "WHERE session_id = ? ORDER BY created_at ASC, id ASC",
            (session_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_recent_messages(session_id: str, n_turns: int) -> list[dict]:
    """直近 n_turns ターン分のメッセージを時系列順で返す。"""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content FROM ("
            "  SELECT role, content, id FROM messages"
            "  WHERE session_id = ? ORDER BY id DESC LIMIT ?"
            ") ORDER BY id ASC",
            (session_id, n_turns * 2),
        ).fetchall()
    return [dict(row) for row in rows]
