from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

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


class AppChatMessageHistory(BaseChatMessageHistory):
    """既存の messages テーブルを LangChain の BaseChatMessageHistory インターフェースで包むクラス。

    セッションごとに履歴の読み書きを行い、ストリーミング応答完了後に保存する。
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    @property
    def messages(self) -> list[BaseMessage]:
        rows = get_messages(self.session_id)
        result = []
        for row in rows:
            if row["role"] == "user":
                result.append(HumanMessage(content=row["content"]))
            else:
                result.append(AIMessage(content=row["content"]))
        return result

    def add_message(self, message: BaseMessage) -> None:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        save_message(self.session_id, role, message.content)

    def clear(self) -> None:
        """セッション内メッセージの全削除は delete_session() で行うため未使用。"""
