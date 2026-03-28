from datetime import datetime
from .message_repo import get_messages
from .session_repo import list_sessions


def build_export_content(session_id: str, title: str) -> str:
    """セッションのメッセージを Markdown 形式の文字列に変換する。"""
    messages = get_messages(session_id)
    lines = [f"# {title}", ""]
    for msg in messages:
        role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
        lines.append(f"{role_label}  ")
        lines.append(msg["content"])
        lines.append("")
    return "\n".join(lines)


def export_filename(prefix: str = "claude_chat") -> str:
    """現在時刻を使ったエクスポートファイル名を返す（例: 202603281205_claude_chat.md）。"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"{timestamp}_{prefix}.md"
