import sqlite3
import pytest
from langchain_core.messages import HumanMessage, AIMessage
from db.session_repo import create_session, delete_session
from db.message_repo import save_message, get_messages, AppChatMessageHistory


def test_save_and_get_messages(tmp_db):
    session_id = create_session()
    save_message(session_id, "user", "こんにちは")
    save_message(session_id, "assistant", "はい、何でしょう？")

    messages = get_messages(session_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "こんにちは"
    assert messages[1]["role"] == "assistant"


def test_get_messages_order(tmp_db):
    session_id = create_session()
    save_message(session_id, "user", "1番目")
    save_message(session_id, "assistant", "2番目")
    save_message(session_id, "user", "3番目")

    messages = get_messages(session_id)
    assert [m["content"] for m in messages] == ["1番目", "2番目", "3番目"]


def test_get_messages_empty(tmp_db):
    session_id = create_session()
    assert get_messages(session_id) == []


def test_messages_isolated_per_session(tmp_db):
    id1 = create_session()
    id2 = create_session()
    save_message(id1, "user", "セッション1のメッセージ")
    save_message(id2, "user", "セッション2のメッセージ")

    assert len(get_messages(id1)) == 1
    assert len(get_messages(id2)) == 1
    assert get_messages(id1)[0]["content"] == "セッション1のメッセージ"


def test_delete_session_cascades_messages(tmp_db):
    session_id = create_session()
    save_message(session_id, "user", "消えるメッセージ")
    delete_session(session_id)
    # セッション削除でメッセージも CASCADE 削除される
    assert get_messages(session_id) == []


def test_invalid_role_raises(tmp_db):
    session_id = create_session()
    with pytest.raises(sqlite3.IntegrityError):
        save_message(session_id, "invalid_role", "テスト")


# ── AppChatMessageHistory ──────────────────────────────────

def test_app_chat_message_history_messages(tmp_db):
    session_id = create_session()
    save_message(session_id, "user", "こんにちは")
    save_message(session_id, "assistant", "はい")

    history = AppChatMessageHistory(session_id)
    msgs = history.messages
    assert len(msgs) == 2
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "こんにちは"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "はい"


def test_app_chat_message_history_add_message(tmp_db):
    session_id = create_session()
    history = AppChatMessageHistory(session_id)
    history.add_message(HumanMessage(content="質問"))
    history.add_message(AIMessage(content="回答"))

    rows = get_messages(session_id)
    assert len(rows) == 2
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "質問"
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content"] == "回答"


def test_app_chat_message_history_empty(tmp_db):
    session_id = create_session()
    history = AppChatMessageHistory(session_id)
    assert history.messages == []


def test_app_chat_message_history_reflects_db(tmp_db):
    """履歴は DB の現在状態をそのまま反映する（キャッシュなし）。"""
    session_id = create_session()
    history = AppChatMessageHistory(session_id)
    assert len(history.messages) == 0

    save_message(session_id, "user", "追加")
    assert len(history.messages) == 1


def test_app_chat_message_history_clear_is_noop(tmp_db):
    """clear() は no-op — メッセージ削除は delete_session() で行う。"""
    session_id = create_session()
    save_message(session_id, "user", "テスト")
    history = AppChatMessageHistory(session_id)
    history.clear()
    assert len(history.messages) == 1
