import pytest
from db.session_repo import create_session, list_sessions, update_session_title, delete_session


def test_create_session_returns_uuid(tmp_db):
    session_id = create_session()
    assert len(session_id) == 36  # UUID4 形式
    assert "-" in session_id


def test_create_session_with_title(tmp_db):
    session_id = create_session(title="テストセッション")
    sessions = list_sessions()
    assert sessions[0]["title"] == "テストセッション"
    assert sessions[0]["id"] == session_id


def test_create_session_default_title(tmp_db):
    create_session()
    sessions = list_sessions()
    assert "チャット" in sessions[0]["title"]


def test_list_sessions_order(tmp_db):
    id1 = create_session(title="first")
    id2 = create_session(title="second")
    sessions = list_sessions()
    # 新しい順なので second が先頭
    assert sessions[0]["id"] == id2
    assert sessions[1]["id"] == id1


def test_list_sessions_empty(tmp_db):
    assert list_sessions() == []


def test_update_session_title(tmp_db):
    session_id = create_session(title="旧タイトル")
    update_session_title(session_id, "新タイトル")
    sessions = list_sessions()
    assert sessions[0]["title"] == "新タイトル"


def test_delete_session(tmp_db):
    session_id = create_session()
    delete_session(session_id)
    assert list_sessions() == []


def test_delete_nonexistent_session(tmp_db):
    # 存在しない ID を削除してもエラーにならない
    delete_session("nonexistent-id")
