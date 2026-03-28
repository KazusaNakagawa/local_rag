import re
from db.session_repo import create_session
from db.message_repo import save_message
from db.export import build_export_content, export_filename


def test_export_filename_format(tmp_db):
    name = export_filename()
    assert re.match(r"^\d{12}_claude_chat\.md$", name)


def test_export_filename_custom_prefix(tmp_db):
    name = export_filename(prefix="my_export")
    assert name.endswith("_my_export.md")


def test_build_export_content_structure(tmp_db):
    session_id = create_session(title="テストセッション")
    save_message(session_id, "user", "こんにちは")
    save_message(session_id, "assistant", "はい、何でしょう？")

    content = build_export_content(session_id, "テストセッション")
    assert "# テストセッション" in content
    assert "**You**" in content
    assert "こんにちは" in content
    assert "**Assistant**" in content
    assert "はい、何でしょう？" in content


def test_build_export_content_empty_session(tmp_db):
    session_id = create_session(title="空セッション")
    content = build_export_content(session_id, "空セッション")
    assert "# 空セッション" in content
    assert "**You**" not in content


def test_build_export_content_message_order(tmp_db):
    session_id = create_session()
    save_message(session_id, "user", "1番目")
    save_message(session_id, "assistant", "2番目")
    save_message(session_id, "user", "3番目")

    content = build_export_content(session_id, "順序テスト")
    pos1 = content.index("1番目")
    pos2 = content.index("2番目")
    pos3 = content.index("3番目")
    assert pos1 < pos2 < pos3
