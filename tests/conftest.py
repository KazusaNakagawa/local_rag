import os
import sys
import tempfile
import pytest

# src/ を import パスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """テスト用に一時 DB ファイルを使うよう config を差し替える。"""
    db_path = str(tmp_path / "test_chat.db")
    monkeypatch.setattr("config.DB_PATH", db_path)

    # connection モジュールも差し替え
    import db.connection as conn_mod
    monkeypatch.setattr(conn_mod, "DB_PATH", db_path)

    from db.models import init_db
    init_db()
    return db_path
