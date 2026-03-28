import os
import sys
import threading

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.dirname(__file__))
from config import VECTORSTORE_PATH, PROJECT_ROOT, EMBED_MODEL, LLM_MODEL, CHAT_HISTORY_TURNS
from db import (
    init_db, create_session, list_sessions, delete_session, update_session_title,
    get_messages, AppChatMessageHistory, save_message,
    build_export_content, export_filename,
)
import rag

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")

# DB 初期化
init_db()

st.set_page_config(page_title="Obsidian RAG", page_icon="📓", layout="wide")

# バックグラウンドスレッドと Streamlit UI の両方から安全にアクセスできるよう
# モジュールレベルの set で管理する（st.session_state はスレッドから参照不可）
_pending_sessions: set[str] = set()
_pending_lock = threading.Lock()


@st.cache_resource
def _load_cached_resources():
    """RAG リソースをロードしてキャッシュする。パス検証は load_resources() 内で実施。"""
    return rag.load_resources(VECTORSTORE_PATH, DOCS_CACHE_PATH, allow_deserialization=True)


# ── ベクトルストア存在チェック ──────────────────────────────
if not os.path.exists(VECTORSTORE_PATH):
    st.error("ベクトルストアが見つかりません。先に `python ingest.py` を実行してください。")
    st.stop()

if not os.path.exists(DOCS_CACHE_PATH):
    st.warning("BM25用のキャッシュがありません。`python ingest.py` を再実行してください。")

llm, hybrid_retrieve = _load_cached_resources()
if llm is None:
    st.error("リソースのロードに失敗しました。`python ingest.py` を実行してください。")
    st.stop()

# ── セッション初期化 ────────────────────────────────────────
if "session_id" not in st.session_state:
    recent = list_sessions()
    st.session_state.session_id = recent[0]["id"] if recent else create_session()


def _run_chain(session_id: str, question: str, chat_history: list) -> None:
    """バックグラウンドスレッドで RAG チェーンを実行して DB に保存する。

    st.write_stream() はスレッドから使えないため invoke_answer() を使用する。
    完了・エラーどちらの場合も _pending_sessions から session_id を削除する。
    """
    try:
        search_query = rag.contextualize_query(llm, question, chat_history)
        source_docs = hybrid_retrieve(search_query)
        answer = rag.invoke_answer(llm, question, chat_history, source_docs)
        save_message(session_id, "assistant", answer)
    except Exception:
        pass
    finally:
        with _pending_lock:
            _pending_sessions.discard(session_id)


# ── サイドバー：セッション一覧 ──────────────────────────────
with st.sidebar:
    st.title("📓 Obsidian RAG")
    st.caption(f"モデル: {LLM_MODEL} | Embedding: {EMBED_MODEL}")

    if st.button("＋ 新しいチャット", use_container_width=True):
        st.session_state.session_id = create_session()
        st.rerun()

    st.divider()

    # 現在のセッションをエクスポート
    current_sessions = list_sessions()
    current_title = next(
        (s["title"] for s in current_sessions if s["id"] == st.session_state.session_id),
        "chat",
    )
    export_content = build_export_content(st.session_state.session_id, current_title)
    st.download_button(
        label="📥 エクスポート",
        data=export_content.encode("utf-8"),
        file_name=export_filename(),
        mime="text/markdown",
        use_container_width=True,
    )

    st.divider()
    st.markdown("#### 履歴")

    sessions = current_sessions
    for s in sessions:
        col1, col2 = st.columns([5, 1])
        is_active = s["id"] == st.session_state.session_id
        with _pending_lock:
            is_pending = s["id"] in _pending_sessions
        label = f"**{s['title']}**" if is_active else s["title"]
        if is_pending:
            label = f"⏳ {label}"
        if col1.button(label, key=f"sel_{s['id']}", use_container_width=True):
            st.session_state.session_id = s["id"]
            st.rerun()
        if col2.button("🗑", key=f"del_{s['id']}"):
            delete_session(s["id"])
            if is_active:
                remaining = [x for x in sessions if x["id"] != s["id"]]
                st.session_state.session_id = remaining[0]["id"] if remaining else create_session()
            st.rerun()


# ── メインエリア ────────────────────────────────────────────
st.title("📓 Obsidian ノート検索")


@st.fragment(run_every=1)
def _chat_area() -> None:
    """チャット表示エリア。1 秒ごとに DB を再取得して完了した回答を表示する。

    _pending_sessions に現在のセッションが含まれる間は生成中インジケーターを表示する。
    """
    messages = get_messages(st.session_state.session_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with _pending_lock:
        is_pending = st.session_state.session_id in _pending_sessions
    if is_pending:
        with st.chat_message("assistant"):
            st.markdown("⏳ 回答を生成中...")


_chat_area()

# 入力欄
if question := st.chat_input("Obsidian ノートに質問する..."):
    session_id = st.session_state.session_id

    # 現在の会話履歴を取得（新しいユーザー質問は含まない）
    history = AppChatMessageHistory(session_id, max_turns=CHAT_HISTORY_TURNS)
    chat_history = history.messages
    is_first_message = len(chat_history) == 0

    # ユーザーメッセージを即時 DB に保存（セッション切替が発生しても消えない）
    save_message(session_id, "user", question)

    if is_first_message:
        update_session_title(session_id, question[:40])

    # バックグラウンドスレッドで LLM 処理を実行
    with _pending_lock:
        _pending_sessions.add(session_id)
    threading.Thread(
        target=_run_chain,
        args=(session_id, question, chat_history),
        daemon=True,
    ).start()

    st.rerun()
