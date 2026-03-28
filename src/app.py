import logging
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
from log_config import setup_logging
import rag

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")

setup_logging(LOG_DIR)
logger = logging.getLogger(__name__)

# DB 初期化
init_db()

st.set_page_config(page_title="Obsidian RAG", page_icon="📓", layout="wide")


@st.cache_resource
def _get_shared_state() -> dict:
    """Streamlit の rerun をまたいで保持するサーバーグローバルな状態。

    st.session_state はスレッドから参照できないため、バックグラウンドスレッドと
    UI スレッドの共有データをここで管理する。
    """
    return {
        "pending": set(),          # 処理中の session_id の集合
        "lock": threading.Lock(),  # pending・status の排他制御
        "status": {},              # session_id -> 現在の処理ステップ名
    }


@st.cache_resource
def _load_cached_resources():
    """ベクトルストアと BM25 リトリーバーをロードしてキャッシュする。

    LLM は ChatOllama のスレッドセーフ性が保証されないため、ここではキャッシュせず
    スレッドごとに rag.make_llm() で新規生成する。
    """
    return rag.load_resources(VECTORSTORE_PATH, DOCS_CACHE_PATH, allow_deserialization=True)


# ── ベクトルストア存在チェック ──────────────────────────────
if not os.path.exists(VECTORSTORE_PATH):
    st.error("ベクトルストアが見つかりません。先に `python ingest.py` を実行してください。")
    st.stop()

if not os.path.exists(DOCS_CACHE_PATH):
    st.warning("BM25用のキャッシュがありません。`python ingest.py` を再実行してください。")

hybrid_retrieve = _load_cached_resources()
if hybrid_retrieve is None:
    st.error("リソースのロードに失敗しました。`python ingest.py` を実行してください。")
    st.stop()

# ── セッション初期化 ────────────────────────────────────────
if "session_id" not in st.session_state:
    recent = list_sessions()
    st.session_state.session_id = recent[0]["id"] if recent else create_session()


def _run_chain(session_id: str, question: str, chat_history: list) -> None:
    """バックグラウンドスレッドで RAG チェーンを実行して DB に保存する。

    各ステップで logger と shared["status"] を更新し、処理状況を追跡できるようにする。
    """
    shared = _get_shared_state()
    sid = session_id[:8]

    def _set_status(msg: str) -> None:
        with shared["lock"]:
            shared["status"][session_id] = msg

    try:
        logger.info("[%s] chain start — question: %.60s", sid, question)

        # スレッドごとに新しい LLM インスタンスを生成（ChatOllama のスレッドセーフ性確保）
        llm = rag.make_llm()

        _set_status("質問を解析中...")
        logger.info("[%s] contextualize_query start", sid)
        search_query = rag.contextualize_query(llm, question, chat_history)
        logger.info("[%s] contextualize_query done → %.60s", sid, search_query)

        _set_status("ノートを検索中...")
        logger.info("[%s] hybrid_retrieve start", sid)
        source_docs = hybrid_retrieve(search_query)
        logger.info("[%s] hybrid_retrieve done → %d docs", sid, len(source_docs))

        _set_status("回答を生成中...")
        logger.info("[%s] invoke_answer start", sid)
        answer = rag.invoke_answer(llm, question, chat_history, source_docs)
        logger.info("[%s] invoke_answer done → %d chars", sid, len(answer))

        save_message(session_id, "assistant", answer)
        logger.info("[%s] answer saved to DB", sid)

    except Exception as e:
        logger.exception("[%s] chain failed: %s", sid, e)
        try:
            save_message(session_id, "assistant", "⚠️ エラーが発生しました。もう一度お試しください。")
        except Exception:
            logger.exception("[%s] failed to save error message", sid)
    finally:
        with shared["lock"]:
            shared["pending"].discard(session_id)
            shared["status"].pop(session_id, None)
        logger.info("[%s] chain finished (pending removed)", sid)


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

    shared = _get_shared_state()
    sessions = current_sessions
    for s in sessions:
        col1, col2 = st.columns([5, 1])
        is_active = s["id"] == st.session_state.session_id
        with shared["lock"]:
            is_pending = s["id"] in shared["pending"]
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

    _pending_sessions に現在のセッションが含まれる間は処理ステップを表示する。
    """
    shared = _get_shared_state()
    messages = get_messages(st.session_state.session_id)
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    with shared["lock"]:
        is_pending = st.session_state.session_id in shared["pending"]
        status_msg = shared["status"].get(st.session_state.session_id, "処理中...")

    if is_pending:
        with st.chat_message("assistant"):
            st.markdown(f"⏳ {status_msg}")


_chat_area()

# 入力欄：処理中はインプットを無効化して多重送信を防ぐ
shared = _get_shared_state()
with shared["lock"]:
    _current_pending = st.session_state.session_id in shared["pending"]

if _current_pending:
    st.chat_input("処理中です...", disabled=True)
elif question := st.chat_input("Obsidian ノートに質問する..."):
    session_id = st.session_state.session_id

    logger.info("[%s] user submitted: %.60s", session_id[:8], question)

    # 現在の会話履歴を取得 (新しいユーザー質問は含まない)
    history = AppChatMessageHistory(session_id, max_turns=CHAT_HISTORY_TURNS)
    chat_history = history.messages
    is_first_message = len(chat_history) == 0

    # ユーザーメッセージを即時 DB に保存 (セッション切替が発生しても消えない)
    save_message(session_id, "user", question)
    logger.info("[%s] user message saved to DB", session_id[:8])

    if is_first_message:
        update_session_title(session_id, question[:40])

    # バックグラウンドスレッドで LLM 処理を実行
    with shared["lock"]:
        shared["pending"].add(session_id)
        shared["status"][session_id] = "質問を解析中..."

    threading.Thread(
        target=_run_chain,
        args=(session_id, question, chat_history),
        daemon=True,
    ).start()
    logger.info("[%s] background thread started", session_id[:8])

    st.rerun()
