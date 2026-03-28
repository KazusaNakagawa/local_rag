import os
import sys

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

sys.path.insert(0, os.path.dirname(__file__))
from config import VECTORSTORE_PATH, PROJECT_ROOT, EMBED_MODEL, LLM_MODEL, CHAT_HISTORY_TURNS
from db import init_db, create_session, list_sessions, delete_session, update_session_title, get_messages, AppChatMessageHistory, build_export_content, export_filename
import rag

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")


# DB 初期化
init_db()

st.set_page_config(page_title="Obsidian RAG", page_icon="📓", layout="wide")


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
# 再起動後は直近セッションを復元し、存在しない場合のみ新規作成
if "session_id" not in st.session_state:
    recent = list_sessions()
    st.session_state.session_id = recent[0]["id"] if recent else create_session()

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
    current_title = next((s["title"] for s in current_sessions if s["id"] == st.session_state.session_id), "chat")
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
        label = f"**{s['title']}**" if is_active else s["title"]
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

# DB からメッセージ復元
messages = get_messages(st.session_state.session_id)
for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 入力欄
if question := st.chat_input("Obsidian ノートに質問する..."):
    history = AppChatMessageHistory(st.session_state.session_id, max_turns=CHAT_HISTORY_TURNS)
    chat_history = history.messages
    is_first_message = len(chat_history) == 0

    with st.chat_message("user"):
        st.markdown(question)

    if is_first_message:
        update_session_title(st.session_state.session_id, question[:40])

    with st.chat_message("assistant"):
        if chat_history:
            with st.spinner("質問を解析中..."):
                search_query = rag.contextualize_query(llm, question, chat_history)
        else:
            search_query = question

        with st.spinner("ノートを検索中..."):
            source_docs = hybrid_retrieve(search_query)

        answer = ""
        try:
            answer = st.write_stream(rag.stream_answer(llm, question, chat_history, source_docs))
        except Exception:
            st.error("回答の生成に失敗しました。Ollama が起動しているか確認してください。")
            raise
        finally:
            history.add_message(HumanMessage(content=question))
            if answer:
                history.add_message(AIMessage(content=answer))

        with st.expander("📎 参照したノート"):
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get("source", "不明")
                filename = os.path.basename(source)
                st.markdown(f"**{i}. {filename}**")
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.divider()
