import os
import pickle
import sys

import streamlit as st
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama, OllamaEmbeddings

sys.path.insert(0, os.path.dirname(__file__))
from config import VECTORSTORE_PATH, PROJECT_ROOT, EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K
from db import init_db, create_session, list_sessions, delete_session, update_session_title, save_message, get_messages, build_export_content, export_filename
from prompts import PROMPT_TEMPLATE

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def _assert_safe_path(path: str) -> None:
    """ファイルパスがデータディレクトリ内の通常ファイルであることを検証する。

    パストラバーサル・シンボリックリンク・ワールドライタブルを拒否する。
    """
    real = os.path.realpath(path)
    data_real = os.path.realpath(DATA_DIR)
    if not real.startswith(data_real + os.sep) and real != data_real:
        raise ValueError(f"安全でないパス: {path}")
    if os.path.islink(path):
        raise ValueError(f"シンボリックリンクは許可されていません: {path}")
    mode = os.stat(real).st_mode
    if mode & 0o002:
        raise ValueError(f"ワールドライタブルなファイルは読み込めません: {path}")


# DB 初期化
init_db()

st.set_page_config(page_title="Obsidian RAG", page_icon="📓", layout="wide")


@st.cache_resource
def load_chain():
    """FAISS と BM25 のハイブリッドリトリーバーと LLM チェーンをロードして返す。"""
    if not os.path.exists(VECTORSTORE_PATH):
        return None, None

    _assert_safe_path(VECTORSTORE_PATH)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )

    bm25_retriever = None
    if os.path.exists(DOCS_CACHE_PATH):
        # docs_cache.pkl は ingest.py がローカルの Obsidian Vault から生成する
        # 信頼済みローカルファイルのため pickle デシリアライズは安全
        _assert_safe_path(DOCS_CACHE_PATH)
        with open(DOCS_CACHE_PATH, "rb") as f:
            all_docs = pickle.load(f)

        def japanese_tokenizer(text: str):
            """日本語テキストを文字・バイグラム単位でトークナイズする。"""
            chars = list(text)
            bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
            return chars + bigrams

        bm25_retriever = BM25Retriever.from_documents(
            all_docs, preprocess_func=japanese_tokenizer
        )
        bm25_retriever.k = TOP_K

    def hybrid_retrieve(question: str):
        """BM25 と FAISS の結果をマージして重複除去したドキュメントリストを返す。"""
        faiss_docs = faiss_retriever.invoke(question)
        if bm25_retriever is None:
            return faiss_docs
        bm25_docs = bm25_retriever.invoke(question)
        seen, combined = set(), []
        for doc in bm25_docs + faiss_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        return combined[:TOP_K]

    def format_docs(docs):
        """ドキュメントをソース名付きの構造化テキストに整形する。"""
        chunks = []
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            chunks.append(f"=== Source {i}: {source} ===\n{doc.page_content}\n---")
        return "\n\n".join(chunks)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    chain = (
        {"context": lambda q: format_docs(hybrid_retrieve(q)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, hybrid_retrieve


# ── ベクトルストア存在チェック ──────────────────────────────
if not os.path.exists(VECTORSTORE_PATH):
    st.error("ベクトルストアが見つかりません。先に `python ingest.py` を実行してください。")
    st.stop()

if not os.path.exists(DOCS_CACHE_PATH):
    st.warning("BM25用のキャッシュがありません。`python ingest.py` を再実行してください。")

chain, retriever = load_chain()

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
        file_name=export_filename(),  # レンダリング時に確定、変数切り出し不要
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
    save_message(st.session_state.session_id, "user", question)
    with st.chat_message("user"):
        st.markdown(question)

    # 最初のメッセージをセッションタイトルに使用
    if len(messages) == 0:
        update_session_title(st.session_state.session_id, question[:40])

    source_docs = retriever(question)

    with st.chat_message("assistant"):
        with st.spinner("考え中..."):
            answer = chain.invoke(question)
        st.markdown(answer)

        with st.expander("📎 参照したノート"):
            for i, doc in enumerate(source_docs, 1):
                source = doc.metadata.get("source", "不明")
                filename = os.path.basename(source)
                st.markdown(f"**{i}. {filename}**")
                st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                st.divider()

    save_message(st.session_state.session_id, "assistant", answer)
