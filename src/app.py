import streamlit as st
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import pickle
from config import VECTORSTORE_PATH, PROJECT_ROOT, EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K

PROMPT_TEMPLATE = """以下のコンテキストを参考に、質問に日本語で答えてください。
コンテキストに情報がない場合は「ノートに該当する情報が見つかりませんでした」と答えてください。

コンテキスト:
{context}

質問: {question}
回答:"""

st.set_page_config(page_title="Obsidian RAG", page_icon="📓", layout="centered")
st.title("📓 Obsidian ノート検索")
st.caption(f"モデル: {LLM_MODEL}  |  Embedding: {EMBED_MODEL}  |  検索: ハイブリッド(BM25 + ベクトル)")

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")

# デバッグ用サイドバー
with st.sidebar:
    st.markdown("### 🔧 デバッグ情報")
    st.code(f"PROJECT_ROOT:\n{PROJECT_ROOT}")
    st.code(f"VECTORSTORE_PATH:\n{VECTORSTORE_PATH}")
    st.code(f"DOCS_CACHE_PATH:\n{DOCS_CACHE_PATH}")
    st.write("vectorstore 存在:", os.path.exists(VECTORSTORE_PATH))
    st.write("docs_cache 存在:", os.path.exists(DOCS_CACHE_PATH))

@st.cache_resource
def load_chain():
    if not os.path.exists(VECTORSTORE_PATH):
        return None, None

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings,
        allow_dangerous_deserialization=True
    )
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K}
    )

    # BM25 リトリーバー（キャッシュから復元）
    bm25_retriever = None
    if os.path.exists(DOCS_CACHE_PATH):
        with open(DOCS_CACHE_PATH, "rb") as f:
            all_docs = pickle.load(f)
        # 日本語対応：文字レベルのトークナイズ（スペース区切りでは日本語に効かないため）
        def japanese_tokenizer(text: str):
            chars = list(text)
            bigrams = ["".join(chars[i:i+2]) for i in range(len(chars) - 1)]
            return chars + bigrams

        bm25_retriever = BM25Retriever.from_documents(
            all_docs, preprocess_func=japanese_tokenizer
        )
        bm25_retriever.k = TOP_K

    def hybrid_retrieve(question: str):
        faiss_docs = faiss_retriever.invoke(question)
        if bm25_retriever is None:
            return faiss_docs
        bm25_docs = bm25_retriever.invoke(question)
        # 重複除去しながらマージ
        seen, combined = set(), []
        for doc in bm25_docs + faiss_docs:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        return combined[:TOP_K]

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": lambda q: format_docs(hybrid_retrieve(q)), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, hybrid_retrieve

# ベクトルストアの存在チェック
if not os.path.exists(VECTORSTORE_PATH):
    st.error("ベクトルストアが見つかりません。先に `python ingest.py` を実行してください。")
    st.stop()

if not os.path.exists(DOCS_CACHE_PATH):
    st.warning("BM25用のキャッシュがありません。`python ingest.py` を再実行してください。")

chain, retriever = load_chain()

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 入力欄
if question := st.chat_input("Obsidian ノートに質問する..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

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

    st.session_state.messages.append({"role": "assistant", "content": answer})
