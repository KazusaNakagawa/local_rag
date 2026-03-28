"""RAG パイプライン。Streamlit に依存しない純粋なロジック層。

app.py・query.py の両エントリーポイントから呼び出される。
"""
import os
import pickle

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K
from prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT


def _japanese_tokenizer(text: str) -> list[str]:
    """日本語テキストを文字・バイグラム単位でトークナイズする。"""
    chars = list(text)
    bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    return chars + bigrams


def load_resources(vectorstore_path: str, docs_cache_path: str | None = None):
    """FAISS・BM25・LLM をロードして (llm, hybrid_retrieve_fn) を返す。

    Args:
        vectorstore_path: FAISS ベクトルストアのパス。
        docs_cache_path: BM25 用 pickle キャッシュのパス。None または存在しない場合は BM25 なし。

    Returns:
        (llm, hybrid_retrieve_fn) のタプル。
    """
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        vectorstore_path, embeddings,
        allow_dangerous_deserialization=True,
    )
    faiss_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K},
    )

    bm25_retriever = None
    if docs_cache_path and os.path.exists(docs_cache_path):
        # docs_cache.pkl は ingest.py がローカルの Obsidian Vault から生成する
        # 信頼済みローカルファイルのため pickle デシリアライズは安全
        with open(docs_cache_path, "rb") as f:
            all_docs = pickle.load(f)
        bm25_retriever = BM25Retriever.from_documents(
            all_docs, preprocess_func=_japanese_tokenizer
        )
        bm25_retriever.k = TOP_K

    def hybrid_retrieve(question: str) -> list:
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

    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)
    return llm, hybrid_retrieve


def format_docs(docs) -> str:
    """ドキュメントをソース名付きの構造化テキストに整形する。"""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        chunks.append(f"=== Source {i}: {source} ===\n{doc.page_content}\n---")
    return "\n\n".join(chunks)


def contextualize_query(llm, question: str, chat_history: list) -> str:
    """会話履歴があるときフォローアップ質問を単独の質問に言い換える。

    履歴がない場合は LLM を呼ばずそのまま返す。
    """
    if not chat_history:
        return question
    return (CONTEXTUALIZE_PROMPT | llm | StrOutputParser()).invoke(
        {"input": question, "chat_history": chat_history}
    )


def stream_answer(llm, question: str, chat_history: list, docs: list):
    """QA チェーンの回答をトークン単位でストリーム返却するジェネレータ。"""
    yield from (QA_PROMPT | llm | StrOutputParser()).stream(
        {"input": question, "chat_history": chat_history, "context": format_docs(docs)}
    )
