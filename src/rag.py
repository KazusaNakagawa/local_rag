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


def _merge_results(bm25_docs: list, faiss_docs: list, top_k: int) -> list:
    """BM25 と FAISS の結果をマージして重複除去したドキュメントリストを返す。

    dedup キーにソースパスと全文を使うことで、同一内容の異なるファイル間の
    誤った重複排除を防ぐ。BM25 の結果を優先して先頭に配置する。
    """
    seen: set[tuple[str, str]] = set()
    combined = []
    for doc in bm25_docs + faiss_docs:
        key = (doc.metadata.get("source", ""), doc.page_content)
        if key not in seen:
            seen.add(key)
            combined.append(doc)
    return combined[:top_k]


def load_resources(
    vectorstore_path: str,
    docs_cache_path: str | None = None,
    allow_deserialization: bool = False,
):
    """FAISS・BM25・LLM をロードして (llm, hybrid_retrieve_fn) を返す。

    Args:
        vectorstore_path: FAISS ベクトルストアのパス。
        docs_cache_path: BM25 用 pickle キャッシュのパス。None または存在しない場合は BM25 なし。
        allow_deserialization: FAISS および pickle の逆シリアライズを許可する場合は True。
            呼び出し元がパスの安全性を確認済みであることを明示するフラグ。

    Returns:
        (llm, hybrid_retrieve_fn) のタプル。
    """
    if not allow_deserialization:
        raise ValueError(
            "allow_deserialization=True を明示的に指定してください。"
            "パスが信頼済みのローカルファイルであることを確認してから呼び出してください。"
        )

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
        return _merge_results(bm25_docs, faiss_docs, TOP_K)

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
