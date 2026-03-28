import os
import sys

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser

from config import VECTORSTORE_PATH, EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K
from prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT


def format_docs(docs):
    """ドキュメントをソース名付きの構造化テキストに整形する。"""
    chunks = []
    for i, doc in enumerate(docs, 1):
        source = os.path.basename(doc.metadata.get("source", "unknown"))
        chunks.append(f"=== Source {i}: {source} ===\n{doc.page_content}\n---")
    return "\n\n".join(chunks)


def query(question: str):
    """ベクトルストアを使って質問に回答する（CLI 用・履歴なし）。"""
    if not os.path.exists(VECTORSTORE_PATH):
        print("❌ ベクトルストアが見つかりません。先に ingest.py を実行してください。")
        return

    print(f"🔍 質問: {question}\n")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K}
    )
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1)

    docs = retriever.invoke(question)
    answer = (QA_PROMPT | llm | StrOutputParser()).invoke(
        {"input": question, "chat_history": [], "context": format_docs(docs)}
    )
    print("💬 回答:")
    print(answer)


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "最近のメモをまとめて"
    query(q)
