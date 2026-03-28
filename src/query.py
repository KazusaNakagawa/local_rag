import os
import sys

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

from config import VECTORSTORE_PATH, EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K
from prompts import CONTEXTUALIZE_PROMPT, QA_PROMPT


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

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXTUALIZE_PROMPT
    )

    doc_prompt = PromptTemplate.from_template(
        "=== {source} ===\n{page_content}\n---"
    )
    qa_chain = create_stuff_documents_chain(
        llm, QA_PROMPT,
        document_prompt=doc_prompt,
        document_separator="\n\n",
    )
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    result = rag_chain.invoke({"input": question, "chat_history": []})
    print("💬 回答:")
    print(result["answer"])


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "最近のメモをまとめて"
    query(q)
