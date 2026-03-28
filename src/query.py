from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import os
from config import VECTORSTORE_PATH, EMBED_MODEL, LLM_MODEL, TOP_K, FETCH_K

PROMPT_TEMPLATE = """以下のコンテキストを参考に、質問に日本語で答えてください。
コンテキストに情報がない場合は「ノートに該当する情報が見つかりませんでした」と答えてください。

コンテキスト:
{context}

質問: {question}
回答:"""

def format_docs(docs):
    """ドキュメントのリストを改行区切りの文字列に整形する。"""
    return "\n\n".join(doc.page_content for doc in docs)

def query(question: str):
    """ベクトルストアを使って質問に回答する。"""
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
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke(question)
    print("💬 回答:")
    print(result)

if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "最近のメモをまとめて"
    query(q)
