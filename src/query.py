import sys
import os

from config import VECTORSTORE_PATH, PROJECT_ROOT
import rag

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")


def query(question: str) -> None:
    """ベクトルストアを使って質問に回答する（CLI 用・履歴なし）。"""
    if not os.path.exists(VECTORSTORE_PATH):
        print("❌ ベクトルストアが見つかりません。先に ingest.py を実行してください。")
        return

    print(f"🔍 質問: {question}\n")

    llm, retrieve = rag.load_resources(
        VECTORSTORE_PATH, DOCS_CACHE_PATH, allow_deserialization=True
    )
    docs = retrieve(question)
    print("💬 回答:")
    for chunk in rag.stream_answer(llm, question, [], docs):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "最近のメモをまとめて"
    query(q)
