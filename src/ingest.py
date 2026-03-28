from langchain_community.document_loaders import ObsidianLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import pickle
from config import (
    OBSIDIAN_VAULT_PATH, VECTORSTORE_PATH, PROJECT_ROOT,
    EMBED_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)

DOCS_CACHE_PATH = os.path.join(PROJECT_ROOT, "data", "docs_cache.pkl")

def ingest():
    print(f"📂 Vault パス: {OBSIDIAN_VAULT_PATH}")

    if not os.path.exists(OBSIDIAN_VAULT_PATH):
        print(f"❌ Vault が見つかりません: {OBSIDIAN_VAULT_PATH}")
        return

    # 1. ノート読み込み
    loader = ObsidianLoader(OBSIDIAN_VAULT_PATH)
    docs = loader.load()
    print(f"✅ {len(docs)} ファイルを読み込みました")

    if len(docs) == 0:
        print("⚠️  ノートが0件です。パスを確認してください。")
        return

    # 2. チャンク分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"✅ {len(chunks)} チャンクに分割しました")

    # モデルのコンテキスト長超過を防ぐため長すぎるチャンクを截断
    MAX_CHUNK_CHARS = 1500
    truncated = 0
    for chunk in chunks:
        if len(chunk.page_content) > MAX_CHUNK_CHARS:
            chunk.page_content = chunk.page_content[:MAX_CHUNK_CHARS]
            truncated += 1
    if truncated:
        print(f"⚠️  {truncated} チャンクを {MAX_CHUNK_CHARS} 文字に截断しました")

    # 3. ベクトル化 & 保存
    print(f"⏳ Embedding 中... (モデル: {EMBED_MODEL})")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"✅ ベクトルストアを保存しました: {VECTORSTORE_PATH}")

    # BM25 用にチャンクをキャッシュ保存
    with open(DOCS_CACHE_PATH, "wb") as f:
        pickle.dump(chunks, f)
    print(f"✅ BM25キャッシュを保存しました: {DOCS_CACHE_PATH}")

if __name__ == "__main__":
    ingest()
