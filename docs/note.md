では、M4 Air向けのローカルRAG構成を具体的に整理しましょう。まず現在の環境を確認させてください。リポジトリの構成を確認しました。Skills作成の経験があるKazusaさん向けに、M4 AirでのローカルRAG構築を実践的にまとめます。

## ローカルRAGナレッジベースの構成

**推奨アーキテクチャ**はこのようなシンプルな構成です。

```bash
Notion/Obsidian のノート
    ↓ (取得・チャンク分割)
Embeddingモデル (ローカル)
    ↓
FAISS or ChromaDB (ベクトルDB)
    ↓
Ollama (LLM推論)
    ↓
CLI or Streamlit UI で回答
```

**ステップ別にやることを整理すると：**

**Step 1 — Ollama + Embeddingモデルのセットアップ**

```bash
# Ollamaインストール (Homebrewまたは公式サイト)
brew install ollama

# LLMモデル（回答生成用）
ollama pull qwen2.5:14b      # M4 Air 16GBで快適に動く
# もしくは軽量版
ollama pull qwen2.5:7b

# Embeddingモデル（ベクトル化用）
ollama pull nomic-embed-text
```

**Step 2 — Pythonプロジェクト構成**

```bash
local-rag/
├── ingest.py          # ノートを読み込み→チャンク→ベクトル化
├── query.py           # 質問→検索→LLM回答
├── config.py          # 設定（パス、モデル名等）
├── requirements.txt
└── data/
    └── vectorstore/   # FAISSインデックス保存先
```

```txt
# requirements.txt
langchain
langchain-community
langchain-ollama
faiss-cpu
unstructured       # Markdown/Notionのパース用
```

**Step 3 — ingest.py（取り込みスクリプト）**

```python
from langchain_community.document_loaders import ObsidianLoader
# Notion exportの場合は DirectoryLoader + UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

OBSIDIAN_VAULT_PATH = "/path/to/your/obsidian/vault"
VECTORSTORE_PATH = "./data/vectorstore"

def ingest():
    # 1. ノート読み込み
    loader = ObsidianLoader(OBSIDIAN_VAULT_PATH)
    docs = loader.load()
    print(f"Loaded {len(docs)} documents")

    # 2. チャンク分割
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # 3. ベクトル化 & 保存
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print("Vectorstore saved.")

if __name__ == "__main__":
    ingest()
```

**Step 4 — query.py（検索＆回答）**

```python
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

VECTORSTORE_PATH = "./data/vectorstore"

PROMPT_TEMPLATE = """以下のコンテキストを参考に、質問に日本語で答えてください。
コンテキストに情報がない場合は「ノートに該当する情報が見つかりませんでした」と答えてください。

コンテキスト:
{context}

質問: {question}
回答:"""

def query(question: str):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH, embeddings,
        allow_dangerous_deserialization=True
    )

    llm = ChatOllama(model="qwen2.5:14b", temperature=0.1)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=PROMPT_TEMPLATE,
                input_variables=["context", "question"]
            )
        }
    )

    result = qa_chain.invoke({"query": question})
    print(result["result"])

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "最近の設計メモをまとめて"
    query(q)
```

使い方はこんな感じです：

```bash
# 初回：ノートをベクトル化
python ingest.py

# 質問する
python query.py "RAGの実装で気をつけるべきポイントは？"
python query.py "Snowflakeの設計メモある？"
```

---

**M4 Air固有のTips**

- `qwen2.5:14b` はM4 Airの16GB RAMでMetal（GPU）アクセラレーションが効くので、トークン生成が体感的にかなり速いです
- Embeddingは `nomic-embed-text` が軽量かつ日本語もそこそこ対応しています。日本語特化なら `intfloat/multilingual-e5-large` も選択肢ですが、Ollamaで直接使うには変換が要ります
- チャンクサイズ500は出発点として妥当ですが、Kazusaさんのノートの粒度に合わせて300〜800で調整すると精度が変わります（以前のRAG記事で書かれていた「精度が出ない原因」に通じる部分ですね）

**次のステップとして考えられること：**
- Streamlit UIを被せて、ブラウザから使えるようにする
- Notion APIから直接取り込む（以前のZenn記事の知見を再利用）
- n8nと連携して、ノート更新時に自動でingestを再実行する

まずはObsidianのVaultパスを指定して `ingest.py` → `query.py` を動かすところから始めてみてください。詰まったところがあれば一緒にデバッグしましょう。