# local-rag

Obsidian のノートをローカルで検索・質問できる RAG（Retrieval-Augmented Generation）システム。
すべての処理はローカルで完結し、外部 API への送信は一切ない。

## 構成

```bash
local-rag/
├── src/
│   ├── config.py      # パス・モデル・チャンク設定
│   ├── ingest.py      # ノートの取り込み・ベクトル化
│   ├── query.py       # CLI での質問
│   └── app.py         # Streamlit UI
├── data/              # ベクトルストア・BM25キャッシュ（.gitignore 対象）
├── requirements.txt
├── .gitignore
└── README.md
```

**検索方式：ハイブリッド検索（BM25 + ベクトル検索）**

| 手法 | 役割 |
|------|------|
| BM25（文字バイグラム） | キーワードの直接マッチ（日本語対応） |
| FAISS + MMR | 意味的な類似・多様性 |
| 両方マージ | 双方の弱点を補完 |

---

## 前提条件

- macOS（M1/M2/M3/M4）
- Python 3.11 以上
- [Ollama](https://ollama.com) インストール済み

---

## セットアップ

### 1. Ollama のインストールとモデルの準備

```bash
# Ollama インストール（未インストールの場合）
brew install ollama

# Ollama を起動（別ターミナルで常時起動しておく）
ollama serve

# 必要なモデルをダウンロード
ollama pull nomic-embed-text   # Embedding 用
ollama pull qwen2.5:7b         # LLM 用（14b でも可）
```

### 2. リポジトリのクローンと仮想環境のセットアップ

```bash
git clone <repository-url>
cd local-rag

# uv を使う場合（推奨）
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# pip を使う場合
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Obsidian Vault パスの設定

`src/config.py` を開き、`OBSIDIAN_VAULT_PATH` を自分の Vault パスに変更する。

```python
OBSIDIAN_VAULT_PATH = os.path.expanduser("~/Documents/Obsidian Vault")
```

Vault のパスは Finder でフォルダを右クリック →「情報を見る」→「場所」で確認できる。

### 4. ノートの取り込み（初回・ノート追加時）

```bash
cd src
python ingest.py
```

成功すると以下のように表示される：

```
📂 Vault パス: /Users/.../Documents/Obsidian Vault
✅ XX ファイルを読み込みました
✅ XX チャンクに分割しました
⏳ Embedding 中... (モデル: nomic-embed-text)
✅ ベクトルストアを保存しました
✅ BM25キャッシュを保存しました
```

---

## 使い方

### Streamlit UI（ブラウザ）

```bash
cd src
streamlit run app.py
```

ブラウザで `http://localhost:8501` が開く。チャット形式でノートに質問できる。
回答の下の「📎 参照したノート」で検索に使われたノートの該当箇所を確認できる。

### CLI（ターミナル）

```bash
cd src
python query.py "副業について書いたメモを教えて"
python query.py "Snowflake と Iceberg の違いは？"
```

---

## 設定のカスタマイズ

`src/config.py` で以下を変更できる。

| 設定 | デフォルト | 説明 |
|------|-----------|------|
| `LLM_MODEL` | `qwen2.5:7b` | LLM モデル。`qwen2.5:14b` にすると精度UP |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding モデル |
| `CHUNK_SIZE` | `300` | チャンクサイズ。小さいほど精度UP・速度DOWN |
| `CHUNK_OVERLAP` | `50` | チャンク間のオーバーラップ |
| `TOP_K` | `6` | 検索で取得するチャンク数 |
| `FETCH_K` | `20` | MMR の候補プール数 |

設定を変更した場合は `python ingest.py` を再実行してベクトルストアを再生成すること。

---

## ノートを追加・更新したとき

Obsidian でノートを追加・編集した後は、再度 ingest を実行して反映させる。

```bash
cd src
python ingest.py
```
