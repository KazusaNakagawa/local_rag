import os

# Obsidian Vault パス
OBSIDIAN_VAULT_PATH = os.path.expanduser("~/Documents/Obsidian Vault")

# プロジェクトルート（src/ の一つ上）
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

# ベクトルストア保存先
VECTORSTORE_PATH = os.path.join(PROJECT_ROOT, "data", "vectorstore")

# モデル設定
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:7b"  # 14b に変えてもOK

# チャンク設定
CHUNK_SIZE = 300       # 細かくして一致しやすくする
CHUNK_OVERLAP = 50
MAX_CHUNK_CHARS = 1500  # コンテキスト長超過防止用の安全上限

# 検索設定
TOP_K = 6              # より多くの候補を拾う
FETCH_K = 20           # MMR の候補プール数（多いほど多様性UP）
