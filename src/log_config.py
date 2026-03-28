"""ロギング設定モジュール。

アプリ全体のログ設定を一元管理する。
logs/<YYYYMMDD>-app.log への追記とコンソール出力を同時に行う。
"""
import logging
import os
from datetime import datetime

# モジュールは sys.modules にキャッシュされるため、このフラグは
# Streamlit のリランをまたいでも True のまま保持される
_initialized = False


def setup_logging(log_dir: str) -> None:
    """ロート logger にファイルハンドラとコンソールハンドラを設定する。

    複数回呼ばれても初回のみ設定する（重複ハンドラ防止）。

    Args:
        log_dir: ログファイルを出力するディレクトリのパス。存在しない場合は作成する。
    """
    global _initialized
    if _initialized:
        return

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, datetime.now().strftime("%Y%m%d") + "-app.log")

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(fmt)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    _initialized = True
    logging.getLogger(__name__).info("Logging initialized → %s", log_file)
