import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    db_path: Path
    checkpoint_path: Path
    dashscope_api_key: Optional[str]
    dashscope_base_url: str
    qwen_model: str


def load_settings() -> Settings:
    """加载运行配置，并为本地演示保留简单默认值。"""
    load_dotenv()
    return Settings(
        db_path=Path(os.getenv("ORDERS_DB_PATH", "orders.db")),
        checkpoint_path=Path(os.getenv("CHECKPOINT_DB_PATH", "checkpoints.db")),
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        dashscope_base_url=os.getenv(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        qwen_model=os.getenv("QWEN_MODEL", "qwen-plus"),
    )
