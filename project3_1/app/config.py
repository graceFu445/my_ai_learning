"""
应用配置模块
集中读取 .env 和环境变量，避免业务代码直接依赖 os.environ。
"""
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """系统运行配置"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "FAQ RAG MVP"
    data_dir: Path = Path("data")
    dashscope_api_key: str | None = None
    dashscope_chat_model: str = "qwen-plus"
    dashscope_embedding_model: str = "text-embedding-v4"
    embedding_dimension: int | None = 1024
    top_k: int = 3
    min_similarity_score: float = 0.0

    @field_validator("embedding_dimension", mode="before")
    @classmethod
    def _blank_dimension_to_default(cls, value):
        if value == "":
            return 1024
        return value


@lru_cache
def get_settings() -> Settings:
    """获取缓存后的配置对象，避免每次请求重复解析 .env"""
    return Settings()
