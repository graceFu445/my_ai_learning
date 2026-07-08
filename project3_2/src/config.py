import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


DEFAULT_ENV_PATH = ".env"


def load_dotenv(path: Union[str, Path] = DEFAULT_ENV_PATH) -> None:
    """读取 .env 文件，把尚未存在于环境变量中的键值写入 os.environ。"""
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_quotes(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AppConfig:
    """集中保存运行配置，避免 API Key、模型名和 Neo4j 参数散落在业务代码里。"""

    dashscope_api_key: str = ""
    qwen_embedding_model: str = "text-embedding-v3"
    qwen_llm_model: str = "qwen-turbo"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    default_data_path: str = "data/company.txt"

    @classmethod
    def from_env(cls, dotenv_path: Optional[Union[str, Path]] = DEFAULT_ENV_PATH) -> "AppConfig":
        """从 .env 和系统环境变量构建配置对象。"""
        if dotenv_path is not None:
            load_dotenv(dotenv_path)
        return cls(
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY", ""),
            qwen_embedding_model=os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v3"),
            qwen_llm_model=os.getenv("QWEN_LLM_MODEL", "qwen-turbo"),
            neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
            neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
            default_data_path=os.getenv("DEFAULT_DATA_PATH", "data/company.txt"),
        )


def _strip_quotes(value: str) -> str:
    """去掉 .env 值两侧成对的单引号或双引号。"""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
