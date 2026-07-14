from src.config import load_settings
from src.db import initialize_database
from src.graph import build_app
from src.llm import ScriptedToolCallingModel, build_qwen_model
from src.rag import build_policy_retriever
from src.tools import build_tools


def create_app():
    """为 CLI 入口组装运行所需的数据库、检索器、工具、模型和图。"""
    settings = load_settings()
    initialize_database(settings.db_path)
    retriever = build_policy_retriever()
    tools = build_tools(db_path=settings.db_path, retriever=retriever)

    if settings.dashscope_api_key:
        model = build_qwen_model(
            api_key=settings.dashscope_api_key,
            base_url=settings.dashscope_base_url,
            model=settings.qwen_model,
        )
    else:
        print("[系统] 未检测到 DASHSCOPE_API_KEY，使用本地脚本模型演示工具调用。")
        model = ScriptedToolCallingModel()

    return build_app(model=model, tools=tools, checkpoint_path=settings.checkpoint_path)
