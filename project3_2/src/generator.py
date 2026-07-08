from typing import Optional, Protocol

import dashscope
from dashscope import Generation

from .config import AppConfig


class TextGenerator(Protocol):
    """文本生成器协议，生产环境使用 Qwen，测试环境可注入假生成器。"""

    def generate(self, prompt: str) -> str:
        """根据提示词生成最终答案。"""
        ...


class QwenTextGenerator:
    """通义千问文本生成客户端，负责最终自然语言答案生成。"""

    def __init__(self, model: Optional[str] = None):
        """从配置读取模型名和 API Key。"""
        config = AppConfig.from_env()
        self.model = model or config.qwen_llm_model
        self.api_key = config.dashscope_api_key

    def generate(self, prompt: str) -> str:
        """调用 DashScope 文本生成接口，并返回 message 格式的文本内容。"""
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for QwenTextGenerator")
        dashscope.api_key = self.api_key
        response = Generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0.2,
        )
        return response.output["choices"][0]["message"]["content"]
