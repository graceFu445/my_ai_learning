"""
DashScope 回答生成模块
把检索命中的 FAQ 和会话历史组织成受控提示词，再调用 Qwen 生成自然语言回答。
"""
from __future__ import annotations

import dashscope
from dashscope import Generation


class DashScopeAnswerGenerator:
    """基于 DashScope Generation API 的 FAQ 回答生成器"""

    def __init__(self, api_key: str | None, model: str | None):
        if not api_key or not model:
            raise ValueError("DASHSCOPE_API_KEY and DASHSCOPE_CHAT_MODEL are required for chat.")
        self.api_key = api_key
        self.model = model

    def generate(self, question: str, matches: list[dict], history: list[dict]) -> str:
        """根据命中 FAQ 和最近会话历史生成回答。"""
        self._configure_api_key()
        context = "\n\n".join(
            f"FAQ {index + 1}\n问题：{match['faq']['question']}\n答案：{match['faq']['answer']}\n标签：{', '.join(match['faq'].get('tags', []))}"
            for index, match in enumerate(matches)
        )
        recent_history = "\n".join(f"{item['role']}: {item['content']}" for item in history[-6:])
        response = Generation.call(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个智能FAQ客服助手。只能根据提供的FAQ内容回答。"
                        "如果FAQ没有覆盖用户问题，明确说明知识库没有足够信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"会话历史：\n{recent_history}\n\n"
                        f"FAQ内容：\n{context}\n\n"
                        f"用户问题：{question}"
                    ),
                },
            ],
            result_format="message",
        )
        if getattr(response, "status_code", 200) != 200:
            code = getattr(response, "code", "Unknown")
            message = getattr(response, "message", "DashScope generation request failed.")
            raise RuntimeError(f"DashScope chat request failed: {code} - {message}")
        return self._extract_text(response)

    def _configure_api_key(self):
        dashscope.api_key = self.api_key

    def _extract_text(self, response) -> str:
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text

        output = getattr(response, "output", None)
        if isinstance(output, dict):
            choices = output.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = message.get("content")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return "".join(
                        piece.get("text", "") if isinstance(piece, dict) else str(piece)
                        for piece in content
                    )
        if output is not None and hasattr(output, "choices"):
            choices = getattr(output, "choices")
            if choices:
                message = getattr(choices[0], "message", None)
                if message is not None:
                    content = getattr(message, "content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        return "".join(
                            piece.get("text", "") if isinstance(piece, dict) else str(piece)
                            for piece in content
                        )
        raise RuntimeError("DashScope response does not contain answer text.")


class MissingDashScopeAnswerGenerator:
    """缺少 DashScope 配置时使用的占位生成器，保证服务仍可启动"""

    def generate(self, question: str, matches: list[dict], history: list[dict]) -> str:
        raise RuntimeError("DASHSCOPE_API_KEY and DASHSCOPE_CHAT_MODEL are required for chat.")
