import re
from typing import Any, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

BOUNDARY_MESSAGE = "我可以帮您查询订单或解答政策相关问题。"


SYSTEM_PROMPT = f"""
你是订单客服助手，只处理三类情况：
1. 订单查询：用户要查询具体订单状态、物流进度、订单详情时，必须先拿到订单号，再调用 check_order。
2. 政策/退款咨询：用户咨询退款、退货、物流规则、质保、支付、订单修改等政策时，调用 search_policy。
3. 其他闲聊或无法识别：不要调用工具，只回复“{BOUNDARY_MESSAGE}”。

不要编造订单、物流或政策内容。工具结果不足时，引导用户核对信息或联系人工客服。
""".strip()


def build_qwen_model(api_key: str, base_url: str, model: str):
    """通过阿里云百炼 OpenAI 兼容接口创建千问聊天模型。"""
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model, temperature=0)


class ScriptedToolCallingModel:
    """用于测试和无 API Key 演示的确定性本地模型。

    该模型只模拟工具调用行为，不伪装成生产级 LLM。
    当配置 DASHSCOPE_API_KEY 后，真实 CLI 会使用千问模型。
    """

    def __init__(self) -> None:
        self._tool_names: set[str] = set()

    def bind_tools(self, tools: Sequence[BaseTool]):
        self._tool_names = {tool.name for tool in tools}
        return self

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        conversation = [message for message in messages if not isinstance(message, SystemMessage)]
        last_message = conversation[-1]

        if isinstance(last_message, ToolMessage):
            return AIMessage(content=self._answer_from_tool(last_message))

        content = str(getattr(last_message, "content", ""))
        order_id = self._extract_order_id(content)
        if order_id and self._looks_like_order_context(conversation):
            return self._tool_call("check_order", {"order_id": order_id})

        if self._is_policy_question(content):
            return self._tool_call("search_policy", {"query": content})

        if self._is_order_question(content):
            return AIMessage(content="请提供订单号。")

        return AIMessage(content=BOUNDARY_MESSAGE)

    def _tool_call(self, name: str, args: dict[str, Any]) -> AIMessage:
        if name not in self._tool_names:
            return AIMessage(content=BOUNDARY_MESSAGE)
        return AIMessage(
            content="",
            tool_calls=[{"name": name, "args": args, "id": f"call_{name}"}],
        )

    def _answer_from_tool(self, message: ToolMessage) -> str:
        content = message.content
        if "未查询到订单" in content:
            return content
        if "退款政策" in content or "物流政策" in content or "质保政策" in content:
            return f"根据政策知识库：{content}"
        return f"查询结果：{content}"

    def _extract_order_id(self, content: str) -> Optional[str]:
        match = re.search(r"\d{4,20}", content)
        return match.group(0) if match else None

    def _looks_like_order_context(self, messages: list[BaseMessage]) -> bool:
        text = "\n".join(str(getattr(message, "content", "")) for message in messages[-4:])
        return bool(self._extract_order_id(text)) or any(
            keyword in text for keyword in ["订单", "物流", "快递", "发货", "签收"]
        )

    def _is_order_question(self, content: str) -> bool:
        return any(keyword in content for keyword in ["订单", "物流", "快递", "发货", "签收", "到哪"])

    def _is_policy_question(self, content: str) -> bool:
        return any(
            keyword in content
            for keyword in ["退款", "退货", "售后", "政策", "质保", "支付", "修改", "地址", "不喜欢"]
        )
