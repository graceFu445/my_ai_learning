from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from src.db import get_order


def build_tools(db_path: Path, retriever: Any):
    """构建 LangChain 工具，并通过闭包绑定本地数据库和检索器。"""

    @tool
    def check_order(order_id: str) -> str:
        """根据订单号查询订单状态、商品信息与物流信息。"""
        order = get_order(db_path, order_id)
        if order is None:
            return f"未查询到订单 {order_id}，请检查订单号是否正确。"
        return (
            f"订单 {order['order_id']}（{order['items']}）：当前状态为 {order['status']}。"
            f"物流信息：{order['logistics_info']}。"
        )

    @tool
    def search_policy(query: str) -> str:
        """查询退款、退货、物流规则、质保、支付、订单修改等政策知识。"""
        docs = retriever.invoke(query)
        return "\n".join(document.page_content for document in docs)

    return [check_order, search_policy]
