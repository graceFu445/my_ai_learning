from dataclasses import dataclass


POLICY_DOCUMENTS = [
    "退款政策：自签收之日起 7 天内，且商品未拆封，可发起退款申请。",
    "物流政策：满 50 美元免邮，标准配送一般为 3-5 个工作日。",
    "质保政策：电子类商品享受 1 年制造商质保服务。",
    "支付政策：支持信用卡、PayPal 和支付宝。",
    "订单修改：订单状态变为“已发货”后不可再修改订单信息。",
]


@dataclass(frozen=True)
class PolicyDocument:
    page_content: str
    source: str = "policy_doc"


class KeywordPolicyRetriever:
    """面向演示政策知识库的轻量本地 RAG 检索器。"""

    def __init__(self, documents: list[str]) -> None:
        self._documents = [PolicyDocument(page_content=document) for document in documents]

    def invoke(self, query: str) -> list[PolicyDocument]:
        query = query.lower()
        keyword_groups = [
            ("退款", "退货", "售后", "不喜欢", "退钱"),
            ("物流", "配送", "快递", "邮", "发货"),
            ("质保", "保修", "维修"),
            ("支付", "付款", "信用卡", "支付宝", "paypal"),
            ("修改", "地址", "改订单", "取消"),
        ]

        matched: list[PolicyDocument] = []
        for document in self._documents:
            content = document.page_content.lower()
            if any(keyword in query and keyword in content for group in keyword_groups for keyword in group):
                matched.append(document)

        return matched or self._documents[:2]


def build_policy_retriever() -> KeywordPolicyRetriever:
    return KeywordPolicyRetriever(POLICY_DOCUMENTS)
