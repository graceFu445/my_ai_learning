from pathlib import Path

from app.services.chat_service import ChatService
from app.storage.json_store import JsonStore


class FakeIndex:
    def __init__(self, matches):
        self.matches = matches

    def search(self, question, top_k):
        self.last_question = question
        self.last_top_k = top_k
        return self.matches


class FakeAnswerGenerator:
    def generate(self, question, matches, history):
        return f"根据FAQ回答：{matches[0]['faq']['answer']}"


def test_chat_returns_controlled_answer_and_records_history(tmp_path: Path):
    store = JsonStore(tmp_path)
    faq = store.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])
    index = FakeIndex([{"faq": faq, "score": 0.92}])
    service = ChatService(store, index, FakeAnswerGenerator(), top_k=3, min_similarity_score=0.5)

    response = service.chat("如何申请退货？", session_id="s1")

    assert response["answer"] == "根据FAQ回答：签收后7天内可申请退货。"
    assert response["matched_faqs"][0]["id"] == faq["id"]
    assert response["session_id"] == "s1"
    assert [m["role"] for m in store.get_session("s1")["messages"]] == ["user", "assistant"]


def test_chat_returns_no_answer_when_score_is_too_low(tmp_path: Path):
    store = JsonStore(tmp_path)
    faq = store.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])
    index = FakeIndex([{"faq": faq, "score": 0.2}])
    service = ChatService(store, index, FakeAnswerGenerator(), top_k=3, min_similarity_score=0.5)

    response = service.chat("你们卖电脑吗？")

    assert response["answer"] == "知识库中没有足够相关的答案。"
    assert response["matched_faqs"] == []
