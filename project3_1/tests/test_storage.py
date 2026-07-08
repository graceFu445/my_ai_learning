from pathlib import Path

from app.storage.json_store import JsonStore


def test_faq_crud_round_trip(tmp_path: Path):
    store = JsonStore(tmp_path)

    created = store.create_faq(
        question="如何退货？",
        answer="签收后7天内可申请退货。",
        tags=["售后", "退货"],
    )
    updated = store.update_faq(created["id"], answer="签收后7天内可在线申请退货。")

    assert updated["question"] == "如何退货？"
    assert updated["answer"] == "签收后7天内可在线申请退货。"
    assert store.get_faq(created["id"]) == updated

    store.delete_faq(created["id"])

    assert store.get_faq(created["id"]) is None
    assert store.list_faqs() == []


def test_session_history_appends_messages(tmp_path: Path):
    store = JsonStore(tmp_path)

    store.append_message("session-1", "user", "配送多久？")
    store.append_message("session-1", "assistant", "普通地区1-3天送达。")

    history = store.get_session("session-1")

    assert [message["role"] for message in history["messages"]] == ["user", "assistant"]
    assert history["messages"][0]["content"] == "配送多久？"
