from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app


class FakeIndex:
    def __init__(self):
        self.faqs = []

    def rebuild(self, faqs):
        self.faqs = faqs

    def search(self, question, top_k):
        if not self.faqs:
            return []
        return [{"faq": self.faqs[0], "score": 0.91}]


class FakeAnswerGenerator:
    def generate(self, question, matches, history):
        return matches[0]["faq"]["answer"]


def test_faq_crud_and_chat_api(tmp_path: Path):
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            index=FakeIndex(),
            answer_generator=FakeAnswerGenerator(),
        )
    )

    created = client.post(
        "/faqs",
        json={"question": "如何退货？", "answer": "签收后7天内可申请退货。", "tags": ["售后"]},
    )
    assert created.status_code == 201
    faq_id = created.json()["id"]

    listed = client.get("/faqs")
    assert listed.json()[0]["id"] == faq_id

    updated = client.put(f"/faqs/{faq_id}", json={"answer": "签收后7天内可在线申请退货。"})
    assert updated.json()["answer"] == "签收后7天内可在线申请退货。"

    chat = client.post("/chat", json={"question": "怎么退货？", "session_id": "s1"})
    assert chat.status_code == 200
    assert chat.json()["matched_faqs"][0]["id"] == faq_id

    deleted = client.delete(f"/faqs/{faq_id}")
    assert deleted.status_code == 204


def test_csv_import_api(tmp_path: Path):
    client = TestClient(
        create_app(
            data_dir=tmp_path,
            index=FakeIndex(),
            answer_generator=FakeAnswerGenerator(),
        )
    )

    response = client.post(
        "/faqs/import",
        files={
            "file": (
                "faqs.csv",
                "question,answer,tags\n配送多久？,普通地区1-3天送达。,配送;时效\n",
                "text/csv",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["imported_count"] == 1
