from datetime import UTC, datetime
from pathlib import Path

from llama_index.core.embeddings import BaseEmbedding

from app.rag import index as rag_index
from app.rag.index import DashScopeFaissIndex


class DeterministicEmbedding(BaseEmbedding):
    """测试用 embedding：不用联网，也能稳定驱动 FAISS 相似度检索。"""

    def __init__(self):
        super().__init__(model_name="test-embedding", embed_batch_size=4)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        if "退货" in text:
            return [1.0, 0.0, 0.0, 0.0]
        return [0.0, 1.0, 0.0, 0.0]


def _faq(faq_id: str, question: str, answer: str) -> dict:
    now = datetime.now(UTC).isoformat()
    return {
        "id": faq_id,
        "question": question,
        "answer": answer,
        "tags": [],
        "created_at": now,
        "updated_at": now,
    }


def test_faiss_index_rebuild_search_reload_and_clear(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rag_index, "DashScopeEmbedding", lambda **_: DeterministicEmbedding())
    index_dir = tmp_path / "faiss_index"
    faqs = [
        _faq("faq_return", "如何退货？", "签收后7天内可申请退货。"),
        _faq("faq_delivery", "配送多久？", "普通地区1到3天送达。"),
    ]

    index = DashScopeFaissIndex(index_dir, api_key="test-key", embedding_model="test-model", embedding_dimension=4)
    index.rebuild(faqs)

    assert index.search("我想退货怎么办？", 1)[0]["faq"]["id"] == "faq_return"

    reloaded = DashScopeFaissIndex(index_dir, api_key="test-key", embedding_model="test-model", embedding_dimension=4)
    assert reloaded.search("退货流程是什么？", 1)[0]["faq"]["id"] == "faq_return"

    reloaded.rebuild([])

    assert reloaded.search("退货流程是什么？", 1) == []
