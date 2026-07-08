from typing import Dict, Iterable, List

from .models import GraphEvidence, RetrievalHit, ScoredDocument


class HybridScorer:
    """联合评分器：融合向量、BM25 和图谱证据对文档进行排序。"""

    def __init__(self, vector_weight: float = 0.5, bm25_weight: float = 0.3, graph_weight: float = 0.2):
        """配置三路分数权重，默认与 README 中的公式保持一致。"""
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.graph_weight = graph_weight

    def score(
        self,
        vector_hits: Iterable[RetrievalHit],
        bm25_hits: Iterable[RetrievalHit],
        graph_evidence: GraphEvidence,
    ) -> List[ScoredDocument]:
        """按文档 ID 合并向量和 BM25 命中，再叠加图谱相关性分数。"""
        by_doc: Dict[str, Dict[str, object]] = {}
        for hit in vector_hits:
            by_doc.setdefault(hit.document.id, {"document": hit.document, "vector": 0.0, "bm25": 0.0})
            by_doc[hit.document.id]["vector"] = max(float(by_doc[hit.document.id]["vector"]), hit.score)
        for hit in bm25_hits:
            by_doc.setdefault(hit.document.id, {"document": hit.document, "vector": 0.0, "bm25": 0.0})
            by_doc[hit.document.id]["bm25"] = max(float(by_doc[hit.document.id]["bm25"]), hit.score)

        results: List[ScoredDocument] = []
        for values in by_doc.values():
            document = values["document"]
            vector_score = float(values["vector"])
            bm25_score = float(values["bm25"])
            graph_score = self._graph_score_for_document(document.content, graph_evidence)
            hybrid_score = (
                self.vector_weight * vector_score
                + self.bm25_weight * bm25_score
                + self.graph_weight * graph_score
            )
            results.append(
                ScoredDocument(
                    document=document,
                    hybrid_score=hybrid_score,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    graph_score=graph_score,
                )
            )
        results.sort(key=lambda result: result.hybrid_score, reverse=True)
        return results

    def _graph_score_for_document(self, content: str, graph_evidence: GraphEvidence) -> float:
        """判断文档是否包含图谱证据实体，命中时赋予图谱置信度。"""
        if not graph_evidence.relations:
            return graph_evidence.confidence
        for relation in graph_evidence.relations:
            if relation.source in content or relation.target in content:
                return graph_evidence.confidence
        return 0.0
