from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Document:
    """检索系统中的文档片段，通常对应 company.txt 的一个段落。"""

    id: str
    content: str
    metadata: Dict[str, str]


@dataclass(frozen=True)
class Relation:
    """图谱中的企业关系边，包含主体、客体、关系类型和持股比例。"""

    source: str
    target: str
    relation_type: str
    share_percent: Optional[float] = None
    confidence: float = 1.0
    source_text: str = ""


@dataclass(frozen=True)
class RetrievalHit:
    """单路检索命中结果，记录文档、分数和来源。"""

    document: Document
    score: float
    source: str


@dataclass(frozen=True)
class GraphPath:
    """图谱多跳路径，保存节点序列、关系序列和路径置信度。"""

    nodes: List[str]
    relations: List[str]
    confidence: float = 1.0

    def render(self) -> str:
        """把路径渲染为“节点 -> 关系 -> 节点”的可读格式。"""
        if not self.nodes:
            return ""
        parts = [self.nodes[0]]
        for relation, node in zip(self.relations, self.nodes[1:]):
            parts.extend([relation, node])
        return " -> ".join(parts)


@dataclass(frozen=True)
class GraphEvidence:
    """图谱检索得到的证据包，供联合评分和 LLM prompt 使用。"""

    confidence: float
    paths: List[str] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)


@dataclass(frozen=True)
class ScoredDocument:
    """联合评分后的文档，保留各路分数便于 trace 和调试。"""

    document: Document
    hybrid_score: float
    vector_score: float
    bm25_score: float
    graph_score: float


@dataclass(frozen=True)
class AnswerResult:
    """问答系统的最终返回对象，包含答案、证据、排序文档和告警。"""

    question: str
    answer: str
    scored_documents: List[ScoredDocument]
    graph_evidence: GraphEvidence
    warnings: List[str] = field(default_factory=list)
