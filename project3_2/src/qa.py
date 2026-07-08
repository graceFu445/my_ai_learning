import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

from .bm25_retriever import BM25Retriever
from .config import AppConfig
from .document_loader import load_documents
from .embedding import QwenEmbeddingClient, VectorRetriever
from .generator import QwenTextGenerator, TextGenerator
from .graph_store import InMemoryGraphStore, Neo4jGraphStore
from .models import AnswerResult, Document, GraphEvidence, Relation
from .relation_extractor import RuleBasedRelationExtractor
from .scorer import HybridScorer


class GraphRAGSystem:
    """协调混合检索、图谱证据和最终答案生成。"""

    def __init__(
        self,
        documents: List[Document],
        vector_retriever: VectorRetriever,
        bm25_retriever: BM25Retriever,
        graph_store: InMemoryGraphStore,
        scorer: HybridScorer,
        text_generator: Optional[TextGenerator],
        document_relations: Optional[List[Relation]] = None,
    ):
        """保存检索器、图谱存储和生成器，并准备图谱实体索引。"""
        self.documents = documents
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.graph_store = graph_store
        self.scorer = scorer
        self.text_generator = text_generator
        self.document_relations = document_relations or RuleBasedRelationExtractor().extract(self._all_text())

    def answer(self, question: str, top_k: int = 5) -> AnswerResult:
        """执行混合检索、图谱取证和 LLM 生成，返回答案及可追踪证据。"""
        if self.text_generator is None:
            raise RuntimeError("必须配置通义千问文本生成器，不能跳过 LLM 生成环节。")

        # 第一阶段：先跑完约定的三路检索，再进入证据合并和判断。
        graph_evidence = self._graph_evidence(question)
        vector_hits = self.vector_retriever.search(question, top_k=top_k)
        bm25_hits = self.bm25_retriever.search(question, top_k=top_k)
        scored_documents = self.scorer.score(vector_hits, bm25_hits, graph_evidence)
        warnings: List[str] = []

        # 第二阶段：只组织证据并交给大语言模型生成，不在代码里手写答案模板。
        evidence_docs = scored_documents[:top_k]
        has_evidence = bool(graph_evidence.paths or evidence_docs)
        if has_evidence:
            answer = self._generate_with_llm(question, graph_evidence, evidence_docs)
        else:
            answer = "当前文本和图谱中没有提供相关信息，无法确定。"

        if "无法确定" in answer:
            warnings.append("未找到明确证据")

        return AnswerResult(
            question=question,
            answer=answer,
            scored_documents=scored_documents[:top_k],
            graph_evidence=graph_evidence,
            warnings=warnings,
        )

    def _graph_evidence(self, question: str) -> GraphEvidence:
        """按问题中出现的实体通用取证，不根据自然语言问法写分支。"""
        entities = _mentioned_entities(question, self.document_relations)
        relations = self._collect_entity_relations(entities)
        paths = self._collect_entity_paths(entities)
        return _to_graph_evidence(relations, paths)

    def _generate_with_llm(self, question, graph_evidence, scored_documents):
        """把图谱证据和高分文档整理为受约束提示词，交给 LLM 生成答案。"""
        evidence = "\n".join(f"- {path}" for path in graph_evidence.paths)
        contexts = "\n".join(
            f"- {item.document.id}: {item.document.content}" for item in scored_documents[:5]
        )
        prompt = f"""你是一个企业关系问答助手。只能基于证据回答问题，不能添加证据之外的事实。
如果证据不足以回答问题，请明确回答“当前证据无法确定”，不要猜测。
只列出证据中明确出现的关系、路径或属性，不要把“未发现其他关系/路径”当成一条答案。

问题：{question}

图谱证据：
{evidence or "- 无"}

文档证据：
{contexts or "- 无"}

请用中文给出简洁答案，并保留证据中的关键持股比例、路径或人物信息。"""
        return self.text_generator.generate(prompt)

    def _all_text(self) -> str:
        """拼接全部文档文本，供初始化时抽取图谱实体和关系。"""
        return "\n".join(document.content for document in self.documents)

    def _collect_entity_relations(self, entities: List[str]) -> List[Relation]:
        """查询每个命中实体的入边和出边，作为通用图谱关系证据。"""
        relations: List[Relation] = []
        for entity in entities:
            relations.extend(self.graph_store.direct_relations(entity))
            relations.extend(self.graph_store.incoming_relations(entity))
        return _dedupe_relations(relations)

    def _collect_entity_paths(self, entities: List[str], max_hops: int = 4) -> List[str]:
        """查询命中实体两两之间的多跳路径，供 LLM 判断路径类问题。"""
        paths: List[str] = []
        for index, source in enumerate(entities):
            for target in entities[index + 1:]:
                paths.extend(_render_paths(self.graph_store.find_paths(source, target, max_hops=max_hops)))
                paths.extend(_render_paths(self.graph_store.find_paths(target, source, max_hops=max_hops)))
        return _dedupe_strings(paths)


def build_local_system(
    company_path: Union[str, Path],
    use_neo4j: bool = False,
) -> GraphRAGSystem:
    """从文本文件构建本地问答系统，向量和生成默认使用通义千问。"""
    config = AppConfig.from_env()
    if not config.dashscope_api_key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY，无法调用通义千问完成向量检索和答案生成。")
    documents = load_documents(company_path)
    raw_text = Path(company_path).read_text(encoding="utf-8")
    relations = RuleBasedRelationExtractor().extract(raw_text)
    if use_neo4j:
        graph_store = Neo4jGraphStore(config.neo4j_uri, config.neo4j_username, config.neo4j_password)
    else:
        graph_store = InMemoryGraphStore()
        graph_store.upsert_relations(relations)
    embedding_client = QwenEmbeddingClient()
    text_generator = QwenTextGenerator()
    return GraphRAGSystem(
        documents=documents,
        vector_retriever=VectorRetriever(documents, embedding_client),
        bm25_retriever=BM25Retriever(documents),
        graph_store=graph_store,
        scorer=HybridScorer(),
        text_generator=text_generator,
        document_relations=relations,
    )


def main() -> None:
    """命令行入口：解析参数、构建系统并打印答案和追踪信息。"""
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="GraphRAG multi-hop QA demo")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--data", default=config.default_data_path, help="Path to company.txt")
    parser.add_argument("--use-neo4j", action="store_true", help="Use configured Neo4j graph instead of in-memory graph")
    parser.add_argument("--show-trace", action="store_true", help="Print retrieval and graph trace")
    args = parser.parse_args()

    use_neo4j = args.use_neo4j or os.getenv("USE_NEO4J") == "1"
    system = build_local_system(
        args.data,
        use_neo4j=use_neo4j,
    )
    result = system.answer(args.question)
    print(result.answer)
    if args.show_trace:
        print("\n[Graph Evidence]")
        for path in result.graph_evidence.paths:
            print(f"- {path}")
        print("\n[Scored Documents]")
        for scored in result.scored_documents:
            print(
                f"- {scored.document.id}: hybrid={scored.hybrid_score:.3f}, "
                f"vector={scored.vector_score:.3f}, bm25={scored.bm25_score:.3f}, "
                f"graph={scored.graph_score:.3f}"
            )
        if result.warnings:
            print("\n[Warnings]")
            for warning in result.warnings:
                print(f"- {warning}")

def _mentioned_entities(question: str, relations: Iterable[Relation]) -> List[str]:
    """基于已抽取的图谱实体，找出问题文本中直接出现的实体。"""
    entities = []
    for relation in relations:
        entities.extend([relation.source, relation.target])
    seen = set()
    mentioned = []
    for entity in entities:
        position = question.find(entity)
        if position >= 0 and entity not in seen:
            mentioned.append((position, entity))
            seen.add(entity)
    return [entity for _, entity in sorted(mentioned, key=lambda item: item[0])]


def _dedupe_relations(relations: Iterable[Relation]) -> List[Relation]:
    """按主体、客体和关系类型去重，避免重复证据干扰评分和提示词。"""
    deduped: List[Relation] = []
    seen = set()
    for relation in relations:
        key = (relation.source, relation.target, relation.relation_type)
        if key not in seen:
            deduped.append(relation)
            seen.add(key)
    return deduped


def _to_graph_evidence(relations: List[Relation], paths: List[str]) -> GraphEvidence:
    """把关系证据和路径证据统一封装成 GraphEvidence。"""
    relations = _dedupe_relations(relations)
    paths = _dedupe_strings([*_render_relation_paths(relations), *paths])
    if not relations and not paths:
        return GraphEvidence(confidence=0.0)
    return GraphEvidence(
        confidence=max([relation.confidence for relation in relations] or [1.0]),
        paths=paths,
        relations=relations,
    )


def _render_relation_paths(relations: Iterable[Relation]) -> List[str]:
    """把单跳关系渲染为可读路径，方便 trace 和 LLM prompt 使用。"""
    return [
        f"{relation.source} -> {relation.relation_type} -> {relation.target}"
        for relation in relations
    ]


def _render_paths(paths) -> List[str]:
    """把多跳路径渲染为带关系类型的链路，方便 LLM 精确引用。"""
    return [path.render() for path in paths]


def _dedupe_strings(items: Iterable[str]) -> List[str]:
    """保持原顺序去重字符串列表，用于合并单跳关系和多跳路径。"""
    deduped = []
    seen = set()
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped


if __name__ == "__main__":
    main()
