from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Set

from neo4j import GraphDatabase

from .models import GraphPath, Relation


class InMemoryGraphStore:
    """内存图谱实现，主要用于单元测试和无 Neo4j 的本地验证。"""

    def __init__(self):
        """初始化关系列表和出边索引。"""
        self.relations: List[Relation] = []
        self.outgoing: Dict[str, List[Relation]] = defaultdict(list)

    def clear(self) -> None:
        """清空内存图谱中的全部关系。"""
        self.relations.clear()
        self.outgoing.clear()

    def upsert_relations(self, relations: Iterable[Relation]) -> None:
        """批量写入关系，并维护按主体查询的出边索引。"""
        for relation in relations:
            self.relations.append(relation)
            self.outgoing[relation.source].append(relation)

    def direct_relations(self, source: str, relation_types: Optional[Set[str]] = None) -> List[Relation]:
        """查询某个主体发出的直接关系，可按关系类型过滤。"""
        return [
            relation
            for relation in self.outgoing.get(source, [])
            if relation_types is None or relation.relation_type in relation_types
        ]

    def incoming_relations(self, target: str, relation_types: Optional[Set[str]] = None) -> List[Relation]:
        """查询指向某个实体的入边关系，可按关系类型过滤。"""
        return [
            relation
            for relation in self.relations
            if relation.target == target and (relation_types is None or relation.relation_type in relation_types)
        ]

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        relation_types: Optional[Set[str]] = None,
    ) -> List[GraphPath]:
        """用广度优先搜索查找两个实体之间的多跳路径。"""
        queue = deque([(source, [source], [], 1.0)])
        paths: List[GraphPath] = []
        while queue:
            current, nodes, relation_labels, confidence = queue.popleft()
            if len(relation_labels) >= max_hops:
                continue
            for relation in self.direct_relations(current, relation_types=relation_types):
                if relation.target in nodes:
                    continue
                next_nodes = nodes + [relation.target]
                next_relations = relation_labels + [relation.relation_type]
                next_confidence = confidence * relation.confidence
                if relation.target == target:
                    paths.append(GraphPath(next_nodes, next_relations, next_confidence))
                queue.append((relation.target, next_nodes, next_relations, next_confidence))
        return paths

class Neo4jGraphStore:
    """图数据库图谱实现，用于工程化运行和 Neo4j 查询。"""

    def __init__(self, uri: str, username: str, password: str):
        """创建 Neo4j 驱动连接，连接参数来自 .env 或命令行。"""
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self) -> None:
        """关闭 Neo4j 驱动连接。"""
        self.driver.close()

    def clear(self) -> None:
        """删除数据库中的所有节点和关系，用于重新初始化 Demo 图谱。"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def upsert_relations(self, relations: Iterable[Relation]) -> None:
        """把抽取出的 Relation 写入 Neo4j，节点用 Company 标签表示。"""
        with self.driver.session() as session:
            for relation in relations:
                session.run(
                    """
                    MERGE (source:Company {name: $source})
                    MERGE (target:Company {name: $target})
                    MERGE (source)-[rel:RELATES_TO {relation_type: $relation_type}]->(target)
                    SET rel.share_percent = $share_percent,
                        rel.confidence = $confidence,
                        rel.source_text = $source_text
                    """,
                    source=relation.source,
                    target=relation.target,
                    relation_type=relation.relation_type,
                    share_percent=relation.share_percent,
                    confidence=relation.confidence,
                    source_text=relation.source_text,
                )

    def direct_relations(self, source: str, relation_types: Optional[Set[str]] = None) -> List[Relation]:
        """查询某个公司发出的直接关系。"""
        query = """
        MATCH (source:Company {name: $source})-[rel:RELATES_TO]->(target:Company)
        WHERE $relation_types IS NULL OR rel.relation_type IN $relation_types
        RETURN source.name AS source,
               target.name AS target,
               rel.relation_type AS relation_type,
               rel.share_percent AS share_percent,
               rel.confidence AS confidence,
               rel.source_text AS source_text
        """
        with self.driver.session() as session:
            records = session.run(query, source=source, relation_types=list(relation_types) if relation_types else None)
            return [_record_to_relation(record) for record in records]

    def incoming_relations(self, target: str, relation_types: Optional[Set[str]] = None) -> List[Relation]:
        """查询指向某个公司的入边关系。"""
        query = """
        MATCH (source:Company)-[rel:RELATES_TO]->(target:Company {name: $target})
        WHERE $relation_types IS NULL OR rel.relation_type IN $relation_types
        RETURN source.name AS source,
               target.name AS target,
               rel.relation_type AS relation_type,
               rel.share_percent AS share_percent,
               rel.confidence AS confidence,
               rel.source_text AS source_text
        """
        with self.driver.session() as session:
            records = session.run(query, target=target, relation_types=list(relation_types) if relation_types else None)
            return [_record_to_relation(record) for record in records]

    def find_paths(
        self,
        source: str,
        target: str,
        max_hops: int = 3,
        relation_types: Optional[Set[str]] = None,
    ) -> List[GraphPath]:
        """使用 Cypher 变长路径查询查找实体间多跳关系。"""
        query = f"""
        MATCH path = (source:Company {{name: $source}})-[rels:RELATES_TO*1..{max_hops}]->(target:Company {{name: $target}})
        WHERE $relation_types IS NULL OR all(rel IN rels WHERE rel.relation_type IN $relation_types)
        RETURN [node IN nodes(path) | node.name] AS nodes,
               [rel IN rels | rel.relation_type] AS relations,
               reduce(confidence = 1.0, rel IN rels | confidence * coalesce(rel.confidence, 1.0)) AS confidence
        """
        with self.driver.session() as session:
            records = session.run(query, source=source, target=target, relation_types=list(relation_types) if relation_types else None)
            return [
                GraphPath(
                    nodes=list(record["nodes"]),
                    relations=list(record["relations"]),
                    confidence=float(record["confidence"]),
                )
                for record in records
            ]


def _record_to_relation(record) -> Relation:
    """把 Neo4j 查询记录转换为项目内部 Relation 对象。"""
    return Relation(
        source=record["source"],
        target=record["target"],
        relation_type=record["relation_type"],
        share_percent=record["share_percent"],
        confidence=float(record["confidence"] if record["confidence"] is not None else 1.0),
        source_text=record["source_text"] or "",
    )
