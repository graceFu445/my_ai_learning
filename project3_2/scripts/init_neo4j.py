import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig
from src.document_loader import load_documents
from src.graph_store import Neo4jGraphStore
from src.relation_extractor import RuleBasedRelationExtractor


def main() -> None:
    """从示例文本抽取关系，并把关系批量写入 Neo4j。"""
    config = AppConfig.from_env()
    parser = argparse.ArgumentParser(description="Initialize Neo4j demo graph from company.txt")
    parser.add_argument("--data", default=config.default_data_path, help="Path to company.txt")
    parser.add_argument("--uri", default=config.neo4j_uri)
    parser.add_argument("--username", default=config.neo4j_username)
    parser.add_argument("--password", default=config.neo4j_password)
    args = parser.parse_args()

    data_path = Path(args.data)
    raw_text = data_path.read_text(encoding="utf-8")
    # 在这里加载文档，顺便校验示例文本是否符合运行时格式。
    documents = load_documents(data_path)
    relations = RuleBasedRelationExtractor().extract(raw_text)

    graph = Neo4jGraphStore(args.uri, args.username, args.password)
    try:
        graph.clear()
        graph.upsert_relations(relations)
    finally:
        graph.close()

    print(f"Loaded {len(documents)} documents and {len(relations)} relations into Neo4j.")


if __name__ == "__main__":
    main()
