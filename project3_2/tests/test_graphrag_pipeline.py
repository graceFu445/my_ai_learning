import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.bm25_retriever import BM25Retriever
from src.document_loader import load_documents
from src.embedding import SimpleEmbeddingClient, VectorRetriever
from src.graph_store import InMemoryGraphStore
from src.models import GraphEvidence, RetrievalHit
from src.qa import GraphRAGSystem, build_local_system
from src.relation_extractor import RuleBasedRelationExtractor
from src.scorer import HybridScorer


class CapturingTextGenerator:
    """测试用文本生成器：记录收到的提示词，并返回预设答案。"""

    def __init__(self, answer: str):
        self.answer = answer
        self.prompts = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


SAMPLE_TEXT = """公司基本信息：
A集团是一家大型综合控股集团，主营业务包括智能制造、能源投资和企业服务。A集团由张三担任董事长，总部位于上海。A集团的主要股东包括B资本、C基金和H投资。

股权结构详情：
B资本持有A集团60%的股份，是A集团的控股股东，也是A集团的最大股东。B资本由李四创立，主要关注长期股权投资。

C基金持有A集团25%的股份，是A集团的重要机构投资者。C基金的管理人为王五。

H投资持有A集团10%的股份，是A集团的财务投资者。H投资不参与A集团的日常经营管理。

集团控股架构：
A集团作为集团母公司，直接控股D科技和E实业。其中，A集团持有D科技80%的股份，持有E实业70%的股份。

D科技是A集团旗下的重要科技子公司，主要从事人工智能设备研发。D科技进一步控股F智能和G能源。其中，D科技持有F智能65%的股份，持有G能源55%的股份。

E实业是A集团旗下的制造业子公司，主要负责传统制造业务。E实业参股K物流，持有K物流30%的股份，但不构成控股关系。

管理层信息：
A集团董事长为张三，总经理为赵六。D科技总经理为孙七。B资本创始人为李四，C基金管理人为王五。

完整的股权和控股关系链条如下：
B资本 控股 A集团（持股60%）
C基金 投资 A集团（持股25%）
H投资 投资 A集团（持股10%）
A集团 控股 D科技（持股80%）
A集团 控股 E实业（持股70%）
D科技 控股 F智能（持股65%）
D科技 控股 G能源（持股55%）
E实业 参股 K物流（持股30%）
"""


class GraphRAGPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.company_path = Path(self.tmpdir.name) / "company.txt"
        self.company_path.write_text(SAMPLE_TEXT, encoding="utf-8")
        self.documents = load_documents(self.company_path)
        self.relations = RuleBasedRelationExtractor().extract(SAMPLE_TEXT)
        self.graph = InMemoryGraphStore()
        self.graph.clear()
        self.graph.upsert_relations(self.relations)

    def tearDown(self):
        self.tmpdir.cleanup()

    def build_system(self, text_generator=None):
        generator = text_generator or CapturingTextGenerator("占位答案")
        return GraphRAGSystem(
            documents=self.documents,
            vector_retriever=VectorRetriever(self.documents, SimpleEmbeddingClient()),
            bm25_retriever=BM25Retriever(self.documents),
            graph_store=self.graph,
            scorer=HybridScorer(),
            text_generator=generator,
        )

    def build_system_without_graph_relations(self, text_generator=None):
        generator = text_generator or CapturingTextGenerator("占位答案")
        empty_graph = InMemoryGraphStore()
        return GraphRAGSystem(
            documents=self.documents,
            vector_retriever=VectorRetriever(self.documents, SimpleEmbeddingClient()),
            bm25_retriever=BM25Retriever(self.documents),
            graph_store=empty_graph,
            scorer=HybridScorer(),
            text_generator=generator,
        )

    def build_system_from_text(self, text, text_generator=None):
        generator = text_generator or CapturingTextGenerator("占位答案")
        path = Path(self.tmpdir.name) / "generic_company.txt"
        path.write_text(text, encoding="utf-8")
        documents = load_documents(path)
        relations = RuleBasedRelationExtractor().extract(text)
        graph = InMemoryGraphStore()
        graph.upsert_relations(relations)
        return GraphRAGSystem(
            documents=documents,
            vector_retriever=VectorRetriever(documents, SimpleEmbeddingClient()),
            bm25_retriever=BM25Retriever(documents),
            graph_store=graph,
            scorer=HybridScorer(),
            text_generator=generator,
            document_relations=relations,
        )

    def test_load_documents_splits_non_empty_paragraphs(self):
        self.assertGreaterEqual(len(self.documents), 7)
        self.assertEqual(self.documents[0].id, "doc_0")
        self.assertIn("A集团", self.documents[0].content)
        self.assertEqual(self.documents[0].metadata["source"], str(self.company_path))

    def test_rule_extractor_preserves_relation_type_and_share(self):
        relation = next(r for r in self.relations if r.source == "E实业" and r.target == "K物流")
        self.assertEqual(relation.relation_type, "参股")
        self.assertEqual(relation.share_percent, 30.0)

        holding = next(r for r in self.relations if r.source == "C基金" and r.target == "A集团")
        self.assertEqual(holding.share_percent, 25.0)

    def test_bm25_retriever_prioritizes_exact_share_question(self):
        hits = BM25Retriever(self.documents).search("C基金持有A集团多少股份？", top_k=3)
        self.assertGreater(hits[0].score, 0)
        self.assertIn("C基金持有A集团25%", hits[0].document.content)

    def test_graph_store_finds_direct_and_multi_hop_paths(self):
        direct = self.graph.direct_relations("A集团", relation_types={"控股"})
        self.assertEqual({r.target for r in direct}, {"D科技", "E实业"})

        paths = self.graph.find_paths("B资本", "F智能", max_hops=4, relation_types={"控股"})
        self.assertEqual(paths[0].nodes, ["B资本", "A集团", "D科技", "F智能"])

    def test_scorer_combines_vector_bm25_and_graph_scores(self):
        doc = self.documents[1]
        vector_hit = RetrievalHit(document=doc, score=0.8, source="vector")
        bm25_hit = RetrievalHit(document=doc, score=0.6, source="bm25")
        graph_evidence = GraphEvidence(confidence=1.0, paths=["B资本 -> 控股 -> A集团"])

        results = HybridScorer().score([vector_hit], [bm25_hit], graph_evidence)

        self.assertAlmostEqual(results[0].hybrid_score, 0.78)
        self.assertAlmostEqual(results[0].vector_score, 0.8)
        self.assertAlmostEqual(results[0].bm25_score, 0.6)
        self.assertAlmostEqual(results[0].graph_score, 1.0)

    def test_qa_builds_prompt_from_retrieved_evidence_for_llm_answer(self):
        generator = CapturingTextGenerator("D科技的总经理是孙七。")
        system = self.build_system(text_generator=generator)

        result = system.answer("D科技总经理是谁？")

        self.assertEqual(result.answer, "D科技的总经理是孙七。")
        self.assertFalse(result.warnings)
        self.assertEqual(len(generator.prompts), 1)
        self.assertIn("D科技总经理是谁？", generator.prompts[0])
        self.assertIn("D科技总经理为孙七", generator.prompts[0])
        self.assertIn("只能基于证据回答", generator.prompts[0])

    def test_qa_puts_graph_and_document_evidence_into_llm_prompt(self):
        generator = CapturingTextGenerator("E实业相关的关系包括E实业参股K物流，A集团控股E实业。")
        system = self.build_system(text_generator=generator)

        result = system.answer("E实业和谁有关系？")

        self.assertIn("K物流", result.answer)
        self.assertTrue(result.graph_evidence.relations)
        self.assertIn("E实业 -> 参股 -> K物流", generator.prompts[0])
        self.assertIn("A集团 -> 控股 -> E实业", generator.prompts[0])
        self.assertIn("E实业参股K物流", generator.prompts[0])

    def test_qa_requires_text_generator(self):
        system = GraphRAGSystem(
            documents=self.documents,
            vector_retriever=VectorRetriever(self.documents, SimpleEmbeddingClient()),
            bm25_retriever=BM25Retriever(self.documents),
            graph_store=self.graph,
            scorer=HybridScorer(),
            text_generator=None,
        )

        with self.assertRaisesRegex(RuntimeError, "必须配置通义千问文本生成器"):
            system.answer("D科技总经理是谁？")

    def test_build_local_system_uses_qwen_by_default(self):
        with patch("src.qa.QwenEmbeddingClient", return_value=SimpleEmbeddingClient()) as embedding_cls:
            with patch("src.qa.QwenTextGenerator", return_value=CapturingTextGenerator("默认 Qwen")) as generator_cls:
                system = build_local_system(self.company_path)

        embedding_cls.assert_called_once()
        generator_cls.assert_called_once()
        self.assertEqual(system.text_generator.answer, "默认 Qwen")

    def test_qa_keeps_retrieval_and_graph_evidence_for_expected_cases(self):
        generator = CapturingTextGenerator("占位答案")
        system = self.build_system(text_generator=generator)

        largest = system.answer("A集团的最大股东是谁？")
        self.assertTrue(largest.graph_evidence.relations)
        self.assertIn("B资本 -> 控股 -> A集团", generator.prompts[-1])

        path = system.answer("从B资本到F智能有哪些控股路径？")
        self.assertIn("B资本 -> 控股 -> A集团 -> 控股 -> D科技 -> 控股 -> F智能", path.graph_evidence.paths)

        controlled = system.answer("D科技控股了哪些公司？")
        self.assertIn("F智能", generator.prompts[-1])
        self.assertIn("G能源", generator.prompts[-1])
        self.assertFalse(controlled.warnings)

        equity = system.answer("E实业参股了哪些公司？")
        self.assertIn("E实业 -> 参股 -> K物流", generator.prompts[-1])

        non_control = system.answer("E实业是否控股K物流？")
        self.assertTrue(non_control.graph_evidence.relations)

    def test_qa_uses_retrieved_documents_when_graph_has_no_match(self):
        generator = CapturingTextGenerator("D科技控股F智能和G能源。")
        system = self.build_system_without_graph_relations(text_generator=generator)

        result = system.answer("D科技控股了哪些公司？")

        self.assertIn("F智能", result.answer)
        self.assertIn("G能源", result.answer)
        self.assertGreater(result.scored_documents[0].bm25_score, 0)
        self.assertIn("D科技持有F智能65%", generator.prompts[0])
        self.assertFalse(result.warnings)

    def test_qa_is_not_hardcoded_to_sample_company_names(self):
        text = """公司资料：
甲控股董事长为周一。甲控股的主要业务是产业投资。

股权关系：
甲控股 控股 乙制造（持股80%）
乙制造 参股 丙物流（持股30%）
丁资本 投资 甲控股（持股55%）
"""
        generator = CapturingTextGenerator("甲控股相关答案。")
        system = self.build_system_from_text(text, text_generator=generator)

        controlled = system.answer("甲控股控股了哪些公司？")
        self.assertIn("乙制造", generator.prompts[-1])
        self.assertIn("80%", generator.prompts[-1])

        equity = system.answer("乙制造参股了哪些公司？")
        self.assertIn("丙物流", generator.prompts[-1])
        self.assertIn("30%", generator.prompts[-1])

        judgement = system.answer("乙制造是否控股丙物流？")
        self.assertIn("乙制造 -> 参股 -> 丙物流", generator.prompts[-1])

        holder = system.answer("丁资本持有甲控股多少股份？")
        self.assertIn("55%", generator.prompts[-1])
        self.assertTrue(holder.graph_evidence.relations)

        chairman = system.answer("甲控股的董事长是谁？")
        self.assertIn("周一", generator.prompts[-1])


if __name__ == "__main__":
    unittest.main()
