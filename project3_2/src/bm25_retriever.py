import math
from collections import Counter
from typing import Iterable, List

from .models import Document, RetrievalHit
from .tokenization import tokenize


class BM25Retriever:
    """基于 BM25 的关键词检索器，用来补足向量检索对精确词匹配的不足。"""

    def __init__(self, documents: Iterable[Document], k1: float = 1.5, b: float = 0.75):
        """预先分词并统计词频、文档频率，便于查询时快速打分。"""
        self.documents = list(documents)
        self.k1 = k1
        self.b = b
        self.tokenized_docs = [tokenize(document.content) for document in self.documents]
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        self.term_frequencies = [Counter(tokens) for tokens in self.tokenized_docs]
        self.document_frequencies = Counter()
        for tokens in self.tokenized_docs:
            self.document_frequencies.update(set(tokens))

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """对问题进行 BM25 检索，并把分数归一化到 0 到 1。"""
        query_tokens = tokenize(query)
        raw_scores = [self._score(query_tokens, index) for index in range(len(self.documents))]
        max_score = max(raw_scores, default=0.0)
        hits: List[RetrievalHit] = []
        for document, score in zip(self.documents, raw_scores):
            normalized = score / max_score if max_score > 0 else 0.0
            if normalized > 0:
                hits.append(RetrievalHit(document=document, score=normalized, source="bm25"))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _score(self, query_tokens: List[str], doc_index: int) -> float:
        """计算单篇文档对查询词集合的 BM25 原始分数。"""
        score = 0.0
        doc_length = self.doc_lengths[doc_index] if self.doc_lengths else 0
        term_frequency = self.term_frequencies[doc_index]
        for term in query_tokens:
            frequency = term_frequency.get(term, 0)
            if frequency == 0:
                continue
            idf = self._idf(term)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * doc_length / (self.avg_doc_length or 1.0)
            )
            score += idf * (frequency * (self.k1 + 1)) / denominator
        return score

    def _idf(self, term: str) -> float:
        """计算词项的逆文档频率，低频词会获得更高权重。"""
        total_docs = len(self.documents)
        doc_frequency = self.document_frequencies.get(term, 0)
        return math.log(1 + (total_docs - doc_frequency + 0.5) / (doc_frequency + 0.5))
