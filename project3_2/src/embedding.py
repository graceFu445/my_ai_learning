import hashlib
import math
from typing import Iterable, List, Optional, Protocol

import dashscope
from dashscope import TextEmbedding

from .config import AppConfig
from .models import Document, RetrievalHit
from .tokenization import tokenize


class EmbeddingClient(Protocol):
    """向量模型协议，便于生产 Qwen 和测试替身共用同一检索器。"""

    def embed(self, text: str) -> List[float]:
        """把文本编码成向量。"""
        ...


class SimpleEmbeddingClient:
    """本地测试用哈希向量模型，不依赖网络，不用于真实运行链路。"""

    def __init__(self, dimensions: int = 128):
        """设置哈希向量维度。"""
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        """用分词哈希构造稀疏向量，保证单元测试可离线运行。"""
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0
        return _normalize(vector)


class QwenEmbeddingClient:
    """通义千问向量客户端，负责调用 text-embedding-v3 生成语义向量。"""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """读取模型名和 API Key，允许测试时显式覆盖。"""
        config = AppConfig.from_env()
        self.model = model or config.qwen_embedding_model
        self.api_key = api_key or config.dashscope_api_key

    def embed(self, text: str) -> List[float]:
        """调用 DashScope 向量接口，并返回第一条文本的 embedding。"""
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Qwen embeddings")

        dashscope.api_key = self.api_key
        response = TextEmbedding.call(model=self.model, input=[text])
        if getattr(response, "status_code", 200) != 200:
            raise RuntimeError(f"Qwen embedding request failed: {response}")
        return response.output["embeddings"][0]["embedding"]


class VectorRetriever:
    """向量检索器：预先编码文档，查询时按余弦相似度排序。"""

    def __init__(self, documents: Iterable[Document], embedding_client: EmbeddingClient):
        """保存文档并预计算文档向量，避免每次查询重复编码语料。"""
        self.documents = list(documents)
        self.embedding_client = embedding_client
        self.document_vectors = [embedding_client.embed(document.content) for document in self.documents]

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        """编码查询文本，返回相似度最高的文档命中列表。"""
        query_vector = self.embedding_client.embed(query)
        scores = [_cosine(query_vector, doc_vector) for doc_vector in self.document_vectors]
        hits = [
            RetrievalHit(document=document, score=score, source="vector")
            for document, score in zip(self.documents, scores)
            if score > 0
        ]
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]


def _cosine(left: List[float], right: List[float]) -> float:
    """计算两个向量的余弦相似度。"""
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize(vector: List[float]) -> List[float]:
    """把向量归一化为单位长度，避免文本长度直接影响相似度。"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]
