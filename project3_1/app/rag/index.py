"""
LlamaIndex + DashScope + FAISS 检索索引模块
LlamaIndex 负责 RAG 文档、索引和 retriever 编排；DashScope 只负责生成向量。
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import dashscope
import faiss
import numpy as np
from dashscope import TextEmbedding
from llama_index.core import Document, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.embeddings import BaseEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import PrivateAttr


class EmptyIndex:
    """缺少 DashScope 配置时使用的空索引，让 CRUD 接口仍可本地演示"""

    def rebuild(self, faqs: list[dict]):
        return None

    def search(self, question: str, top_k: int) -> list[dict]:
        return []


class DashScopeEmbedding(BaseEmbedding):
    """把 DashScope text-embedding-v4 适配成 LlamaIndex 可调用的 embedding 模型"""

    _api_key: str = PrivateAttr()

    def __init__(self, api_key: str, model_name: str):
        # DashScope 批量请求在当前环境下 4 条一批最稳定。
        super().__init__(model_name=model_name, embed_batch_size=4)
        self._api_key = api_key

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_texts([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_texts([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in self._chunked(texts, 4):
            embeddings.extend(self._embed_texts(batch))
        return embeddings

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        dashscope.api_key = self._api_key
        response = TextEmbedding.call(model=self.model_name, input=texts)
        if getattr(response, "status_code", 200) != 200:
            code = getattr(response, "code", "Unknown")
            message = getattr(response, "message", "DashScope embedding request failed.")
            raise RuntimeError(f"DashScope embedding request failed: {code} - {message}")

        vectors = self._extract_embeddings(response)
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"DashScope embedding count mismatch: expected {len(texts)}, got {len(vectors)}."
            )

        matrix = np.asarray(vectors, dtype="float32")
        norm = np.linalg.norm(matrix, axis=1, keepdims=True)
        norm[norm == 0] = 1
        return (matrix / norm).tolist()

    def _extract_embeddings(self, response) -> list[list[float]]:
        output = getattr(response, "output", None)
        items = None
        if isinstance(output, dict):
            items = output.get("embeddings")
        elif output is not None and hasattr(output, "embeddings"):
            items = getattr(output, "embeddings")
        elif hasattr(response, "embeddings"):
            items = getattr(response, "embeddings")
        if not items:
            return []

        indexed_items = list(enumerate(items))
        ordered = sorted(
            indexed_items,
            key=lambda pair: pair[1].get("index", pair[1].get("text_index", pair[0]))
            if isinstance(pair[1], dict)
            else pair[0],
        )
        vectors = []
        for _, item in ordered:
            if isinstance(item, dict):
                vector = item.get("embedding") or item.get("vector")
            else:
                vector = getattr(item, "embedding", None) or getattr(item, "vector", None)
            if vector is not None:
                vectors.append(vector)
        return vectors

    def _chunked(self, values: list[str], size: int):
        for start in range(0, len(values), size):
            yield values[start : start + size]


class DashScopeFaissIndex:
    """封装 LlamaIndex、DashScope embeddings 与 FAISS 文件索引的检索组件"""

    def __init__(
        self,
        index_dir: Path | str,
        api_key: str | None,
        embedding_model: str | None,
        embedding_dimension: int | None = None,
    ):
        if not api_key or not embedding_model:
            raise ValueError("DASHSCOPE_API_KEY and DASHSCOPE_EMBEDDING_MODEL are required for retrieval.")
        self.index_dir = Path(index_dir)
        self.api_key = api_key
        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension
        self._index = None
        self._load_if_present()

    def rebuild(self, faqs: list[dict]):
        """全量重建 FAISS 索引；LlamaIndex 负责文档入库和索引持久化。"""
        tmp_index_dir = self.index_dir.with_name(f"{self.index_dir.name}_tmp")
        if tmp_index_dir.exists():
            shutil.rmtree(tmp_index_dir)
        tmp_index_dir.mkdir(parents=True, exist_ok=True)
        if not faqs:
            self._index = None
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            if tmp_index_dir.exists():
                shutil.rmtree(tmp_index_dir)
            return

        try:
            embed_model = self._embed_model()
            dimension = self.embedding_dimension or len(embed_model.get_text_embedding("dimension probe"))
            vector_store = FaissVectorStore(faiss_index=faiss.IndexFlatIP(dimension))
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # metadata 保留完整 FAQ JSON，检索命中后能映射回原始 FAQ 记录。
            documents = [self._faq_to_document(faq) for faq in faqs]
            new_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                embed_model=embed_model,
            )
            storage_context.persist(persist_dir=str(tmp_index_dir))
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            tmp_index_dir.rename(self.index_dir)
            self._index = new_index
        except Exception:
            if tmp_index_dir.exists():
                shutil.rmtree(tmp_index_dir)
            raise

    def search(self, question: str, top_k: int) -> list[dict]:
        """通过 LlamaIndex retriever 按语义相似度检索 FAQ。"""
        if self._index is None:
            self._load_if_present()
        if self._index is None:
            return []

        retriever = self._index.as_retriever(similarity_top_k=top_k)
        matches = []
        for result in retriever.retrieve(question):
            faq = json.loads(result.node.metadata["faq_json"])
            matches.append({"faq": faq, "score": float(result.score or 0.0)})
        return matches

    def _load_if_present(self):
        """服务重启后通过 LlamaIndex 加载已落盘的 FAISS 索引。"""
        if not self.index_dir.exists():
            return
        if not (self.index_dir / "docstore.json").exists():
            return
        if not (self.index_dir / "index_store.json").exists():
            return

        # LlamaIndex 的 FAISS 插件会用自己的默认文件名保存向量索引；
        # 通过 from_persist_dir 加载，避免业务代码绑定具体落盘文件名。
        vector_store = FaissVectorStore.from_persist_dir(str(self.index_dir))
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            persist_dir=str(self.index_dir),
        )
        self._index = load_index_from_storage(storage_context, embed_model=self._embed_model())

    def _embed_model(self):
        return DashScopeEmbedding(api_key=self.api_key, model_name=self.embedding_model)

    def _faq_to_document(self, faq: dict):
        """把结构化 FAQ 转换成 LlamaIndex Document。"""
        return Document(
            text=(
                f"问题：{faq['question']}\n"
                f"答案：{faq['answer']}\n"
                f"标签：{'、'.join(faq.get('tags', []))}"
            ),
            metadata={
                "faq_id": faq["id"],
                "faq_json": json.dumps(faq, ensure_ascii=False),
            },
            excluded_embed_metadata_keys=["faq_id", "faq_json"],
            excluded_llm_metadata_keys=["faq_id", "faq_json"],
        )
