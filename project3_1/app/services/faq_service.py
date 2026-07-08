"""
FAQ业务服务
封装 FAQ 增删改查、CSV导入和索引重建触发逻辑。
"""
from __future__ import annotations

import csv
from copy import deepcopy
from io import StringIO

from app.storage.json_store import JsonStore


class FaqService:
    """连接 JSON 存储和向量索引的 FAQ 应用服务"""

    def __init__(self, store: JsonStore, index):
        self.store = store
        self.index = index

    def list_faqs(self) -> list[dict]:
        return self.store.list_faqs()

    def get_faq(self, faq_id: str) -> dict | None:
        return self.store.get_faq(faq_id)

    def create_faq(self, question: str, answer: str, tags: list[str] | None = None) -> dict:
        """创建单条FAQ；索引重建成功后才提交到JSON文件"""
        faqs = self.store.list_faqs()
        faq = self.store.build_faq(question, answer, self._clean_tags(tags or []))
        next_faqs = [*faqs, faq]
        self.index.rebuild(next_faqs)
        self.store.replace_faqs(next_faqs)
        return faq

    def update_faq(self, faq_id: str, **changes) -> dict | None:
        if "tags" in changes and changes["tags"] is not None:
            changes["tags"] = self._clean_tags(changes["tags"])
        faqs = self.store.list_faqs()
        next_faqs = deepcopy(faqs)
        for faq in next_faqs:
            if faq["id"] == faq_id:
                for key in ("question", "answer", "tags"):
                    if key in changes and changes[key] is not None:
                        value = changes[key]
                        faq[key] = value.strip() if isinstance(value, str) else value
                faq["updated_at"] = self.store.now()
                self.index.rebuild(next_faqs)
                self.store.replace_faqs(next_faqs)
                return faq
        return None

    def delete_faq(self, faq_id: str) -> bool:
        faqs = self.store.list_faqs()
        next_faqs = [faq for faq in faqs if faq["id"] != faq_id]
        if len(next_faqs) == len(faqs):
            return False
        self.index.rebuild(next_faqs)
        self.store.replace_faqs(next_faqs)
        return True

    def import_csv(self, csv_text: str) -> list[dict]:
        """从CSV文本批量导入FAQ，要求包含 question、answer、tags 三列"""
        reader = csv.DictReader(StringIO(csv_text))
        faqs = self.store.list_faqs()
        imported = []
        for row in reader:
            question = (row.get("question") or "").strip()
            answer = (row.get("answer") or "").strip()
            if not question or not answer:
                continue
            # 兼容 tags 中用英文逗号导致的额外列，避免简单CSV样例导入失败。
            raw_tags = [row.get("tags") or ""]
            raw_tags.extend(row.get(None) or [])
            imported.append(
                self.store.build_faq(
                    question=question,
                    answer=answer,
                    tags=self._parse_tags(raw_tags),
                )
            )
        if not imported:
            return []
        next_faqs = [*faqs, *imported]
        self.index.rebuild(next_faqs)
        self.store.replace_faqs(next_faqs)
        return imported

    def rebuild_index(self):
        self.index.rebuild(self.store.list_faqs())

    def _parse_tags(self, values: list[str]) -> list[str]:
        """解析标签字段，兼容中英文逗号和分号"""
        tags = []
        for value in values:
            for token in value.replace("；", ";").replace("，", ",").replace(";", ",").split(","):
                clean = token.strip()
                if clean:
                    tags.append(clean)
        return self._clean_tags(tags)

    def _clean_tags(self, tags: list[str]) -> list[str]:
        return list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
