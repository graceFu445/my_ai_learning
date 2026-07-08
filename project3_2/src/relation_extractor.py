import json
import re
from typing import List, Optional, Protocol

import dashscope
from dashscope import Generation

from .config import AppConfig
from .models import Relation


class RelationExtractor(Protocol):
    """关系抽取器协议，便于规则抽取和 Qwen 抽取互换。"""

    def extract(self, text: str) -> List[Relation]:
        """从原始文本中抽取企业关系。"""
        ...


class RuleBasedRelationExtractor:
    """从示例语料中抽取股权关系，不绑定到固定样例公司名。"""

    RELATION_PATTERN = re.compile(
        r"(?P<source>[\u4e00-\u9fffA-Za-z0-9]+)\s+"
        r"(?P<relation>控股|投资|参股)\s+"
        r"(?P<target>[\u4e00-\u9fffA-Za-z0-9]+)"
        r"(?:（持股(?P<share>[0-9]+(?:\.[0-9]+)?)%）)?"
    )
    HOLDING_PATTERN = re.compile(
        r"(?P<source>[\u4e00-\u9fffA-Za-z0-9]+)持有"
        r"(?P<target>[\u4e00-\u9fffA-Za-z0-9]+?)"
        r"(?P<share>[0-9]+(?:\.[0-9]+)?)%的股份"
    )
    SAME_SENTENCE_SHARE_PATTERN = re.compile(
        r"(?P<source>[\u4e00-\u9fffA-Za-z0-9]+)"
        r"(?P<relation>控股|投资|参股)"
        r"(?P<target>[\u4e00-\u9fffA-Za-z0-9]+)"
        r"[^。；\n]*?持有(?P=target)(?P<share>[0-9]+(?:\.[0-9]+)?)%"
    )

    def extract(self, text: str) -> List[Relation]:
        """按规则抽取显式关系、持股句式和同句持股关系。"""
        relations: List[Relation] = []
        seen = set()
        for match in self.SAME_SENTENCE_SHARE_PATTERN.finditer(text):
            self._append_unique(
                relations,
                seen,
                Relation(
                    source=match.group("source"),
                    target=match.group("target"),
                    relation_type=match.group("relation"),
                    share_percent=float(match.group("share")),
                    source_text=match.group(0),
                ),
            )
        for match in self.RELATION_PATTERN.finditer(text):
            relation = self._from_match(match)
            self._append_unique(relations, seen, relation)
        for match in self.HOLDING_PATTERN.finditer(text):
            source = match.group("source")
            target = match.group("target")
            share = float(match.group("share"))
            relation_type = "控股" if share >= 50 else "投资"
            self._append_unique(
                relations,
                seen,
                Relation(
                    source=source,
                    target=target,
                    relation_type=relation_type,
                    share_percent=share,
                    source_text=match.group(0),
                ),
            )
        return relations

    def _from_match(self, match: re.Match) -> Relation:
        """把正则匹配结果转换为 Relation 对象。"""
        share = match.group("share")
        return Relation(
            source=match.group("source"),
            target=match.group("target"),
            relation_type=match.group("relation"),
            share_percent=float(share) if share is not None else None,
            source_text=match.group(0),
        )

    def _append_unique(self, relations: List[Relation], seen: set, relation: Relation) -> None:
        """按主体、客体和关系类型去重追加，避免重复证据进入图谱。"""
        key = (relation.source, relation.target, relation.relation_type)
        if key not in seen:
            relations.append(relation)
            seen.add(key)


class QwenRelationExtractor:
    """通义千问关系抽取器，用于更复杂语料的结构化关系抽取。"""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        """读取抽取所需模型名和 API Key。"""
        config = AppConfig.from_env()
        self.model = model or config.qwen_llm_model
        self.api_key = api_key or config.dashscope_api_key

    def extract(self, text: str) -> List[Relation]:
        """提示 Qwen 输出 JSON，再转换为 Relation 列表。"""
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for Qwen relation extraction")

        dashscope.api_key = self.api_key
        prompt = (
            "从文本中抽取企业股权/控股关系，只返回JSON。"
            "格式：{\"relationships\":[{\"source\":\"主体公司\",\"target\":\"目标公司\","
            "\"relation_type\":\"控股\",\"share_percent\":60.0}]}。\n\n"
            f"文本：{text}"
        )
        response = Generation.call(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            result_format="message",
            temperature=0,
        )
        content = response.output["choices"][0]["message"]["content"]
        payload = json.loads(content)
        return [
            Relation(
                source=item["source"],
                target=item["target"],
                relation_type=item["relation_type"],
                share_percent=item.get("share_percent"),
                confidence=float(item.get("confidence", 1.0)),
            )
            for item in payload.get("relationships", [])
        ]
