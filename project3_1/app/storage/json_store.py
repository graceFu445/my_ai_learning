"""
JSON文件存储模块
MVP阶段使用 JSON 保存 FAQ 原文和会话历史，便于本地演示和人工检查。
"""
from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from uuid import uuid4


class JsonStore:
    """负责读写 FAQ 和会话历史的轻量持久化组件"""

    def __init__(self, data_dir: Path | str):
        self._lock = RLock()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.faqs_path = self.data_dir / "faqs.json"
        self.sessions_path = self.data_dir / "sessions.json"
        self._ensure_file(self.faqs_path, [])
        self._ensure_file(self.sessions_path, {})

    def list_faqs(self) -> list[dict]:
        return self._read_json(self.faqs_path)

    def get_faq(self, faq_id: str) -> dict | None:
        return next((faq for faq in self.list_faqs() if faq["id"] == faq_id), None)

    def create_faq(self, question: str, answer: str, tags: list[str] | None = None) -> dict:
        """创建FAQ条目并写入 data/faqs.json"""
        faq = self.build_faq(question, answer, tags)
        faqs = self.list_faqs()
        faqs.append(faq)
        self.replace_faqs(faqs)
        return faq

    def build_faq(self, question: str, answer: str, tags: list[str] | None = None) -> dict:
        """生成一条FAQ记录但不写入文件，便于上层先重建索引再提交数据"""
        now = self._now()
        return {
            "id": f"faq_{uuid4().hex}",
            "question": question.strip(),
            "answer": answer.strip(),
            "tags": tags or [],
            "created_at": now,
            "updated_at": now,
        }

    def replace_faqs(self, faqs: list[dict]):
        """整体替换FAQ列表；配合索引重建实现数据和索引尽量一致提交"""
        self._write_json(self.faqs_path, deepcopy(faqs))

    def update_faq(self, faq_id: str, **changes) -> dict | None:
        """按ID更新FAQ；不存在时返回 None 交给上层转成404"""
        faqs = self.list_faqs()
        for faq in faqs:
            if faq["id"] == faq_id:
                for key in ("question", "answer", "tags"):
                    if key in changes and changes[key] is not None:
                        value = changes[key]
                        faq[key] = value.strip() if isinstance(value, str) else value
                faq["updated_at"] = self._now()
                self._write_json(self.faqs_path, faqs)
                return faq
        return None

    def delete_faq(self, faq_id: str) -> bool:
        faqs = self.list_faqs()
        kept = [faq for faq in faqs if faq["id"] != faq_id]
        if len(kept) == len(faqs):
            return False
        self._write_json(self.faqs_path, kept)
        return True

    def get_session(self, session_id: str) -> dict:
        sessions = self._read_json(self.sessions_path)
        return sessions.get(session_id, {"id": session_id, "messages": []})

    def append_message(self, session_id: str, role: str, content: str) -> dict:
        """向指定会话追加一条消息，用于支持多轮上下文"""
        sessions = self._read_json(self.sessions_path)
        session = sessions.setdefault(session_id, {"id": session_id, "messages": []})
        session["messages"].append(
            {
                "role": role,
                "content": content,
                "created_at": self._now(),
            }
        )
        self._write_json(self.sessions_path, sessions)
        return session

    def _ensure_file(self, path: Path, default):
        if not path.exists():
            self._write_json(path, default)

    def _read_json(self, path: Path):
        with self._lock:
            return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data):
        # ensure_ascii=False 保留中文原文，方便直接打开 JSON 检查数据。
        with self._lock:
            tmp_path = path.with_name(f".{path.name}.tmp")
            tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp_path.replace(path)

    def _now(self) -> str:
        return datetime.now(UTC).isoformat()

    def now(self) -> str:
        """对服务层暴露统一时间格式，避免业务层自己拼时间"""
        return self._now()
