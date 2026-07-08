"""
聊天业务服务
负责会话ID生成、历史保存、FAQ检索、相似度过滤和回答生成。
"""
from __future__ import annotations

from uuid import uuid4

from app.storage.json_store import JsonStore


class ChatService:
    """面向 API/CLI/Web UI 复用的多轮问答服务"""

    def __init__(
        self,
        store: JsonStore,
        index,
        answer_generator,
        top_k: int,
        min_similarity_score: float,
    ):
        self.store = store
        self.index = index
        self.answer_generator = answer_generator
        self.top_k = top_k
        self.min_similarity_score = min_similarity_score

    def chat(self, question: str, session_id: str | None = None) -> dict:
        """处理一次用户提问；没有 session_id 时自动创建新会话"""
        session_id = session_id or f"session_{uuid4().hex}"
        self.store.append_message(session_id, "user", question)

        # 只取最近几轮对话，避免上下文无限膨胀，同时保留追问所需信息。
        history = self.store.get_session(session_id)["messages"][-8:]
        matches = [
            match
            for match in self.index.search(question, self.top_k)
            if match.get("score", 0.0) >= self.min_similarity_score
        ]
        if not matches:
            answer = "知识库中没有足够相关的答案。"
            matched_faqs = []
        else:
            # 回答生成器只能基于检索命中的FAQ回答，降低知识库外编造风险。
            answer = self.answer_generator.generate(question, matches, history)
            matched_faqs = [
                {
                    **match["faq"],
                    "score": match.get("score", 0.0),
                }
                for match in matches
            ]
        self.store.append_message(session_id, "assistant", answer)
        return {
            "answer": answer,
            "session_id": session_id,
            "matched_faqs": matched_faqs,
        }
