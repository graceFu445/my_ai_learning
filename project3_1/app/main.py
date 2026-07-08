"""
FAQ RAG MVP Web API
基于 FastAPI 暴露 FAQ 管理、CSV导入、聊天问答和索引重建接口。
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response

from app.config import get_settings
from app.schemas import ChatRequest, ChatResponse, FaqCreate, FaqUpdate, ImportResponse
from app.rag.answer import DashScopeAnswerGenerator, MissingDashScopeAnswerGenerator
from app.rag.index import DashScopeFaissIndex, EmptyIndex
from app.services.chat_service import ChatService
from app.services.faq_service import FaqService
from app.storage.json_store import JsonStore


def create_app(data_dir: Path | None = None, index=None, answer_generator=None) -> FastAPI:
    """创建FastAPI应用；测试时可注入临时数据目录和假索引"""
    settings = get_settings()
    store = JsonStore(data_dir or settings.data_dir)

    # 依赖在应用启动时组装，路由层只负责HTTP协议转换。
    index = index or _build_index(settings, Path(data_dir or settings.data_dir))
    answer_generator = answer_generator or _build_answer_generator(settings)
    faq_service = FaqService(store, index)
    chat_service = ChatService(
        store,
        index,
        answer_generator,
        top_k=settings.top_k,
        min_similarity_score=settings.min_similarity_score,
    )
    app = FastAPI(title=settings.app_name)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/faqs", status_code=201)
    def create_faq(payload: FaqCreate):
        return faq_service.create_faq(payload.question, payload.answer, payload.tags)

    @app.get("/faqs")
    def list_faqs():
        return faq_service.list_faqs()

    @app.get("/faqs/{faq_id}")
    def get_faq(faq_id: str):
        faq = faq_service.get_faq(faq_id)
        if faq is None:
            raise HTTPException(status_code=404, detail="FAQ not found.")
        return faq

    @app.put("/faqs/{faq_id}")
    def update_faq(faq_id: str, payload: FaqUpdate):
        faq = faq_service.update_faq(faq_id, **payload.model_dump(exclude_unset=True))
        if faq is None:
            raise HTTPException(status_code=404, detail="FAQ not found.")
        return faq

    @app.delete("/faqs/{faq_id}", status_code=204)
    def delete_faq(faq_id: str):
        if not faq_service.delete_faq(faq_id):
            raise HTTPException(status_code=404, detail="FAQ not found.")
        return Response(status_code=204)

    @app.post("/faqs/import", response_model=ImportResponse, status_code=201)
    async def import_faqs(file: UploadFile = File(...)):
        """上传CSV并批量导入FAQ"""
        content = (await file.read()).decode("utf-8-sig")
        faqs = faq_service.import_csv(content)
        return {"imported_count": len(faqs), "faqs": faqs}

    @app.post("/chat", response_model=ChatResponse)
    def chat(payload: ChatRequest):
        """用户问答入口；session_id 为空时服务层会自动创建会话"""
        try:
            return chat_service.chat(payload.question, payload.session_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str):
        return store.get_session(session_id)

    @app.post("/index/rebuild")
    def rebuild_index():
        faq_service.rebuild_index()
        return {"status": "rebuilt", "faq_count": len(store.list_faqs())}

    return app


def _build_index(settings, data_dir: Path):
    """根据配置创建真实 FAISS 索引或空索引"""
    if not settings.dashscope_api_key or not settings.dashscope_embedding_model:
        return EmptyIndex()
    return DashScopeFaissIndex(
        data_dir / "faiss_index",
        api_key=settings.dashscope_api_key,
        embedding_model=settings.dashscope_embedding_model,
        embedding_dimension=settings.embedding_dimension,
    )


def _build_answer_generator(settings):
    """根据配置创建真实回答生成器或缺省占位生成器"""
    if not settings.dashscope_api_key or not settings.dashscope_chat_model:
        return MissingDashScopeAnswerGenerator()
    return DashScopeAnswerGenerator(
        api_key=settings.dashscope_api_key,
        model=settings.dashscope_chat_model,
    )


app = create_app()
