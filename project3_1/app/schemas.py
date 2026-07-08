"""
API请求和响应 Schema
所有接口入参、出参在这里统一声明，便于 Swagger 展示和参数校验。
"""
from pydantic import BaseModel, Field


class FaqCreate(BaseModel):
    """创建FAQ请求模型"""

    question: str = Field(..., min_length=1, description="FAQ问题")
    answer: str = Field(..., min_length=1, description="FAQ答案")
    tags: list[str] = Field(default_factory=list, description="FAQ标签列表")


class FaqUpdate(BaseModel):
    """更新FAQ请求模型，所有字段都可选"""

    question: str | None = Field(default=None, min_length=1, description="FAQ问题")
    answer: str | None = Field(default=None, min_length=1, description="FAQ答案")
    tags: list[str] | None = Field(default=None, description="FAQ标签列表")


class ChatRequest(BaseModel):
    """聊天请求模型"""

    question: str = Field(..., min_length=1, description="用户问题")
    session_id: str | None = Field(default=None, description="会话ID；为空时后端自动创建")


class ChatResponse(BaseModel):
    """聊天响应模型"""

    answer: str
    session_id: str
    matched_faqs: list[dict]


class ImportResponse(BaseModel):
    """CSV导入响应模型"""

    imported_count: int
    faqs: list[dict]
