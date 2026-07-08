# FAQ RAG MVP Backend

FastAPI backend for an intelligent FAQ retrieval system. It stores FAQ and session data in JSON files, uses LlamaIndex to orchestrate the RAG retrieval flow, persists a FAISS vector index to disk, and uses DashScope for embeddings and controlled answer generation.

Project layout is documented in [项目结构说明文档.md](项目结构说明文档.md).

## Setup

This project must run in its own Conda environment. Do not install dependencies into `base`.

```bash
conda env create -f environment.yml
conda activate faq-rag
cp .env.example .env
uvicorn app.main:app --reload
```

If the environment already exists, refresh the editable install with:

```bash
conda run -n faq-rag pip install -e .
```

For non-interactive runs:

```bash
conda run -n faq-rag pytest tests -q
conda run -n faq-rag uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Environment

```bash
DASHSCOPE_API_KEY=your_api_key
DASHSCOPE_CHAT_MODEL=qwen-plus
DASHSCOPE_EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_DIMENSION=1024
TOP_K=3
MIN_SIMILARITY_SCORE=0.0
```

## CSV Import Format

```csv
question,answer,tags
如何退货？,签收后7天内可申请退货。,售后;退货
配送多久？,普通地区1-3天送达。,配送;时效
```

## API

- `POST /faqs`
- `GET /faqs`
- `GET /faqs/{id}`
- `PUT /faqs/{id}`
- `DELETE /faqs/{id}`
- `POST /faqs/import`
- `POST /chat`
- `GET /sessions/{session_id}`
- `POST /index/rebuild`
- `GET /health`
