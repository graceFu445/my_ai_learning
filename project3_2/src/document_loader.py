from pathlib import Path
from typing import List, Union

from .models import Document


def load_documents(path: Union[str, Path], min_length: int = 10) -> List[Document]:
    """按空行切分输入文本，生成带来源和段落编号的文档对象。"""
    source = Path(path)
    raw_text = source.read_text(encoding="utf-8")
    paragraphs = [paragraph.strip() for paragraph in raw_text.split("\n\n") if paragraph.strip()]

    documents: List[Document] = []
    for index, paragraph in enumerate(paragraphs):
        # 保留段落内部换行结构，同时去掉空行和多余空格，便于检索展示。
        compact = "\n".join(line.strip() for line in paragraph.splitlines() if line.strip())
        if len(compact) >= min_length:
            documents.append(
                Document(
                    id=f"doc_{len(documents)}",
                    content=compact,
                    metadata={"source": str(source), "paragraph_id": str(index)},
                )
            )
    return documents
