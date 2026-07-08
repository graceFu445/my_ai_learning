import re
from typing import List

import jieba


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> List[str]:
    """统一分词入口，优先使用 jieba，并过滤过短的无效 token。"""
    tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]
    return [token for token in tokens if len(token) > 1 or token.isdigit()]


def _fallback_tokenize(text: str) -> List[str]:
    """无 jieba 时的备用分词逻辑，保留给极简环境或调试使用。"""
    coarse_tokens = _TOKEN_RE.findall(text)
    tokens: List[str] = []
    for token in coarse_tokens:
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            tokens.extend(_char_ngrams(token))
        else:
            tokens.append(token.lower())
    return tokens


def _char_ngrams(text: str) -> List[str]:
    """为连续中文生成 2-gram 和 3-gram，提升备用分词的匹配能力。"""
    if len(text) <= 2:
        return [text]
    grams = [text[i : i + 2] for i in range(len(text) - 1)]
    grams.extend(text[i : i + 3] for i in range(len(text) - 2))
    return grams
