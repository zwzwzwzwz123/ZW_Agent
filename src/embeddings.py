from __future__ import annotations

import re

import jieba
import numpy as np


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]+", re.UNICODE)
CHINESE_PATTERN = re.compile(r"[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in TOKEN_PATTERN.findall(text):
        raw = raw.strip().lower()
        if not raw:
            continue
        if CHINESE_PATTERN.search(raw):
            words = [word.strip() for word in jieba.lcut(raw) if word.strip()]
            tokens.extend(words)
            if len(raw) > 1:
                tokens.extend(raw[idx : idx + 2] for idx in range(len(raw) - 1))
        else:
            tokens.append(raw)
    return tokens


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)
