"""HyDE 实验共用：假设文档生成、磁盘缓存、文本编码。"""
import hashlib
import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from rag import config

ROOT = Path(__file__).resolve().parent.parent
PROMPT_VERSION = "hyde-zh-v1"

HYDE_PROMPT = """你是电子电路课程的助教。请针对下面的问题写一段 150-250 字的中文解答，\
像教材或讲义中的段落。不要提及“片段”或引用编号，只输出正文。

问题：{question}
"""


def load_cache(path: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["key"]] = row
    return cache


def append_cache(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_hypotheses(
    client: OpenAI,
    model: str,
    question: str,
    samples: int,
    cache: dict[str, dict],
    cache_path: Path,
) -> dict:
    """生成 samples 段假设答案，命中缓存时不调用 API。"""
    texts: list[str] = []
    calls = cached = input_tokens = output_tokens = 0
    for idx in range(samples):
        key = hashlib.sha1(
            f"{PROMPT_VERSION}|{model}|{question}|{idx}".encode("utf-8")
        ).hexdigest()
        if key in cache:
            row = cache[key]
            texts.append(row["text"])
            cached += 1
            input_tokens += row["input_tokens"]
            output_tokens += row["output_tokens"]
            continue

        response = client.chat.completions.create(
            model=model,
            temperature=0.7,
            messages=[
                {"role": "user", "content": HYDE_PROMPT.format(question=question)}
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError(f"空假设文档: {question}")

        row = {
            "key": key,
            "model": model,
            "prompt_version": PROMPT_VERSION,
            "question": question,
            "sample": idx,
            "text": text,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }
        cache[key] = row
        append_cache(cache_path, row)
        texts.append(text)
        calls += 1
        input_tokens += row["input_tokens"]
        output_tokens += row["output_tokens"]

    return {
        "texts": texts,
        "calls": calls,
        "cached": cached,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def encode(encoder, text: str) -> np.ndarray:
    return encoder.encode([text], normalize_embeddings=True)[0]
