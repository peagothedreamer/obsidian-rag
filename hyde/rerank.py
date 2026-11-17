"""把 HyDE 向量加入混合检索候选池，再做精排。

对比：
1. hybrid+rerank：query 向量 + BM25 → RRF → 精排
2. hyde-aug-hybrid+rerank：query 向量 + HyDE 向量 + BM25 → RRF → 精排
3. hyde-aug-hybrid：同上但不精排，用于观察融合本身的效果

每题的候选只精排一次；基线结果取同一份精排分数在基线候选子集上的排序。
HyDE 假设文档复用 hyde/cache/hyde.jsonl，不重复调用 API。

用法（项目根目录）：
    uv run python -m hyde.rerank --questions eval/questions.paraphrase.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

from .common import encode, generate_hypotheses, load_cache, ROOT

from rag import config  # noqa: E402
from rag.eval import load_questions, mrr_at_k, recall_at_k  # noqa: E402
from rag.search import Searcher, rrf_fuse  # noqa: E402

MODES = ("hybrid+rerank", "hyde-aug-hybrid+rerank", "hyde-aug-hybrid")


def top_indices(searcher: Searcher, vector: np.ndarray, k: int) -> list[int]:
    scores = searcher.embeddings @ vector
    return [int(i) for i in np.argsort(-scores)[:k]]


def notes_at(searcher: Searcher, indices: list[int], k: int) -> list[str]:
    return [searcher.chunks[i].note_path for i in indices[:k]]


def rerank(searcher: Searcher, query: str, indices: list[int]) -> dict[int, float]:
    """对候选统一打分，返回 chunk 下标 → 分数。"""
    pairs = [(query, searcher.chunks[i].text) for i in indices]
    scores = searcher.reranker.predict(pairs)
    return {i: float(score) for i, score in zip(indices, scores)}


def ranked_by_rerank(
    scores: dict[int, float], allowed: set[int] | None = None
) -> list[int]:
    items = [
        (i, score)
        for i, score in scores.items()
        if allowed is None or i in allowed
    ]
    return [i for i, _ in sorted(items, key=lambda item: item[1], reverse=True)]


def evaluate(questions_path: Path, args) -> tuple[dict, dict]:
    searcher = Searcher()
    questions = load_questions(
        questions_path, {chunk.note_path for chunk in searcher.chunks}
    )
    if args.limit:
        questions = questions[: args.limit]

    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    cache = load_cache(cache_path)
    client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    details = {mode: [] for mode in MODES}
    api_calls = cached_hits = 0

    for number, question in enumerate(questions, start=1):
        generated = generate_hypotheses(
            client,
            args.model,
            question["question"],
            1,
            cache,
            cache_path,
        )
        api_calls += generated["calls"]
        cached_hits += generated["cached"]
        hypothesis = generated["texts"][0]

        query_hits = top_indices(
            searcher, encode(searcher.encoder, question["question"]), args.recall_k
        )
        hyde_hits = top_indices(
            searcher, encode(searcher.encoder, hypothesis), args.recall_k
        )
        bm25_hits = [
            i for i, _ in searcher.bm25_candidates(question["question"], args.recall_k)
        ]

        baseline_fused = rrf_fuse([query_hits, bm25_hits], config.RRF_K)
        baseline_allowed = {i for i, _ in baseline_fused}
        union_fused = rrf_fuse([query_hits, hyde_hits, bm25_hits], config.RRF_K)
        union_indices = [i for i, _ in union_fused]

        rerank_scores = rerank(searcher, question["question"], union_indices)
        baseline_ranked = ranked_by_rerank(rerank_scores, baseline_allowed)
        union_ranked = ranked_by_rerank(rerank_scores)

        mode_notes = {
            "hybrid+rerank": notes_at(searcher, baseline_ranked, args.top_k),
            "hyde-aug-hybrid+rerank": notes_at(searcher, union_ranked, args.top_k),
            "hyde-aug-hybrid": notes_at(searcher, union_indices, args.top_k),
        }
        for mode, notes in mode_notes.items():
            details[mode].append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "gold_note": question["gold_note"],
                    "retrieved_notes": notes,
                    "recall@5": recall_at_k(notes, question["gold_note"], 5),
                    "mrr@10": mrr_at_k(notes, question["gold_note"], 10),
                }
            )
        print(f"[{number}/{len(questions)}] {question['id']}", file=sys.stderr)

    meta = {
        "questions": str(questions_path.relative_to(ROOT)),
        "n": len(questions),
        "recall_k": args.recall_k,
        "top_k": args.top_k,
        "embedding": config.EMBEDDING_MODEL,
        "rerank_model": config.RERANK_MODEL,
        "api_calls": api_calls,
        "cache_hits": cached_hits,
    }
    return details, meta


def summarize(details: dict[str, list[dict]]) -> list[dict]:
    return [
        {
            "mode": mode,
            "recall@5": float(np.mean([row["recall@5"] for row in rows])),
            "mrr@10": float(np.mean([row["mrr@10"] for row in rows])),
            "questions": len(rows),
        }
        for mode, rows in details.items()
    ]


def write_markdown(path: Path, summary: list[dict], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HyDE 向量加入混合检索的实验结果",
        "",
        f"- 题集：`{meta['questions']}`",
        f"- 题数：{meta['n']}，召回 top-{meta['recall_k']}，精排 top-{meta['top_k']}",
        f"- embedding：`{meta['embedding']}`，rerank：`{meta['rerank_model']}`",
        f"- 假设文档：API 调用 {meta['api_calls']} 次，缓存命中 {meta['cache_hits']} 次",
        "",
        "| 模式 | Recall@5 | MRR@10 | 题目数 |",
        "|---|---|---|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['mode']} | {row['recall@5']:.3f} | {row['mrr@10']:.3f} | "
            f"{row['questions']} |"
        )
    lines += [
        "",
        "`hyde-aug-hybrid` 不经过精排，用于区分「融合收益」和「精排收益」。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE + 混合检索 + 精排")
    parser.add_argument("--questions", default="eval/questions.paraphrase.jsonl")
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default=config.DEEPSEEK_MODEL)
    parser.add_argument("--cache", default="hyde/cache/hyde.jsonl")
    parser.add_argument(
        "--out", default="reports/hyde/rerank_paraphrase.json"
    )
    args = parser.parse_args()

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = ROOT / questions_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    details, meta = evaluate(questions_path, args)
    summary = summarize(details)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"meta": meta, "summary": summary, "details": details},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(out_path.with_suffix(".md"), summary, meta)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
