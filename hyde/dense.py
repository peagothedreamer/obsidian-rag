"""HyDE 纯稠密检索实验：比较 query / hyde / query+hyde / multi-hyde。

用法（项目根目录）:
    uv run python -m hyde.dense --questions eval/questions.jsonl --samples 3
    uv run python -m hyde.dense --questions eval/questions.paraphrase.jsonl \
        --samples 3 --out reports/hyde/dense_paraphrase.json

只改变查询表示，不动索引、embedding 模型和切块。
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
from rag.search import Searcher  # noqa: E402

MODES = ("query", "hyde", "query+hyde", "multi-hyde")


def mode_vector(mode: str, question: str, hypotheses: list[str], encoder) -> np.ndarray:
    if mode == "query":
        return encode(encoder, question)
    if mode == "hyde":
        return encode(encoder, hypotheses[0])
    if mode == "query+hyde":
        return encode(encoder, f"{question}\n\n{hypotheses[0]}")
    if mode == "multi-hyde":
        vectors = [encode(encoder, text) for text in hypotheses]
        mean = np.mean(vectors, axis=0)
        return mean / np.linalg.norm(mean)
    raise ValueError(f"未知模式: {mode}")


def bootstrap_ci(values: list[float], iters: int = 2000, seed: int = 42) -> list:
    if len(values) < 2:
        return [None, None]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    means = [
        float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(iters)
    ]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(details: dict[str, list[dict]]) -> list[dict]:
    baseline = {row["id"]: row for row in details["query"]}
    summary = []
    for mode in MODES:
        rows = details[mode]
        deltas = [row["mrr@10"] - baseline[row["id"]]["mrr@10"] for row in rows]
        summary.append(
            {
                "mode": mode,
                "recall@5": float(np.mean([row["recall@5"] for row in rows])),
                "mrr@10": float(np.mean([row["mrr@10"] for row in rows])),
                "mrr_delta": float(np.mean(deltas)),
                "mrr_delta_ci95": bootstrap_ci(deltas),
                "wins": sum(delta > 1e-9 for delta in deltas),
                "losses": sum(delta < -1e-9 for delta in deltas),
                "ties": sum(abs(delta) <= 1e-9 for delta in deltas),
            }
        )
    return summary


def write_markdown(path: Path, summary: list[dict], cost: dict, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HyDE 实验结果",
        "",
        f"- 题集：`{meta['questions']}`",
        f"- 题数：{meta['n']}，每问假设文档数：{meta['samples']}",
        f"- 生成模型：`{meta['model']}`，检索模型：`{meta['embedding']}`",
        f"- API 调用 {cost['api_calls']} 次，缓存命中 {cost['cached']} 次，"
        f"输入 {cost['input_tokens']} tokens，输出 {cost['output_tokens']} tokens",
        "",
        "| 模式 | Recall@5 | MRR@10 | ΔMRR vs query | 95% CI | W/L/T |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary:
        ci = row["mrr_delta_ci95"]
        ci_text = "-" if ci[0] is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(
            f"| {row['mode']} | {row['recall@5']:.3f} | {row['mrr@10']:.3f} | "
            f"{row['mrr_delta']:+.3f} | {ci_text} | "
            f"{row['wins']}/{row['losses']}/{row['ties']} |"
        )
    lines += [
        "",
        "W/L/T：与 query 基线逐题比较 MRR@10 的胜/负/平。",
        "小样本下以方向和逐题胜负为主，不把小差距写成精确提升。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run(args) -> None:
    if not config.DEEPSEEK_API_KEY:
        raise SystemExit("需要设置 DEEPSEEK_API_KEY")

    questions_path = Path(args.questions)
    if not questions_path.is_absolute():
        questions_path = ROOT / questions_path
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = ROOT / cache_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    searcher = Searcher()
    questions = load_questions(
        questions_path, {chunk.note_path for chunk in searcher.chunks}
    )
    if args.limit:
        questions = questions[: args.limit]

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL
    )
    cache = load_cache(cache_path)

    details: dict[str, list[dict]] = {mode: [] for mode in MODES}
    cost = {"api_calls": 0, "cached": 0, "input_tokens": 0, "output_tokens": 0}

    for n, question in enumerate(questions, start=1):
        generated = generate_hypotheses(
            client,
            args.model,
            question["question"],
            args.samples,
            cache,
            cache_path,
        )
        cost["api_calls"] += generated["calls"]
        cost["cached"] += generated["cached"]
        cost["input_tokens"] += generated["input_tokens"]
        cost["output_tokens"] += generated["output_tokens"]

        for mode in MODES:
            vector = mode_vector(
                mode, question["question"], generated["texts"], searcher.encoder
            )
            scores = searcher.embeddings @ vector
            top = np.argsort(-scores)[:10]
            notes = [searcher.chunks[int(i)].note_path for i in top]
            row = {
                "id": question["id"],
                "question": question["question"],
                "gold_note": question["gold_note"],
                "retrieved_notes": notes,
                "recall@5": recall_at_k(notes, question["gold_note"], 5),
                "mrr@10": mrr_at_k(notes, question["gold_note"], 10),
            }
            if mode == "hyde":
                row["hypothesis"] = generated["texts"][0]
            details[mode].append(row)

        print(f"[{n}/{len(questions)}] {question['id']}", file=sys.stderr)

    summary = summarize(details)
    meta = {
        "questions": str(questions_path.relative_to(ROOT)),
        "n": len(questions),
        "samples": args.samples,
        "model": args.model,
        "embedding": config.EMBEDDING_MODEL,
    }
    payload = {"meta": meta, "cost": cost, "summary": summary, "details": details}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(out_path.with_suffix(".md"), summary, cost, meta)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="HyDE 纯稠密检索实验")
    parser.add_argument(
        "--questions", default="eval/questions.jsonl", help="评测题集 jsonl"
    )
    parser.add_argument("--samples", type=int, default=3, help="每题生成几段假设答案")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 题（冒烟用）")
    parser.add_argument("--model", default=config.DEEPSEEK_MODEL)
    parser.add_argument("--cache", default="hyde/cache/hyde.jsonl")
    parser.add_argument("--out", default="reports/hyde/dense.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
