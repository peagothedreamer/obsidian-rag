"""检索评测：4 种配置消融，输出 Recall@5 / MRR@10。"""
import argparse
import json
import random
import time
from pathlib import Path

from . import config
from .index import load_chunks
from .search import Searcher


def load_questions(path: Path, note_paths: set[str] | None = None) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line) for line in lines if line.strip()]
    if note_paths is not None:
        bad = [q["id"] for q in questions if q["gold_note"] not in note_paths]
        if bad:
            raise SystemExit(f"以下题目的 gold_note 不存在: {bad}")
    return questions


def recall_at_k(notes: list[str], gold_note: str, k: int = 5) -> bool:
    return gold_note in notes[:k]


def mrr_at_k(notes: list[str], gold_note: str, k: int = 10) -> float:
    seen: list[str] = []
    for note in notes[:k]:
        if note not in seen:
            seen.append(note)
    if gold_note not in seen:
        return 0.0
    return 1.0 / (seen.index(gold_note) + 1)


def run_config(searcher: Searcher, questions: list[dict], mode: str) -> dict:
    started = time.perf_counter()
    details: list[dict] = []
    for question in questions:
        query = question["question"]
        if mode == "vector":
            notes = [searcher.chunks[i].note_path for i, _ in searcher.vector_candidates(query, 10)]
        elif mode == "bm25":
            notes = [searcher.chunks[i].note_path for i, _ in searcher.bm25_candidates(query, 10)]
        else:
            results = searcher.search(query, top_k=10, no_rerank=mode == "hybrid")
            notes = [r.chunk.note_path for r in results]
        details.append(
            {
                "id": question["id"],
                "question": query,
                "gold_note": question["gold_note"],
                "retrieved_notes": notes,
                "recall@5": recall_at_k(notes, question["gold_note"], 5),
                "mrr@10": mrr_at_k(notes, question["gold_note"], 10),
            }
        )
    count = len(details)
    return {
        "mode": mode,
        "recall@5": sum(d["recall@5"] for d in details) / count,
        "mrr@10": sum(d["mrr@10"] for d in details) / count,
        "seconds": time.perf_counter() - started,
        "questions": count,
        "details": details,
    }


def run_eval(
    searcher: Searcher, questions: list[dict], with_generation: bool = False
) -> list[dict]:
    results = [run_config(searcher, questions, mode) for mode in
               ("vector", "bm25", "hybrid", "hybrid+rerank")]
    if with_generation:
        from .ask import ask

        started = time.perf_counter()
        hits = 0
        answered = 0
        for question in questions:
            out = ask(question["question"], searcher)
            if out["mode"] != "generation":
                raise SystemExit("引用命中率需要设置 DEEPSEEK_API_KEY")
            if out["sources"]:
                answered += 1
                if any(s["note_path"] == question["gold_note"] for s in out["sources"]):
                    hits += 1
        results.append(
            {
                "mode": "hybrid+rerank 引用命中率",
                "recall@5": None,
                "mrr@10": None,
                "citation_hit_rate": hits / answered if answered else 0.0,
                "seconds": time.perf_counter() - started,
                "questions": len(questions),
                "details": [],
            }
        )
    return results


def write_report(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 检索评测结果",
        "",
        "| 配置 | Recall@5 | MRR@10 | 引用命中率 | 题目数 | 耗时(s) |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        recall = "-" if result.get("recall@5") is None else f"{result['recall@5']:.3f}"
        mrr = "-" if result.get("mrr@10") is None else f"{result['mrr@10']:.3f}"
        citation = result.get("citation_hit_rate")
        citation_text = "-" if citation is None else f"{citation:.3f}"
        lines.append(
            f"| {result['mode']} | {recall} | {mrr} | {citation_text} | "
            f"{result['questions']} | {result['seconds']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draft_questions(index_dir: Path, n: int, out_path: Path) -> None:
    chunks = load_chunks(index_dir)
    rng = random.Random(42)
    samples = rng.sample(chunks, min(n, len(chunks)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for i, chunk in enumerate(samples, start=1):
            heading = chunk.heading_path.split(" > ")[-1] if chunk.heading_path else chunk.note_title
            question = {
                "id": f"draft-{i:02d}",
                "question": f"{chunk.note_title} 中的 {heading} 讲了什么？",
                "gold_note": chunk.note_path,
                "gold_heading": chunk.heading_path,
            }
            fh.write(json.dumps(question, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--questions", type=Path, default=config.PROJECT_ROOT / "eval/questions.jsonl")
    parser.add_argument(
        "--out", type=Path, default=config.PROJECT_ROOT / "reports/eval/base.md"
    )
    parser.add_argument("--generation", action="store_true", help="额外统计引用命中率")
    parser.add_argument("--draft", type=int, default=0, help="生成 N 条草稿题后退出")
    args = parser.parse_args()

    if args.draft:
        out = config.PROJECT_ROOT / "eval/questions.draft.jsonl"
        draft_questions(config.INDEX_DIR, args.draft, out)
        print(f"草稿已写入 {out}")
        return

    searcher = Searcher()
    questions = load_questions(args.questions, {c.note_path for c in searcher.chunks})
    results = run_eval(searcher, questions, with_generation=args.generation)
    write_report(args.out, results)
    json_path = args.out.with_suffix(".json")
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入 {args.out} 和 {json_path}")
    for result in results:
        print(result["mode"], result.get("recall@5"), result.get("mrr@10"))


if __name__ == "__main__":
    main()