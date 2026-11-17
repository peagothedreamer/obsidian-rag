"""检索 + DeepSeek 生成带引用的答案。"""
import argparse
import re
import sys

from openai import OpenAI

from . import config
from .search import RetrievedChunk, Searcher, print_results

CITATION = re.compile(r"\[(\d+)\]")
BLOCK_MATH = re.compile(r"\\\[(.+?)\\\]", re.S)
INLINE_MATH = re.compile(r"\\\((.+?)\\\)", re.S)

SYSTEM_PROMPT = (
    "你是笔记库问答助手。只根据给出的编号片段回答；片段不足时直接说明。"
    "引用事实时在句末标注片段编号，如 [1]。"
    "每条答案至少引用一个片段，禁止引用未给出的编号。"
    r"数学公式使用 $...$（行内）或 $$...$$（独立成行），不要使用 \( \) 或 \[ \]。"
)


def normalize_math(text: str) -> str:
    r"""把 \( \) / \[ \] 转成 Streamlit 支持的 $...$ / $$...$$。"""
    text = BLOCK_MATH.sub(lambda m: "$$\n" + m.group(1).strip() + "\n$$", text)
    return INLINE_MATH.sub(lambda m: "$" + m.group(1).strip() + "$", text)


def build_context(results: list[RetrievedChunk]) -> tuple[str, list[dict]]:
    blocks: list[str] = []
    sources: list[dict] = []
    for n, result in enumerate(results, start=1):
        blocks.append(
            f"[{n}] {result.chunk.note_path} > {result.chunk.heading_path}\n{result.chunk.text}"
        )
        sources.append(
            {
                "n": n,
                "note_path": result.chunk.note_path,
                "heading_path": result.chunk.heading_path,
                "score": result.rerank_score,
            }
        )
    return "\n\n".join(blocks), sources


def ask(
    query: str,
    searcher: Searcher | None = None,
    top_k: int = config.TOP_K,
) -> dict:
    searcher = searcher or Searcher()
    results = searcher.search(query, top_k=top_k)
    context, sources = build_context(results)

    if not config.DEEPSEEK_API_KEY:
        return {
            "mode": "retrieval",
            "answer": None,
            "sources": sources,
            "results": results,
        }

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL
    )
    response = client.chat.completions.create(
        model=config.DEEPSEEK_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"片段：\n{context}\n\n问题：{query}"},
        ],
    )
    answer = response.choices[0].message.content or ""
    cited = sorted({int(n) for n in CITATION.findall(answer)})
    cited_sources = [source for source in sources if source["n"] in cited]
    return {
        "mode": "generation",
        "answer": answer,
        "sources": cited_sources,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="问笔记库")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    searcher = Searcher()
    try:
        out = ask(args.query, searcher, top_k=args.top_k)
    except Exception as exc:
        print(f"生成失败：{exc}", file=sys.stderr)
        print_results(searcher.search(args.query, top_k=args.top_k))
        raise SystemExit(1)

    if out["mode"] == "retrieval":
        print("仅检索模式（未配置 DEEPSEEK_API_KEY）\n")
        print_results(out["results"])
        return

    print(out["answer"])
    print("\n来源：")
    for source in out["sources"]:
        print(f"[{source['n']}] {source['note_path']} > {source['heading_path']}")


if __name__ == "__main__":
    main()