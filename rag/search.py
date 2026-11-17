"""混合检索：向量 + BM25，RRF 融合，可选精排。"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from . import config  # 必须先于 sentence_transformers 导入（设置 HF 镜像）
from .chunker import Chunk
from .index import load_chunks
from sentence_transformers import CrossEncoder, SentenceTransformer


@dataclass
class RetrievedChunk:
    chunk: Chunk
    vector_rank: int | None
    bm25_rank: int | None
    rrf_score: float
    rerank_score: float | None = None


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for token in jieba.lcut(text.lower()):
        token = token.strip()
        if token and any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in token):
            tokens.append(token)
    return tokens


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class Searcher:
    def __init__(self, index_dir: Path | None = None, embedding_model: str | None = None):
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.chunks = load_chunks(self.index_dir)
        if not self.chunks:
            raise FileNotFoundError(
                f"索引为空：{self.index_dir}，先运行 python -m rag.index"
            )
        emb_path = self.index_dir / "embeddings.npy"
        if not emb_path.exists():
            raise FileNotFoundError(f"缺少 {emb_path}，先运行 python -m rag.index")
        self.embeddings = np.load(emb_path)
        if len(self.embeddings) != len(self.chunks):
            raise ValueError("embeddings.npy 与 chunks.jsonl 数量不一致，请重建索引")
        self.encoder = SentenceTransformer(embedding_model or config.EMBEDDING_MODEL)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])
        self._reranker: CrossEncoder | None = None

    @property
    def reranker(self) -> CrossEncoder:
        if self._reranker is None:
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    def encode_query(self, query: str) -> np.ndarray:
        return self.encoder.encode([query], normalize_embeddings=True)[0]

    def vector_candidates(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self.embeddings @ self.encode_query(query)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def bm25_candidates(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in order]

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K,
        no_rerank: bool = False,
        recall_k: int | None = None,
        # rerank_n: int = 20,
        rerank_n: int | None = None,
    ) -> list[RetrievedChunk]:
        recall_k = recall_k or config.RECALL_K
        rerank_n = rerank_n or recall_k * 2
        vector_hits = self.vector_candidates(query, recall_k)
        bm25_hits = self.bm25_candidates(query, recall_k)
        fused = rrf_fuse(
            [[i for i, _ in vector_hits], [i for i, _ in bm25_hits]], config.RRF_K
        )
        vector_rank = {i: r for r, (i, _) in enumerate(vector_hits, start=1)}
        bm25_rank = {i: r for r, (i, _) in enumerate(bm25_hits, start=1)}

        rerank_scores: dict[int, float] = {}
        candidates = fused[:top_k] if no_rerank else fused[:rerank_n]
        if not no_rerank:
            pairs = [(query, self.chunks[i].text) for i, _ in candidates]
            predictions = self.reranker.predict(pairs)
            rerank_scores = {
                i: float(score) for (i, _), score in zip(candidates, predictions)
            }
            candidates = sorted(
                candidates, key=lambda item: rerank_scores[item[0]], reverse=True
            )[:top_k]

        return [
            RetrievedChunk(
                chunk=self.chunks[i],
                vector_rank=vector_rank.get(i),
                bm25_rank=bm25_rank.get(i),
                rrf_score=score,
                rerank_score=rerank_scores.get(i),
            )
            for i, score in candidates
        ]


def print_results(results: list[RetrievedChunk]) -> None:
    for n, result in enumerate(results, start=1):
        score = f"{result.rerank_score:.3f}" if result.rerank_score is not None else "-"
        print(f"[{n}] {result.chunk.note_path} > {result.chunk.heading_path}")
        print(f"    rerank={score} rrf={result.rrf_score:.4f}")
        print("    " + result.chunk.text[:200].replace("\n", " "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="检索 Obsidian 笔记")
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()
    searcher = Searcher()
    print_results(searcher.search(args.query, top_k=args.top_k, no_rerank=args.no_rerank))


if __name__ == "__main__":
    main()