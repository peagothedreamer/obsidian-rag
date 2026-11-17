"""构建和增量更新 chunk + embedding 索引。"""
import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import config
from .chunker import Chunk, chunk_note
from sentence_transformers import SentenceTransformer

_model = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


def _default_embed(texts: list[str]) -> np.ndarray:
    return _get_model().encode(
        texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iter_note_files(vault_dir: Path, course_dirs: list[str]) -> list[Path]:
    files: list[Path] = []
    for course in course_dirs:
        root = vault_dir / course
        if not root.is_dir():
            raise FileNotFoundError(f"课程目录不存在: {root}")
        for path in sorted(root.rglob("*.md")):
            dirs = path.relative_to(root).parts[:-1]
            if any(d.startswith(config.EXCLUDE_DIR_PREFIXES) for d in dirs):
                continue
            files.append(path)
    return files


def load_chunks(index_dir: Path) -> list[Chunk]:
    path = index_dir / "chunks.jsonl"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Chunk(**json.loads(line)) for line in lines if line.strip()]


def build_index(
    index_dir: Path | None = None,
    vault_dir: Path | None = None,
    course_dirs: list[str] | None = None,
    rebuild: bool = False,
    embed_fn=None,
) -> dict:
    index_dir = Path(index_dir or config.INDEX_DIR)
    vault_dir = Path(vault_dir or config.VAULT)
    course_dirs = course_dirs or config.COURSE_DIRS
    embed_fn = embed_fn or _default_embed
    index_dir.mkdir(parents=True, exist_ok=True)

    files = iter_note_files(vault_dir, course_dirs)
    current_hash = {f.relative_to(vault_dir).as_posix(): _sha256(f) for f in files}

    old_state: dict[str, str] = {}
    state_path = index_dir / "state.json"
    if not rebuild and state_path.exists():
        old_state = json.loads(state_path.read_text(encoding="utf-8"))

    old_chunks = [] if rebuild else load_chunks(index_dir)
    emb_path = index_dir / "embeddings.npy"
    old_vec_by_id: dict[str, np.ndarray] = {}
    if old_chunks and emb_path.exists():
        old_vectors = np.load(emb_path)
        if len(old_vectors) == len(old_chunks):
            for chunk, vec in zip(old_chunks, old_vectors):
                old_vec_by_id[chunk.chunk_id] = vec

    old_chunks_by_note: dict[str, list[Chunk]] = {}
    for chunk in old_chunks:
        old_chunks_by_note.setdefault(chunk.note_path, []).append(chunk)

    all_chunks: list[Chunk] = []
    new_chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    reused = 0
    for path in files:
        rel = path.relative_to(vault_dir).as_posix()
        old_note_chunks = old_chunks_by_note.get(rel, [])
        unchanged = old_state.get(rel) == current_hash[rel] and all(
            c.chunk_id in old_vec_by_id for c in old_note_chunks
        )
        if unchanged:
            all_chunks.extend(old_note_chunks)
            reused += len(old_note_chunks)
        else:
            try:
                note_chunks = chunk_note(
                    rel, path.stem, path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                skipped.append((rel, str(exc)))
                continue
            all_chunks.extend(note_chunks)
            new_chunks.extend(note_chunks)

    # 解析失败的文件不写入 state，下次运行会重试。
    for rel, _ in skipped:
        current_hash.pop(rel, None)
    if skipped:
        print(f"跳过 {len(skipped)} 个文件:", file=sys.stderr)
        for rel, message in skipped:
            print(f"  {rel}: {message}", file=sys.stderr)

    new_vec_by_id: dict[str, np.ndarray] = {}
    if new_chunks:
        new_vectors = np.asarray(embed_fn([c.text for c in new_chunks]), dtype=np.float32)
        if len(new_vectors) != len(new_chunks):
            raise ValueError("embed_fn 返回的向量数与 chunk 数不一致")
        for chunk, vec in zip(new_chunks, new_vectors):
            new_vec_by_id[chunk.chunk_id] = vec

    if all_chunks:
        vectors = np.stack(
            [
                new_vec_by_id[c.chunk_id] if c.chunk_id in new_vec_by_id else old_vec_by_id[c.chunk_id]
                for c in all_chunks
            ]
        )
    else:
        vectors = np.zeros((0, 0), dtype=np.float32)

    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(emb_path, vectors)
    state_path.write_text(
        json.dumps(current_hash, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    return {
        "notes": len(files),
        "chunks": len(all_chunks),
        "reused_chunks": reused,
        "embedded_chunks": len(new_chunks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 Obsidian 笔记索引")
    parser.add_argument("--rebuild", action="store_true", help="忽略缓存全量重建")
    args = parser.parse_args()
    stats = build_index(rebuild=args.rebuild)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()