import numpy as np

from rag.chunker import chunk_note, pack_paragraphs

# 测试切块逻辑
def test_chunk_heading_paths_ignore_code_fence():
    text = (
        "# 第一章\n\n"
        + "引言内容，" * 20 + "\n\n"
        "```python\n# 这不是标题\nprint(1)\n```\n\n"
        "## 1.1 小节\n\n"
        + "小节正文内容，" * 20 + "\n"
    )
    chunks = chunk_note("课程/笔记.md", "笔记", text)
    paths = {c.heading_path for c in chunks}
    assert "第一章" in paths
    assert "第一章 > 1.1 小节" in paths
    assert all("这不是标题" not in p for p in paths)
    assert all(c.text.startswith("笔记 > 第一章") for c in chunks)

# 长章节切分与100字重叠
def test_pack_paragraphs_overlaps_and_caps_length():
    parts = pack_paragraphs("字" * 2000, max_chars=800, overlap=100)
    assert len(parts) >= 3
    assert all(len(p) <= 800 for p in parts)
    assert parts[0][-100:] == parts[1][:100]

from rag import index as index_mod

# 更改后重建索引
def test_incremental_index_reuses_unchanged(tmp_path):
    vault = tmp_path / "vault"
    (vault / "课程").mkdir(parents=True)
    note_a = vault / "课程" / "a.md"
    note_b = vault / "课程" / "b.md"
    note_a.write_text("# A\n\n" + "甲" * 300, encoding="utf-8")
    note_b.write_text("# B\n\n" + "乙" * 300, encoding="utf-8")

    calls: list[int] = []

    def fake_embed(texts):
        calls.append(len(texts))
        return np.ones((len(texts), 4), dtype=np.float32)

    index_dir = tmp_path / "index"

    stats1 = index_mod.build_index(
        index_dir=index_dir, vault_dir=vault, course_dirs=["课程"], embed_fn=fake_embed
    )
    assert stats1["chunks"] == 2
    assert stats1["embedded_chunks"] == 2

    stats2 = index_mod.build_index(
        index_dir=index_dir, vault_dir=vault, course_dirs=["课程"], embed_fn=fake_embed
    )
    assert stats2["embedded_chunks"] == 0
    assert stats2["reused_chunks"] == 2

    note_a.write_text("# A\n\n" + "甲" * 400, encoding="utf-8")
    stats3 = index_mod.build_index(
        index_dir=index_dir, vault_dir=vault, course_dirs=["课程"], embed_fn=fake_embed
    )
    assert stats3["embedded_chunks"] == 1
    assert stats3["reused_chunks"] == 1
    assert calls[-1] == 1

from rag.search import rrf_fuse


def test_rrf_fuse_prefers_consistent_rank():
    # 9 在两路都排第 5（2/65）；0 只在一路排第 1（1/61），应排在 9 后面。
    fused = rrf_fuse([[0, 1, 2, 3, 9], [8, 7, 6, 5, 9]], k=60)
    assert fused[0][0] == 9
    assert fused[0][1] > fused[1][1]