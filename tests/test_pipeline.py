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