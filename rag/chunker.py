"""Markdown → chunks，按 Obsidian 标题层级切分。"""
import hashlib
import re
from dataclasses import dataclass

from . import config

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$") # 正则表达式 ：一到六个# + 空格 + 标题文字
IMAGE_EMBED = re.compile(r"!\[\[[^\]]*\]\]") # 图片
WIKILINK = re.compile(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]") # obsidian 双链语法


@dataclass
class Chunk:
    chunk_id: str
    note_path: str
    note_title: str
    heading_path: str
    text: str
    char_count: int
    ordinal: int


def strip_frontmatter(text: str) -> str:
    """去掉文件开头的 YAML frontmatter。"""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4:]
    return text.lstrip("\n")


def clean_obsidian(text: str) -> str:
    """删掉图片嵌入，把 wikilink 还原成纯文本。"""
    text = IMAGE_EMBED.sub("", text)
    return WIKILINK.sub(lambda m: (m.group(2) or m.group(1)).strip(), text)


def split_sections(text: str) -> list[tuple[str, str]]:
    """按 ATX 标题切段，返回 (标题路径, 正文)。代码块内的 # 不算标题。"""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    buf: list[str] = []
    in_fence = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append((" > ".join(t for _, t in stack), body))
        buf.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            buf.append(line)
            continue
        match = None if in_fence else HEADING.match(stripped) # if in_fence: match = None
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
        else:
            buf.append(line)
    flush()
    return sections


def pack_paragraphs(body: str, max_chars: int, overlap: int) -> list[str]:
    """按段落打包，超长时切分，相邻块保留 overlap 字符重叠。"""
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur = ""
    for para in paragraphs:
        while para:
            room = max_chars - len(cur) - (2 if cur else 0)
            if room <= 0:
                chunks.append(cur)
                cur = cur[-overlap:] if overlap else ""
                continue
            piece, para = para[:room], para[room:]
            cur = f"{cur}\n\n{piece}" if cur else piece
            if para:
                chunks.append(cur)
                cur = cur[-overlap:] if overlap else ""
    if cur.strip():
        chunks.append(cur)
    return chunks


def _prefix(note_title: str, heading_path: str) -> str:
    """chunk 前缀：笔记名 + 标题路径，去重。"""
    parts = [note_title]
    parts += [h for h in heading_path.split(" > ") if h and h != note_title]
    return " > ".join(dict.fromkeys(parts))


def chunk_note(
    note_path: str,
    note_title: str,
    text: str,
    max_chars: int | None = None,
    overlap: int | None = None,
    min_chars: int | None = None,
) -> list[Chunk]:
    max_chars = max_chars or config.CHUNK_MAX_CHARS
    overlap = overlap if overlap is not None else config.CHUNK_OVERLAP_CHARS
    min_chars = min_chars or config.CHUNK_MIN_CHARS

    text = clean_obsidian(strip_frontmatter(text))
    chunks: list[Chunk] = []
    for heading_path, body in split_sections(text):
        if len(body) < min_chars and chunks:
            prev = chunks[-1]
            prev.text = f"{prev.text}\n\n{body}"
            prev.char_count = len(prev.text)
            continue
        prefix = _prefix(note_title, heading_path)
        for piece in pack_paragraphs(body, max_chars, overlap):
            full = f"{prefix}\n{piece}"
            ordinal = len(chunks)
            cid = hashlib.sha1(
                f"{note_path}#{heading_path}#{ordinal}".encode("utf-8")
            ).hexdigest()[:16]
            chunks.append(
                Chunk(cid, note_path, note_title, heading_path, full, len(full), ordinal)
            )
    return chunks