"""Streamlit 单页问答界面。"""
import streamlit as st

from rag import config
from rag.ask import ask, normalize_math
from rag.search import Searcher

st.set_page_config(page_title="电子电路基础 Obsidian 笔记问答", page_icon="📚")
st.title("电子电路基础 Obsidian 笔记问答")


@st.cache_resource
def get_searcher() -> Searcher:
    return Searcher()


if not (config.INDEX_DIR / "chunks.jsonl").exists():
    st.error("索引不存在，先在项目根目录运行 `python -m rag.index`")
    st.stop()

searcher = get_searcher()

st.sidebar.metric("笔记数", len({c.note_path for c in searcher.chunks}))
st.sidebar.metric("chunk 数", len(searcher.chunks))
st.sidebar.caption(f"embedding: {config.EMBEDDING_MODEL}")
if not config.DEEPSEEK_API_KEY:
    st.sidebar.warning("未配置 DEEPSEEK_API_KEY，仅检索模式")

if "history" not in st.session_state:
    st.session_state.history = []

for entry in st.session_state.history:
    with st.chat_message("assistant"):
        st.markdown(normalize_math(entry["answer"]))
        if entry["sources"]:
            st.caption(
                "来源：" + "；".join(f"[{s['n']}] {s['note_path']}" for s in entry["sources"])
            )

if query := st.chat_input("问点什么……"):
    st.chat_message("user").markdown(query)
    with st.spinner("检索中……"):
        try:
            out = ask(query, searcher)
        except Exception as exc:
            st.error(f"生成失败：{exc}")
            st.stop()

    answer = out["answer"] or "未配置 API key，下面是检索到的片段。"
    with st.chat_message("assistant"):
        st.markdown(normalize_math(answer))
        if out["sources"]:
            st.caption(
                "来源：" + "；".join(f"[{s['n']}] {s['note_path']}" for s in out["sources"])
            )
        for result in out["results"]:
            label = f"{result.chunk.note_path} > {result.chunk.heading_path}"
            with st.expander(label):
                st.write(result.chunk.text)
                score = (
                    f"{result.rerank_score:.3f}"
                    if result.rerank_score is not None
                    else "-"
                )
                st.caption(f"rerank={score} | rrf={result.rrf_score:.4f}")

    st.session_state.history.append(
        {"answer": answer, "sources": out["sources"]}
    )