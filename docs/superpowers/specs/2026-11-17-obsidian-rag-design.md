# 本地 Obsidian 笔记 RAG Demo 设计文档

- 日期：2026-11-17
- 状态：待实现
- 项目位置：`F:\Desktop\obsidian-rag`

## 1. 背景与目标

面向简历的 RAG 项目。数据源是作者本机 Obsidian 笔记库中五门课的笔记，共 804 篇 Markdown：

- 复变函数与积分变换（51）
- 概率论与数理统计（95）
- 大学物理Ⅱ（46）
- 电子电路基础（478）
- 软件技术基础（134）

内容为中文教材笔记，含 LaTeX 公式。

目标：

1. 对笔记库做可引用的问答：答案给出 `[n]` 引用，可追溯到具体笔记和标题。
2. 检索质量可量化：自建评测集，对 4 种检索配置做消融，产出可写进简历的指标。

成功标准：

- `python -m rag.eval` 一条命令产出 `reports/eval.md`，含 Recall@5、MRR@10 的四行对比表。
- 典型问题（如「柯西积分公式的成立条件」）的 top-5 检索结果命中对应笔记。
- Streamlit 页面可完成一次带引用的问答，并能看到来源片段。

## 2. 非目标

- 只索引上述五门课的目录，vault 内其他文件夹（AI 学习、偏微分方程、西班牙语等）不索引。
- 不解析图片、PDF 等附件（vault 内 1.1G 主要是附件）。
- 不做文件实时监听；索引按命令手动更新。
- 不做多用户、鉴权、部署。
- 不引入向量数据库，不做 ANN 索引。
- 不做 query rewriting、多跳检索、fine-tune。
- 不做流式输出。

## 3. 技术选型

| 组件 | 选择 | 理由 |
|---|---|---|
| 运行环境 | Python 3.13 + uv + CPU 版 torch | 4G 显存笔记本，embedding 和 rerank 放 CPU |
| Embedding | `BAAI/bge-m3` | 中英混排和公式文本表现稳，1024 维 |
| 稀疏检索 | `rank_bm25` + `jieba` | 中文分词后 BM25，补足专有名词和公式关键词 |
| 融合 | RRF（k=60） | 无需调权重，对两路召回分数尺度不敏感 |
| 精排 | `BAAI/bge-reranker-base` | 278M，CPU 上对 20 个候选精排约 1 秒内 |
| 生成 | DeepSeek `deepseek-chat` | OpenAI 兼容接口，价格低，中文好 |
| 界面 | Streamlit 单页 | 100 行内完成聊天和来源展示 |
| 存储 | jsonl + numpy | 约 3000 个 chunk，全量点积毫秒级，不需要数据库 |

依赖带来的成本：首次运行下载 bge-m3（约 2.2GB）和 reranker（约 1.1GB）；804 篇笔记全量建索引预计 5–10 分钟（CPU），之后按文件变更增量更新。

## 4. 架构与数据流

```
vault/*.md
   │  chunker（按标题切块，保留路径元数据）
   ▼
chunks.jsonl ──► bge-m3 ──► embeddings.npy
   │
   └──────────► jieba + BM25 ──► bm25 语料（运行时由 chunks.jsonl 重建）

query
   ├─ bge-m3 → 余弦 top-20
   └─ jieba  → BM25 top-20
        └─ RRF(k=60) → 候选 top-20
             └─ bge-reranker-base → top-5
                  ├─ DeepSeek 生成答案（带 [n] 引用）
                  └─ Streamlit 展示来源片段
```

## 5. 模块设计

### 5.1 配置（`rag/config.py`）

环境变量，均有默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OBSIDIAN_VAULT` | `F:\Desktop\Obsidian Vault` | 笔记库根目录 |
| `COURSE_DIRS` | 五门课目录名，逗号分隔 | 限定索引范围；目录不存在时报错 |
| `DEEPSEEK_API_KEY` | 无 | 缺失时进入仅检索模式 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 生成模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 便于消融时替换 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 精排模型 |

模型名集中在此处，代码其他部分不写死。

### 5.2 切块（`rag/chunker.py`）

- 只遍历 `COURSE_DIRS` 下所有 `.md`，跳过 `.obsidian/` 目录。
- 去掉 YAML frontmatter 后再切。
- 按 ATX 标题（`#` 到 `######`）分段，忽略代码块内的 `#`。
- chunk 正文 = `笔记名 > H1 > H2` 前缀 + 该标题下正文。
- 段落拼接至不超过 800 字符；超出时按段落切开，相邻 chunk 重叠 100 字符。小于 50 字符的碎块并入上一块。
- Obsidian 语法处理：
  - `![[image.png]]` 嵌入直接删除。
  - `[[note|alias]]` 保留 alias，`[[note]]` 保留 note 文本。
  - 普通 Markdown 链接保留锚文本。

### 5.3 索引（`rag/index.py`）

命令：`python -m rag.index`（默认增量）。

产物：

- `index/chunks.jsonl`：每行一个 chunk，字段见 6.1。
- `index/embeddings.npy`：float32，L2 归一化，行序与 chunks.jsonl 一致。
- `index/state.json`：`{note_path: sha256}`，用于增量。

增量逻辑：

1. 扫描 `COURSE_DIRS`，计算每个 md 文件内容 sha256。
2. 与 state.json 对比，找出新增、修改、删除的笔记。
3. 删除笔记的所有 chunk；新增和修改笔记重新切块、重新 embedding。
4. 其他 chunk 的向量从旧 npy 按索引拼接保留。
5. 重写三个产物。BM25 语料不落盘，检索时从 chunks.jsonl 重建（几千行，毫秒级）。

`--rebuild` 参数：忽略 state，全量重建。

模型编码使用 `sentence-transformers`，batch size 32，CPU。

### 5.4 检索（`rag/search.py`）

接口：`search(query: str, top_k: int = 5, no_rerank: bool = False) -> list[RetrievedChunk]`

1. 向量召回：query 用同一 bge-m3 编码，与 embeddings.npy 点积，取 top-20。
2. BM25 召回：jieba 精确模式分词，过滤空白和纯标点 token，取 top-20。
3. RRF 融合：`score = Σ 1/(60 + rank)`，融合后取 top-20。只出现在一路的 chunk 仅累加该路得分。
4. 精排：reranker 对 20 个候选打分，取 top_k。
5. 返回 chunk 文本、note_path、heading_path、各阶段分数。

`search.py` 同时提供 CLI（`python -m rag.search "问题"`），打印 top-5 片段和分数。评测时以 `top_k=10` 调用，供 MRR@10 使用。

### 5.5 生成（`rag/ask.py`）

命令：`python -m rag.ask "问题"`。

- 调用 `search()` 取 top-5，拼 prompt：

```
你是笔记库问答助手。只根据下面编号片段回答；片段不足时直接说明。
引用事实时在句末标注片段编号，如 [1]。
每条答案至少引用一个片段，禁止引用未给出的编号。

[1] 复变函数与积分变换/03-复变函数的积分/3.3-柯西积分公式.md > 柯西积分公式
<片段正文>
...
```

- `temperature=0.2`，非流式。
- 解析答案中的 `[n]`，映射为来源列表输出（note_path + heading_path）。
- 无 `DEEPSEEK_API_KEY` 时：跳过生成，打印检索结果并提示「仅检索模式」。
- 生成失败（网络、限流）时打印错误并回退到仅检索输出，不静默吞错。

### 5.6 评测（`rag/eval.py`）

数据：`eval/questions.jsonl`，每行：

```json
{"id": "q01", "question": "柯西积分公式成立的条件是什么？", "gold_note": "复变函数与积分变换/03-复变函数的积分/3.3-柯西积分公式.md", "gold_heading": "柯西积分公式"}
```

- `gold_heading` 可选，v1 指标只按 note 层面计算。
- 规模：30–50 题，覆盖复变、概统、大物、电子电路、软件技术基础等至少 4 门课。
- 草稿生成：`python -m rag.eval --draft 20` 从 vault 随机抽样标题，按「<笔记名> 中的 <标题> 讲了什么？」模板写入 `eval/questions.draft.jsonl`。人工筛选、改写后合入正式文件，草稿本身不参与评测。
- vector 和 bm25 单路配置直接调用对应召回函数，不经过融合。

四种配置，每组跑全部题目：

| 配置 | 召回 |
|---|---|
| vector | 余弦 top-10 |
| bm25 | BM25 top-10 |
| hybrid | RRF top-10，不精排 |
| hybrid+rerank | RRF top-20 → 精排 top-10 |

指标（按笔记去重计算）：

- **Recall@5**：每组配置返回结果的前 5 条涉及的笔记集合包含 `gold_note` 的题目占比。
- **MRR@10**：取 top-10 中去重后第一次出现 `gold_note` 的排名 r，得分 1/r；未命中记 0；对所有题取平均。
- **引用命中率**（需要 API）：在答案至少含一个有效 `[n]` 引用的题目中，被引用片段至少一个来自 `gold_note` 的比例。仅对 hybrid+rerank 配置统计。

输出：

- `reports/eval.md`：四行配置 × 三列指标的表，附题目数和运行时间。
- `reports/eval.json`：逐题明细，便于排查失败案例。

评测不调用 DeepSeek 也能跑检索三指标，README 中的消融表以此为准。

### 5.7 界面（`app.py`）

Streamlit 单页：

- 输入框提问，提交后展示答案；答案下方列出被引用的来源。
- `st.expander` 展示全部 top-5 片段：笔记路径、标题路径、rerank 分数。
- 侧边栏显示索引统计（chunk 数、笔记数、embedding 模型名）和「仅检索模式」提示。
- 首屏无索引时提示先运行 `python -m rag.index`，不自动建索引。

## 6. 数据契约

### 6.1 chunk（chunks.jsonl）

```json
{
  "chunk_id": "sha1(note_path + '#' + heading_path + '#' + ordinal)",
  "note_path": "复变函数与积分变换/03-复变函数的积分/3.3-柯西积分公式.md",
  "note_title": "3.3-柯西积分公式",
  "heading_path": "柯西积分公式 > 定理陈述",
  "text": "柯西积分公式 > 定理陈述\n若 f(z) 在简单闭曲线 C 及其内部解析……",
  "char_count": 412,
  "ordinal": 0
}
```

`text` 是送入 embedding 和 prompt 的完整文本，前缀已包含。

### 6.2 检索结果（内存对象）

```python
@dataclass
class RetrievedChunk:
    chunk: Chunk
    vector_rank: int | None
    bm25_rank: int | None
    rrf_score: float
    rerank_score: float | None
```

### 6.3 索引状态（state.json）

```json
{"复变函数与积分变换/01-预备知识/1.1-复数及其表达.md": "<sha256>"}
```

## 7. 代码结构

```
obsidian-rag/
├── pyproject.toml
├── README.md
├── app.py                  # Streamlit
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── chunker.py
│   ├── index.py            # python -m rag.index
│   ├── search.py
│   ├── ask.py              # python -m rag.ask
│   └── eval.py             # python -m rag.eval
├── eval/
│   └── questions.jsonl
├── index/                  # gitignore
├── reports/                # 评测产物，eval.md 与 eval.json 提交
├── tests/
│   └── test_pipeline.py
└── docs/superpowers/specs/2026-11-17-obsidian-rag-design.md
```

## 8. 错误处理与降级

| 场景 | 行为 |
|---|---|
| 无 API key | 仅检索模式，打印片段并提示 |
| DeepSeek 请求失败 | 打印错误，回退仅检索输出，退出码非 0 |
| 索引文件缺失 | 提示运行 `python -m rag.index`，退出 |
| vault 路径不存在 | 明确报错并打印当前 `OBSIDIAN_VAULT` |
| 首次运行需下载模型 | sentence-transformers 自行下载；下载失败时打印原始异常 |
| 单个 md 解析异常 | 跳过该文件并汇总到 stderr，不中断整体索引 |
| 题目 gold_note 不存在 | 评测启动时检查，列出无效条目并退出 |

## 9. 测试

`tests/test_pipeline.py`，pytest，不加载真实模型（用假向量和假分词结果）：

1. 切块：给定包含两级标题和代码块的 Markdown，断言标题路径正确、代码块内 `#` 不切分。
2. 切块：长章节按 800 字符切分且相邻块有重叠。
3. RRF：构造两路排名，断言融合顺序符合公式。
4. 评测指标：给定 top-5 与 gold_note，断言 Recall@5 和 MRR 计算正确。
5. 索引增量：模拟一个文件修改，断言只有该文件的 chunk 被替换。

## 10. 里程碑与验证

1. **索引 + 检索 CLI**
   验证：`python -m rag.search "柯西积分公式"` 的 top-5 包含 `3.3-柯西积分公式.md`。
2. **混合检索 + 精排 + 评测**
   验证：`python -m rag.eval` 生成 `reports/eval.md`，四组配置各有一行指标。
3. **DeepSeek 生成 + 引用**
   验证：`python -m rag.ask "柯西积分公式的条件"` 输出答案，且来源列表非空。
4. **Streamlit UI**
   验证：`streamlit run app.py` 完成一次问答并展示来源。
5. **README 与交付整理**
   验证：README 含架构图、指标表、复现命令；`uv sync && python -m rag.index && python -m rag.eval` 从零可跑通。

## 11. 交付物

- 可运行仓库（含 `pyproject.toml`、`uv.lock`）。
- `reports/eval.md` 消融结果表。
- README：项目动机、架构图、技术选型理由、指标表、复现步骤、局限与后续计划。
- 局限要诚实写明：评测题由作者编写，存在偏差；笔记为个人课程笔记，领域窄；附件未解析。

## 12. 风险

| 风险 | 应对 |
|---|---|
| CPU 建索引慢 | 一次全量后增量；embedding 模型可通过 `EMBEDDING_MODEL` 换成 bge-small-zh 加速 |
| bge-m3 首次下载大 | 首次运行预留 10 分钟；下载中断重跑会续传 |
| 评测题质量影响结论 | 题目覆盖多门课、避免只挑简单题；报告中标注题数 |
| 公式在检索中语义弱 | BM25 补关键词；README 中作为已知局限 |
| DeepSeek 不可用 | 仅检索模式仍可演示与评测 |
