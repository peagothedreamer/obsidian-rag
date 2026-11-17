# 本地 Obsidian 笔记 RAG Demo 设计文档

- 日期：2026-11-17
- 状态：待实现
- 项目位置：仓库根目录 `obsidian-rag`

## 1. 背景与目标

面向简历的 RAG 项目。数据源是作者本机 Obsidian 笔记库中的「电子电路基础」课程笔记，覆盖 27 个章节。

该课程目录原有 478 篇 Markdown，其中 `lec1`–`lec11` 是前 11 章的重复副本，`_源课件`、`_素材` 是源文件和附件目录。索引时排除这些目录，实际索引 346 篇正文笔记。正文已复制到项目 `notes/电子电路基础`，索引针对这份本地副本；vault 只作只读来源。内容为中文教材笔记，含 LaTeX 公式。

目标：

1. 对笔记库做可引用的问答：答案给出 `[n]` 引用，可追溯到具体笔记和标题。
2. 检索质量可量化：自建评测集，对 4 种检索配置做消融，产出可写进简历的指标。

成功标准：

- `python -m rag.eval` 一条命令产出 `reports/eval/base.md`，含 Recall@5、MRR@10 的四行对比表。
- 典型问题（如「理想运放为什么有虚短虚断」）的 top-5 检索结果命中对应笔记。
- Streamlit 页面可完成一次带引用的问答，并能看到来源片段。

## 2. 非目标

- 只索引「电子电路基础」目录，排除 `lec*`、`_源课件`、`_素材`；vault 内其他课程不索引。
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
| 精排 | `BAAI/bge-reranker-base` | 278M，CPU 上对 20–40 个候选精排约 4–11 秒 |
| 生成 | DeepSeek `deepseek-chat` | OpenAI 兼容接口，价格低，中文好 |
| 界面 | Streamlit 单页 | 100 行内完成聊天和来源展示 |
| 存储 | jsonl + numpy | 约 1200 个 chunk，全量点积毫秒级，不需要数据库 |

依赖带来的成本：首次运行下载 bge-m3（约 2.2GB）和 reranker（约 1.1GB）；346 篇笔记全量建索引预计 3–5 分钟（CPU），之后按文件变更增量更新。

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
        └─ RRF(k=60) → 两路并集（≤40）
             └─ bge-reranker-base → top-5
                  ├─ DeepSeek 生成答案（带 [n] 引用）
                  └─ Streamlit 展示来源片段
```

## 5. 模块设计

### 5.1 配置（`rag/config.py`）

环境变量，均有默认值：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OBSIDIAN_VAULT` | `notes`（项目内副本） | 笔记根目录 |
| `COURSE_DIRS` | `电子电路基础` | 限定索引范围；目录不存在时报错 |
| `EXCLUDE_DIR_PREFIXES` | `("lec", "_")` | 跳过目录名以这些前缀开头的重复副本和源文件目录 |
| `DEEPSEEK_API_KEY` | 无 | 缺失时进入仅检索模式 |
| `DEEPSEEK_MODEL` | `deepseek-chat` | 生成模型 |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | 便于消融时替换 |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | 精排模型 |

模型名集中在此处，代码其他部分不写死。

### 5.2 切块（`rag/chunker.py`）

- 只遍历 `COURSE_DIRS` 下所有 `.md`，跳过 `.obsidian/`，并跳过目录名以 `EXCLUDE_DIR_PREFIXES` 开头的子目录。
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

1. 扫描 `COURSE_DIRS`（应用排除规则），计算每个 md 文件内容 sha256。
2. 与 state.json 对比，找出新增、修改、删除的笔记。
3. 删除笔记的所有 chunk；新增和修改笔记重新切块、重新 embedding。
4. 其他 chunk 的向量从旧 npy 按索引拼接保留。
5. 重写三个产物。BM25 语料不落盘，检索时从 chunks.jsonl 重建（几千行，毫秒级）。

`--rebuild` 参数：忽略 state，全量重建。

模型编码使用 `sentence-transformers`，batch size 32，CPU。

### 5.4 检索（`rag/search.py`）

接口：`search(query: str, top_k: int = 5, no_rerank: bool = False, recall_k: int | None = None, rerank_n: int | None = None) -> list[RetrievedChunk]`

1. 向量召回：query 用同一 bge-m3 编码，与 embeddings.npy 点积，取 top-20。
2. BM25 召回：jieba 精确模式分词，过滤空白和纯标点 token，取 top-20。
3. RRF 融合：`score = Σ 1/(60 + rank)`，对两路召回的全部候选排序（最多 `2 × recall_k`）。只出现在一路的 chunk 仅累加该路得分。
4. 精排：reranker 对两路召回并集（最多 `2 × recall_k`，默认 40 个）打分，取 top_k。RRF 只用于排序，不用于截断候选池，否则单路命中的正确结果会在精排前被丢弃。
5. 返回 chunk 文本、note_path、heading_path、各阶段分数。

`search.py` 同时提供 CLI（`python -m rag.search "问题"`），打印 top-5 片段和分数。评测时以 `top_k=10` 调用，供 MRR@10 使用。

### 5.5 生成（`rag/ask.py`）

命令：`python -m rag.ask "问题"`。

- 调用 `search()` 取 top-5，拼 prompt：

```
你是笔记库问答助手。只根据下面编号片段回答；片段不足时直接说明。
引用事实时在句末标注片段编号，如 [1]。
每条答案至少引用一个片段，禁止引用未给出的编号。

[1] 电子电路基础/04-电子电路 · 运算放大器/1.3-理想运放与虚短虚断.md > 理想运放与虚短虚断
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
{"id": "q01", "question": "理想运放为什么有虚短和虚断？", "gold_note": "电子电路基础/04-电子电路 · 运算放大器/1.3-理想运放与虚短虚断.md", "gold_heading": "理想运放与虚短虚断"}
```

- `gold_heading` 可选，v1 指标只按 note 层面计算。
- 规模：30–50 题，全部来自电子电路基础，覆盖至少 4 个章节（建议：电路定律、运放、二极管、三极管/MOSFET、频率响应/反馈）。
- 另建 10–15 条**不包含笔记标题原词**的改写题，写入 `eval/questions.paraphrase.jsonl`，单独评测，用于验证语义召回；基础集关键词重合度高，改写集才能拉开各检索配置的差距。
- 草稿生成：`python -m rag.eval --draft 20` 从 vault 随机抽样标题，按「<笔记名> 中的 <标题> 讲了什么？」模板写入 `eval/questions.draft.jsonl`。人工筛选、改写后合入正式文件，草稿本身不参与评测。
- vector 和 bm25 单路配置直接调用对应召回函数，不经过融合。

四种配置，每组跑全部题目：

| 配置 | 召回 |
|---|---|
| vector | 余弦 top-10 |
| bm25 | BM25 top-10 |
| hybrid | RRF top-10，不精排 |
| hybrid+rerank | 两路召回并集（≤40）→ 精排 top-10 |

指标（按笔记去重计算）：

- **Recall@5**：每组配置返回结果的前 5 条涉及的笔记集合包含 `gold_note` 的题目占比。
- **MRR@10**：取 top-10 中去重后第一次出现 `gold_note` 的排名 r，得分 1/r；未命中记 0；对所有题取平均。
- **引用命中率**（需要 API）：在答案至少含一个有效 `[n]` 引用的题目中，被引用片段至少一个来自 `gold_note` 的比例。仅对 hybrid+rerank 配置统计。

小样本提醒：12 题时 1 题约等于 8.3 个百分点，46 题时约 2.2 个百分点。报告和简历必须注明题目数、题目构造方式，不要把小差距写成精确的百分比提升。

输出：

- `reports/eval/base.md`：四行配置 × 三列指标的表，附题目数和运行时间。
- `reports/eval/base.json`：逐题明细，便于排查失败案例。

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
  "note_path": "电子电路基础/04-电子电路 · 运算放大器/1.3-理想运放与虚短虚断.md",
  "note_title": "1.3-理想运放与虚短虚断",
  "heading_path": "理想运放与虚短虚断 > 虚短",
  "text": "理想运放与虚短虚断 > 虚短\n由于开环增益趋于无穷，两输入端电压差近似为零……",
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
{"电子电路基础/01-电路基本概念与基本定律/1.10-基尔霍夫定律.md": "<sha256>"}
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
├── notes/                  # gitignore，课程笔记本地副本
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
| notes 路径不存在 | 明确报错并打印当前 `OBSIDIAN_VAULT` |
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
   验证：`python -m rag.search "理想运放虚短虚断"` 的 top-5 包含 `1.3-理想运放与虚短虚断.md`。
2. **混合检索 + 精排 + 评测**
   验证：`python -m rag.eval` 生成 `reports/eval/base.md`，四组配置各有一行指标。
3. **DeepSeek 生成 + 引用**
   验证：`python -m rag.ask "理想运放为什么有虚短虚断"` 输出答案，且来源列表非空。
4. **Streamlit UI**
   验证：`streamlit run app.py` 完成一次问答并展示来源。
5. **README 与交付整理**
   验证：README 含架构图、指标表、复现命令；`uv sync && python -m rag.index && python -m rag.eval` 从零可跑通。

## 11. 交付物

- 可运行仓库（含 `pyproject.toml`、`uv.lock`）。
- `reports/eval/base.md` 消融结果表。
- README：项目动机、架构图、技术选型理由、指标表、复现步骤、局限与后续计划。
- 局限要诚实写明：评测题由作者编写，存在偏差；只覆盖电子电路基础一门课，领域窄；附件未解析。

## 12. 风险

| 风险 | 应对 |
|---|---|
| CPU 建索引慢 | 一次全量后增量；embedding 模型可通过 `EMBEDDING_MODEL` 换成 bge-small-zh 加速 |
| bge-m3 首次下载大 | 首次运行预留 10 分钟；下载中断重跑会续传 |
| 评测题质量影响结论 | 题目覆盖多门课、避免只挑简单题；报告中标注题数 |
| 公式在检索中语义弱 | BM25 补关键词；README 中作为已知局限 |
| DeepSeek 不可用 | 仅检索模式仍可演示与评测 |

## 13. 后续实验：HyDE 复现协议

- 状态：预注册 v1（2026-11-17）
- 论文：*Precise Zero-Shot Dense Retrieval without Relevance Labels*（Gao et al., ACL 2023）
- 脚本：`hyde/dense.py`（纯稠密 HyDE）、`hyde/rerank.py`（HyDE + 混合检索 + 精排）
- 目的：验证零样本「假设文档检索」能否改善中文课程笔记上的改写题召回，并把实验过程固定在可复现的协议里。

### 13.1 研究问题与假设

RQ1：先用 LLM 生成一段假设答案，再用它的向量检索，能否提升不含标题原词问题的 Recall@5？

- H1：在改写集上，`hyde` 的 Recall@5 高于 `query` 基线，提升集中在此前向量完全召不回的题目上。
- H2：在基础集上，`hyde` 与 `query` 基本持平或略降，因为原词已经足以命中。
- H3：`query+hyde` 在两类题上更稳；`multi-hyde` 在改写集上不低于单次 `hyde`，但生成成本更高。
- H4：收益不均匀，失败主要来自假设文档跑题或编造内容，把检索带向错误笔记。

### 13.2 数据与使用规则

- 开发集：`eval/questions.jsonl`（46 题），用于调试 prompt 和参数。
- 测试集：`eval/questions.paraphrase.jsonl`（12 题），用于报告最终结果。
- prompt 只在开发集上调整；一旦冻结，不在测试集上继续调参。
- 正式结论前把改写集扩充到 30 条以上；12 条结果只能作为 pilot，1 题约等于 8.3 个百分点。

### 13.3 方法

所有模式共用同一份索引、同一个 bge-m3 编码器和同一套切块，只改变查询表示：

| 模式 | 检索向量 |
|---|---|
| `query` | `encode(question)` |
| `hyde` | `encode(hypothesis[0])` |
| `query+hyde` | `encode(question + "\n\n" + hypothesis[0])` |
| `multi-hyde` | 归一化后的 `mean(encode(hypothesis[i]))` |

生成设置：

- 模型：`deepseek-chat`
- temperature：0.7
- 每题生成 3 段假设答案（可调）
- prompt：中文，要求 150–250 字、像教材段落、不出现「片段」或引用编号
- 缓存：按 `prompt 版本 + 模型 + 问题 + 样本序号` 的 SHA1 落盘，重跑不重复计费

检索设置：

- 稠密检索取 top-10，计算 Recall@5 与 MRR@10
- 主实验不接 BM25、不接 reranker，隔离 HyDE 变量
- 精排与混合检索作为后续补充实验，不混入主结论

### 13.4 控制变量

- 同一题集、同一 chunk、同一 embedding 模型、同一 top-k
- 同一 DeepSeek 模型与 prompt 版本
- 所有模式使用同一批假设文档（`hyde` 用第 1 段，`multi-hyde` 用全部）
- 失败题目不删除、不替换；排除规则在 13.7 预先声明

### 13.5 指标与统计

- **Recall@5**：top-5 按笔记去重后是否包含 gold_note
- **MRR@10**：金标笔记首次出现排名的倒数
- **逐题 ΔMRR**：每个模式相对 `query` 的差值
- **W/L/T**：逐题胜/负/平数量
- **95% bootstrap CI**：对逐题 ΔMRR 重采样 2000 次
- **成本**：API 调用次数、缓存命中、输入/输出 token；按运行时的实际价格换算
- **延迟**：生成耗时与检索耗时分开记录

12 题样本下，置信区间会很宽。结论以方向和逐题胜负为主，不把 1–2 题的波动写成显著提升。

### 13.6 实验流程

1. 预注册本协议（本节）。
2. 在基础集上调试 prompt 和 samples，记录改动。
3. 冻结 prompt 版本号。
4. 在改写集上跑一次正式实验，输出 JSON 与 Markdown。
5. 失败分析：逐题看假设文档、检索结果和金标笔记。
6. 把结果写入 README 的「实验」小节和 `docs/improvements.md`。

### 13.7 失效与排除规则

- 假设文档为空：视为该次生成失败，重试一次；仍失败则记录并终止该题。
- API 错误：指数退避重试，最多 3 次；记录重试次数。
- 缓存命中：计入 token 统计，但不重复调用 API。
- 不因结果不好而删除题目；不针对测试集改 prompt。
- 不手动挑选「最好看」的假设文档。

### 13.8 与论文的差异

必须在报告中写明：

- 论文用 Contriever，本文用 bge-m3
- 论文是英文通用语料（web search / QA / fact verification），本文是中文电子电路课程笔记
- 论文使用 InstructGPT，本文使用 DeepSeek
- 论文评估 nDCG/Recall 等，本文主指标为 Recall@5 与 MRR@10
- 本文每题 3 段假设文档，论文的设置不同

### 13.9 结果判读规则

- H1 成立的条件：改写集上 `hyde` 的平均 ΔMRR > 0，且逐题胜多于负；CI 不跨 0 作为辅助证据。
- 若改写集无提升：作为负结果报告，分析是生成内容跑题、embedding 不适配，还是题目本身仍有关键词重合。
- 基础集下降不构成失败，本身就是 H2 的预期。
- 不把 `multi-hyde` 的优势归因于方法本身，除非同时报告生成成本。

### 13.10 失败分类

1. 关键词漂移：假设文档用了相近但不同的术语
2. 实体/公式错误：生成的公式或元件名称与笔记不符
3. 过于泛化：内容正确但太笼统，命中多篇相似笔记
4. 跑题：假设文档回答了另一个问题
5. 检索仍失败：假设文档没问题，但 embedding 仍排不到前 5

### 13.11 交付物

- `reports/hyde/dense_<题集>.json`：逐题明细与汇总
- 同名 `.md`：对比表、W/L/T、成本
- README「实验：HyDE」小节
- 失败案例 3–5 个，带假设文档原文和检索结果

### 13.12 复现命令

```bash
# 冒烟
uv run python -m hyde.dense --limit 3 --samples 1 \
  --out reports/hyde/dense_smoke.json

# 开发集
uv run python -m hyde.dense --questions eval/questions.jsonl --samples 3 \
  --out reports/hyde/dense_base.json

# 测试集
uv run python -m hyde.dense --questions eval/questions.paraphrase.jsonl \
  --samples 3 --out reports/hyde/dense_paraphrase.json

# HyDE 加入混合检索候选池，再做精排
uv run python -m hyde.rerank --questions eval/questions.paraphrase.jsonl \
  --out reports/hyde/rerank_paraphrase.json
```

### 13.13 效度威胁

- 测试集仅 12 题，且由作者编写，存在选题偏差
- 单门课、单一 embedding 模型，结论不能外推到通用检索
- API 生成存在随机性，缓存只保证可复现，不消除采样差异
- gold_note 是笔记级标注，同一知识点的其他笔记会被算作错误

### 13.14 实验结果（2026-11-17）

基础集 46 题（纯稠密检索）：

| 模式 | Recall@5 | MRR@10 | ΔMRR | 胜/负/平 |
|---|---|---|---|---|
| query | 0.978 | 0.905 | - | - |
| hyde | 0.978 | 0.866 | -0.039 | 4/6/36 |
| query+hyde | 1.000 | 0.900 | -0.005 | 5/6/35 |
| multi-hyde | 0.978 | 0.879 | -0.026 | 5/5/36 |

改写集 78 题（纯稠密检索）：

| 模式 | Recall@5 | MRR@10 | ΔMRR | 95% CI | 胜/负/平 |
|---|---|---|---|---|---|
| query | 0.705 | 0.569 | - | - | - |
| hyde | 0.846 | 0.646 | +0.077 | [-0.000, +0.160] | 22/11/45 |
| query+hyde | 0.795 | 0.649 | +0.080 | [+0.020, +0.148] | 17/8/53 |
| multi-hyde | 0.808 | 0.661 | +0.092 | [+0.011, +0.177] | 24/11/43 |

结论：

- H1 成立：改写集上 HyDE 明显提升召回，`query+hyde` 与 `multi-hyde` 的 95% CI 不跨 0。
- H2 成立：基础集上 HyDE 略降，关键词重合度高时假设文档引入噪声。
- H3 部分成立：`multi-hyde` 的 MRR 最高，但生成成本是单次 HyDE 的 3 倍。
- HyDE 的稠密 Recall@5（0.846）接近 hybrid+rerank（0.872），但 MRR 更低；两者互补，下一步应把 HyDE 向量接入混合检索的候选池，再做精排。

成本：78 题 × 3 样本共 234 次 API 调用，输入 17,136 tokens，输出 42,614 tokens（2026-11-17 价格下成本很低）。

原始报告：`reports/hyde/dense_base.json`、`reports/hyde/dense_paraphrase.json`。

#### 补充实验：HyDE 加入混合检索候选池（78 题）

| 模式 | Recall@5 | MRR@10 |
|---|---|---|
| hybrid+rerank | 0.872 | 0.683 |
| hyde-aug-hybrid+rerank | 0.872 | 0.692 |
| hyde-aug-hybrid（不精排） | 0.769 | 0.636 |

- 加入精排后 HyDE 基本中性：1 胜 1 负 76 平，ΔMRR +0.009，95% CI `[-0.001, +0.029]` 跨 0。
- 不精排时 HyDE 把混合检索从 0.679 / 0.580 提升到 0.769 / 0.636，说明收益集中在第一阶段。
- 当前精排候选池已经足够大，精排弥补了召回缺口；HyDE 的额外生成成本不划算。
- 待验证：限制精排候选数（如 top-20）时，HyDE 是否重新产生收益。

报告：`reports/hyde/rerank_paraphrase.json`。
