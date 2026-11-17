# HyDE 向量加入混合检索的实验结果

- 题集：`eval\questions.paraphrase.jsonl`
- 题数：78，召回 top-20，精排 top-10
- embedding：`BAAI/bge-m3`，rerank：`BAAI/bge-reranker-base`
- 假设文档：API 调用 0 次，缓存命中 78 次

| 模式 | Recall@5 | MRR@10 | 题目数 |
|---|---|---|---|
| hybrid+rerank | 0.872 | 0.683 | 78 |
| hyde-aug-hybrid+rerank | 0.872 | 0.692 | 78 |
| hyde-aug-hybrid | 0.769 | 0.636 | 78 |

`hyde-aug-hybrid` 不经过精排，用于区分「融合收益」和「精排收益」。
