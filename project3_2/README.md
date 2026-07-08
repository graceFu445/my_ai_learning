# GraphRAG 多跳问答系统

这个项目实现一个融合文档检索、图谱推理与 LLM 生成的企业关系问答 Demo。输入采用样例工程同款方式：仓库内置一份 `data/company.txt`，系统从文本中切分文档、抽取关系、构建检索索引，并回答 “A集团的最大股东是谁？” 这类多跳问题。

## 技术选型

- 文档语义检索：默认使用通义千问 `text-embedding-v3`，单元测试使用本地替身避免访问网络
- 关键词检索：`jieba` 分词 + BM25 思路，本项目内置轻量 BM25 实现
- 图谱推理：Neo4j，测试环境使用 `InMemoryGraphStore`
- 关系抽取：通义千问 LLM 可接入，Demo 默认使用规则抽取保证可离线验证
- 联合评分：

```text
hybrid_score = 0.50 * vector_score + 0.30 * bm25_score + 0.20 * graph_score
```

## 项目结构

```text
project3_2/
├── data/company.txt              # 示例输入文本，按段落提供企业、股权和管理层信息
├── scripts/init_neo4j.py         # 从 company.txt 抽取关系并初始化 Neo4j 图谱
├── src/config.py                 # 读取 .env 和运行配置
├── src/document_loader.py        # 文档加载与段落切分
├── src/embedding.py              # 通义向量和本地测试向量客户端
├── src/bm25_retriever.py         # BM25 关键词检索
├── src/graph_store.py            # 内存图谱和 Neo4j 图谱访问层
├── src/relation_extractor.py     # 从文本抽取企业关系
├── src/scorer.py                 # 向量、BM25、图谱证据的联合评分
├── src/generator.py              # 通义千问最终答案生成
├── src/qa.py                     # 混合检索问答主流程和 CLI 入口
└── tests/                        # 单元测试和回归测试
```

## 问答链路

每个问题都会先形成 evidence bundle，再决定答案：

```text
用户问题
  -> 向量检索
  -> BM25 关键词检索
  -> Neo4j/内存图谱检索
  -> 联合评分
  -> 合并图谱关系和文档证据
  -> 通义千问生成答案
```

图谱检索只按问题中出现的实体统一取入边、出边和实体间路径，不按固定问法写分支；最终答案统一由通义千问基于证据生成。

## 运行测试

```bash
conda env create -f environment.yml
conda activate project3_2
python -m unittest discover tests
```

## 运行问答

运行问答前需要在 `.env` 中配置 `DASHSCOPE_API_KEY`，因为向量检索和最终答案生成都默认使用通义千问。

```bash
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY
conda activate project3_2
python -m src.qa "A集团的最大股东是谁？" --show-trace
python -m src.qa "D科技控股了哪些公司？" --show-trace
python -m src.qa "D科技旗下有哪些公司？" --show-trace
python -m src.qa "从B资本到F智能有哪些控股路径？" --show-trace
python -m src.qa "A集团的注册地址在哪里？" --show-trace
```

## Neo4j 初始化

先确保 Neo4j 在本地运行，然后执行：

```bash
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="password"
conda activate project3_2
python scripts/init_neo4j.py
python -m src.qa "A集团的最大股东是谁？" --use-neo4j --show-trace
```

## 通义千问配置

本项目默认调用通义千问，不再提供额外的大模型启用开关：

```bash
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY
conda activate project3_2
python -m src.qa "A集团的最大股东是谁？" --show-trace
```

## 测试样例

- `A集团的最大股东是谁？` 预期包含 `B资本`、`60%`
- `C基金持有A集团多少股份？` 预期包含 `25%`
- `A集团直接控股哪些公司？` 预期包含 `D科技`、`E实业`
- `A集团通过D科技间接控股哪些公司？` 预期包含 `F智能`、`G能源`
- `从B资本到F智能有哪些控股路径？` 预期包含 `B资本 -> A集团 -> D科技 -> F智能`
- `E实业是否控股K物流？` 预期回答不控股，因为它是参股关系
- `A集团的注册地址在哪里？` 预期拒答，不编造地址
