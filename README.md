# GraphWalks DeepSeek V4 预处理

这个目录用于准备 OpenAI GraphWalks 数据集，目标是后续测试
DeepSeek V4 Pro / Flash 在 GraphWalks 上的表现。

目前已经实现的部分是：读取 GraphWalks 的 parquet 数据，用 DeepSeek V4
tokenizer 统计每条 `prompt` 的输入 token 数，然后写出带 token 统计列的新
parquet 文件。

## 本地文件

数据集、tokenizer 和编译产物都不进 git。它们是下载或生成出来的本地工件。

当前期望目录结构：

```text
dataset/
  graphwalks_128k_and_shorter.parquet
  graphwalks_256k_to_1mil.parquet
  graphwalks_128k_and_shorter.deepseek_v4_tokens.parquet
  graphwalks_256k_to_1mil.deepseek_v4_tokens.parquet
  graphwalks_256k.deepseek_v4_tokens.parquet          # 由 256k_to_1mil 拆分，token <= 1M
  graphwalks_1mil.deepseek_v4_tokens.parquet          # 由 256k_to_1mil 拆分，token > 1M
tokenizer/
  deepseek-v4-pro/
    tokenizer.json
    tokenizer_config.json
```

下载时 `deepseek-ai/DeepSeek-V4-Pro` 仓库里没有
`special_tokens_map.json`。当前 token 统计程序只依赖 `tokenizer.json`，
所以不受影响。

## 下载数据集

```powershell
New-Item -ItemType Directory -Force -Path dataset

Invoke-WebRequest `
  -Uri https://huggingface.co/datasets/openai/graphwalks/resolve/main/graphwalks_128k_and_shorter.parquet `
  -OutFile dataset\graphwalks_128k_and_shorter.parquet

Invoke-WebRequest `
  -Uri https://huggingface.co/datasets/openai/graphwalks/resolve/main/graphwalks_256k_to_1mil.parquet `
  -OutFile dataset\graphwalks_256k_to_1mil.parquet
```

## 下载 DeepSeek V4 tokenizer

```powershell
New-Item -ItemType Directory -Force -Path tokenizer\deepseek-v4-pro

Invoke-WebRequest `
  -Uri https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/tokenizer.json `
  -OutFile tokenizer\deepseek-v4-pro\tokenizer.json

Invoke-WebRequest `
  -Uri https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/tokenizer_config.json `
  -OutFile tokenizer\deepseek-v4-pro\tokenizer_config.json
```

## 统计 token 数

```powershell
cargo run --release -- --batch-size 16 `
  dataset\graphwalks_128k_and_shorter.parquet `
  dataset\graphwalks_256k_to_1mil.parquet
```

输出文件：

```text
dataset/graphwalks_128k_and_shorter.deepseek_v4_tokens.parquet
dataset/graphwalks_256k_to_1mil.deepseek_v4_tokens.parquet
```

新增列：

```text
deepseek_v4_input_tokens: Int32
deepseek_v4_over_1m: Boolean
```

当前统计方式是：

```rust
tokenizer.encode(prompt, false)
```

也就是说，只统计原始 `prompt` 文本本身，不包含 special tokens，也不包含
Chat API 里 role / chat template 包装带来的额外 token。

## 拆分 256k_to_1mil 数据集

`graphwalks_256k_to_1mil` 内部实际上是两组长度离散的样本：
一组在 ~257k tokens，另一组在 ~1031k tokens（略超 1M）。
用 `split_by_token_count` 按 1,000,000 为阈值把它们拆开：

```powershell
cargo run --release --bin split_by_token_count -- `
  --input  dataset\graphwalks_256k_to_1mil.deepseek_v4_tokens.parquet `
  --output-low  dataset\graphwalks_256k.deepseek_v4_tokens.parquet `
  --output-high dataset\graphwalks_1mil.deepseek_v4_tokens.parquet
```

输出：

```text
完成。threshold=1000000
  <= threshold: 200 行 → dataset/graphwalks_256k.deepseek_v4_tokens.parquet
  >  threshold: 200 行 → dataset/graphwalks_1mil.deepseek_v4_tokens.parquet
```

## token 分布

运行：

```powershell
cargo run --release --bin token_stats -- `
  dataset\graphwalks_128k_and_shorter.deepseek_v4_tokens.parquet `
  dataset\graphwalks_256k_to_1mil.deepseek_v4_tokens.parquet
```

当前按 DeepSeek V4 tokenizer 得到的分布，按数据文件和 `problem_type` 分别展示。

### 全量汇总（两个文件合计）

行数: 1150 | 最小值: 1050 | 最大值: 1032668 | 均值: 246579.3 | >1M: 200

| Token 区间 | 合计 | bfs | parents |
|---|---:|---:|---:|
| <=8k | 250 (21.74%) | 100 (18.18%) | 150 (25.00%) |
| 8k-16k | 100 (8.70%) | 50 (9.09%) | 50 (8.33%) |
| 16k-32k | 100 (8.70%) | 50 (9.09%) | 50 (8.33%) |
| 32k-64k | 100 (8.70%) | 50 (9.09%) | 50 (8.33%) |
| 64k-128k | 100 (8.70%) | 50 (9.09%) | 50 (8.33%) |
| 128k-256k | 100 (8.70%) | 50 (9.09%) | 50 (8.33%) |
| 256k-512k | 200 (17.39%) | 100 (18.18%) | 100 (16.67%) |
| 512k-1M | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| >1M | 200 (17.39%) | 100 (18.18%) | 100 (16.67%) |
| **合计** | **1150** | **550** | **600** |

### graphwalks_128k_and_shorter（750 行）

行数: 750 | 最小值: 1050 | 最大值: 129381 | 均值: 34239.8 | >1M: 0

| Token 区间 | 合计 | bfs | parents |
|---|---:|---:|---:|
| <=8k | 250 (33.33%) | 100 (28.57%) | 150 (37.50%) |
| 8k-16k | 100 (13.33%) | 50 (14.29%) | 50 (12.50%) |
| 16k-32k | 100 (13.33%) | 50 (14.29%) | 50 (12.50%) |
| 32k-64k | 100 (13.33%) | 50 (14.29%) | 50 (12.50%) |
| 64k-128k | 100 (13.33%) | 50 (14.29%) | 50 (12.50%) |
| 128k-256k | 100 (13.33%) | 50 (14.29%) | 50 (12.50%) |
| 256k-512k | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| 512k-1M | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| >1M | 0 (0.00%) | 0 (0.00%) | 0 (0.00%) |
| **合计** | **750** | **350** | **400** |

bfs 均值: 36585.7 | parents 均值: 32187.2

### graphwalks_256k_to_1mil（400 行，拆分前）

行数: 400 | 最小值: 257107 | 最大值: 1032668 | 均值: 644715.9 | >1M: 200

内部含两组离散长度，已用 `split_by_token_count` 按 1,000,000 拆开，见下方两表。

### graphwalks_256k（200 行）

行数: 200 | 最小值: 257107 | 最大值: 258176 | 均值: 257659.4 | >1M: 0

| Token 区间 | 合计 | bfs | parents |
|---|---:|---:|---:|
| 256k-512k | 200 (100.00%) | 100 (100.00%) | 100 (100.00%) |
| **合计** | **200** | **100** | **100** |

bfs 均值: 257642.2 | parents 均值: 257676.7

### graphwalks_1mil（200 行）

行数: 200 | 最小值: 1030800 | 最大值: 1032668 | 均值: 1031772.4 | >1M: 200

| Token 区间 | 合计 | bfs | parents |
|---|---:|---:|---:|
| >1M | 200 (100.00%) | 100 (100.00%) | 100 (100.00%) |
| **合计** | **200** | **100** | **100** |

bfs 均值: 1031784.0 | parents 均值: 1031760.9

## 程序

默认 binary：

```text
count_deepseek_v4_tokens
```

作用：读取一个或多个 parquet 文件，默认统计 `prompt` 列，并写出同目录的
`.deepseek_v4_tokens.parquet` 文件。

统计 binary：

```text
token_stats
```

作用：读取已经生成的 token parquet 文件，输出 token 分段统计表。
支持按 `problem_type` 列（如 `bfs` / `parents`）自动分组输出，同时输出每个输入文件的独立统计。

拆分 binary：

```text
split_by_token_count
```

作用：按 `deepseek_v4_input_tokens` 阈值（默认 1,000,000）把一个 parquet 文件
拆成两个，保留所有原始列。用于把 `graphwalks_256k_to_1mil` 拆成
`graphwalks_256k`（~257k tokens）和 `graphwalks_1mil`（~1031k tokens）。
