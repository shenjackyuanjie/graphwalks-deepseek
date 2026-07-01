# OpenPangu 2.0 Flash Tokenizer

本目录包含 OpenPangu 2.0 Flash 的 tokenizer 配置和 Transformers 自定义加载代码。
仓库的 Rust token 统计程序可以直接加载 `tokenizer.json`。

## 目录内文件

| 文件 | 用途 |
|---|---|
| `tokenizer.json` | Hugging Face `tokenizers` 的完整 BPE 定义，Rust 代码只需要该文件 |
| `tokenizer_config.json` | Transformers 特殊 token、chat template 和自定义类配置 |
| `tokenization_openpangu_v2.py` | `OpenPanguV2Tokenizer` 自定义 Transformers tokenizer |

## Tokenizer 结构

- 模型：BPE。
- 基础词表：148,899 tokens。
- Added tokens：701。
- 总词表：149,600 tokens，ID 范围为 `0..149599`。
- BPE merges：148,643。
- Normalizer：无，输入文本不会先做 Unicode 归一化。
- Pre-tokenizer：自定义正则 `Split` 后接 `ByteLevel`。
- Decoder：`ByteLevel`。
- `unk_token`：无。
- `byte_fallback`：`tokenizer.json` 中为 `false`；输入通过 ByteLevel 预分词转为字节表示。
- JSON 内置 post-processor：`ByteLevel`，不会自动添加 BOS/EOS。

预分词正则会单独处理英文缩写、汉字前的空格、中英文标点、数字、换行和其他空白。
因此，OpenPangu 和 V4P 的差异不只是词表大小，还包括预分词边界和 BPE merge 学习结果的差异。

### 主要特殊 token

| Token | ID | `special` |
|---|---:|---:|
| `<\|pangu_text_start\|>` | 148899 | true |
| `<\|pangu_text_end\|>` | 148900 | true |
| `<\|message_start\|>` | 148901 | true |
| `<\|message_end\|>` | 148902 | true |
| `<\|tool_call_start\|>` | 148903 | false |
| `<\|tool_call_end\|>` | 148904 | false |
| `<think>` | 148905 | false |
| `</think>` | 148906 | false |
| `<\|vision_start\|>` | 148907 | true |
| `<\|vision_end\|>` | 148908 | true |
| `<\|image_pad\|>` | 148909 | true |
| `<\|video_pad\|>` | 148910 | true |


`special=false` 的工具和 thinking 标记仍然是 added token，可以作为一个 token 匹配；
但它们不会被 `decode(..., skip_special_tokens=true)` 当作 special token 删除。

`pad_token` 和 `eos_token` 都是 `<|pangu_text_end|>`（ID 148900）。
Python 自定义类设置了左侧 padding。`model_max_length` 是一个极大的占位值，
不代表模型真实支持这个上下文长度。

## 三种加载路径的语义

### 1. Rust 直接加载 `tokenizer.json`

```rust
use tokenizers::Tokenizer;

let tokenizer = Tokenizer::from_file(
    "tokenizer/openpangu-2-0-flash/tokenizer.json",
)?;
let encoding = tokenizer.encode(text, false)?;
```

当前仓库使用 `tokenizers 0.21.4`，已验证可直接加载该 JSON。
这条路径不会读取 `tokenizer_config.json`，也不会执行 Python 自定义类或 Jinja chat template。

JSON 的 post-processor 不含 BOS/EOS TemplateProcessing，因此对裸 `Tokenizer` 而言，
`encode(text, true)` 和 `encode(text, false)` 都不会自动添加 BOS。

### 2. Transformers 自定义 tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "tokenizer/openpangu-2-0-flash",
    trust_remote_code=True,
)
```

`OpenPanguV2Tokenizer` 的默认值是：

```text
add_bos_token = true
add_eos_token = false
```

加载时它会用 `TemplateProcessing` 覆盖 JSON 中的 post-processor。所以：

- `tokenizer(text, add_special_tokens=False)` 与 Rust `encode(text, false)` 的 token ID 一致。
- `tokenizer(text, add_special_tokens=True)` 会在开头增加 ID 148899 的 BOS。
- 默认不在末尾增加 EOS。

### 3. Chat template

`tokenizer_config.json` 内含完整 Jinja chat template。默认行为包括：

- 文本以 `<|pangu_text_start|><|message_start|>system\n` 开始。
- 即使没有 system message，也会生成空的 system 轮次。
- `thinking=true`。
- `context_thinking="Interleave"`。
- `add_generation_prompt=true`。
- 生成 assistant 轮次时，thinking 开启则以 `<think>` 结尾，关闭则以 `</think>` 结尾。

Chat template 已经把 BOS 作为字面 token 写入结果。
`apply_chat_template(..., tokenize=True)` 会得到一个 BOS；不要把渲染后的文本再用
`add_special_tokens=True` 编码，否则会得到两个 BOS。

Rust `tokenizers` 不负责渲染 Jinja template。如果需要统计真实 Chat API 输入，
应先渲染 chat template，再用 `encode(rendered_text, false)` 编码。

## GraphWalks token 统计

仓库的统计命令：

```powershell
cargo run --release --bin count_tokens -- `
  --tokenizer-json tokenizer\openpangu-2-0-flash\tokenizer.json `
  --output-tag openpangu_2_0_flash `
  --batch-size 8 `
  dataset\graphwalks_128k_and_shorter.parquet `
  dataset\graphwalks_256k_to_1mil.parquet
```

输出列：

```text
openpangu_2_0_flash_input_tokens: Int32
openpangu_2_0_flash_over_1m: Boolean
```

统计方式是 `tokenizer.encode(prompt, false)`：只统计原始 `prompt`，不含 BOS、EOS、role 标记或 chat template。
这与仓库中 V4P 统计的口径一致。如果只对普通文本使用 Transformers 默认 BOS，
OpenPangu 的每条计数会在本文数值上再增加 1。

## OpenPangu 与 V4P 对比

对两个 GraphWalks 输出文件按原始行顺序对齐，共 1,150 条。
两侧行数和 `problem_type` 顺序一致，且都不包含 special tokens。

### 总体结果

| 指标 | V4P | OpenPangu | 差值 |
|---|---:|---:|---:|
| 总 token 数 | 283,566,252 | 392,641,706 | +109,075,454 |
| 平均每条 | 246,579 | 341,428 | +94,848 |
| 总量倍率 | 1.000x | 1.38466x | +38.47% |

- 1,150 条中，OpenPangu token count 全部大于 V4P。
- 完全相同：0 条。
- OpenPangu 小于 V4P：0 条。
- 先计算每条增幅再平均：+35.87%。
- 每条增幅中位数：+38.13%。

“总量增幅”和“每条增幅的平均”不同，是因为短文本的倍率更低，
而总量口径中长文本占更大权重。

### 按数据集

| 数据集 | 条数 | V4P 总量 | OpenPangu 总量 | 总量增幅 | 平均单条差值 | 单条增幅中位数 |
|---|---:|---:|---:|---:|---:|---:|
| 128k 及以下 | 750 | 25,679,876 | 35,425,970 | +37.95% | +12,995 | +37.12% |
| 256k–1M | 400 | 257,886,376 | 357,215,736 | +38.52% | +248,323 | +38.53% |
| 全部 | 1,150 | 283,566,252 | 392,641,706 | +38.47% | +94,848 | +38.13% |

长数据集的倍率非常稳定：最低 `1.38287x`，中位数 `1.38527x`，最高 `1.38975x`。

### 按 V4P token 长度

| V4P 区间 | 条数 | V4P 平均 | OpenPangu 平均 | 平均多出 | 平均增幅 |
|---|---:|---:|---:|---:|---:|
| <2k | 75 | 1,079 | 1,292 | +213 | +19.75% |
| 2k–8k | 175 | 3,263 | 4,330 | +1,067 | +31.89% |
| 8k–16k | 100 | 8,178 | 11,119 | +2,940 | +35.96% |
| 16k–32k | 100 | 16,233 | 22,308 | +6,075 | +37.43% |
| 32k–64k | 100 | 32,307 | 44,523 | +12,216 | +37.81% |
| 64k–128k | 100 | 64,567 | 89,187 | +24,620 | +38.13% |
| 128k–256k | 100 | 128,996 | 178,578 | +49,582 | +38.44% |
| 256k–512k | 200 | 257,659 | 357,039 | +99,380 | +38.57% |
| >1M | 200 | 1,031,772 | 1,429,039 | +397,267 | +38.50% |

短 prompt 的相对增幅较小；16k 以上逐渐稳定，64k 以上基本可用
`OpenPangu tokens ≈ V4P tokens × 1.385` 估算。

### 逐条差值分位数

| 分位数 | OpenPangu 多出的 tokens | OpenPangu/V4P |
|---|---:|---:|
| 最小 | +187 | 1.17188x |
| P10 | +600 | 1.28641x |
| P25 | +2,919 | 1.35722x |
| P50 | +24,509 | 1.38133x |
| P75 | +99,414 | 1.38500x |
| P90 | +397,156 | 1.38596x |
| P99 | +398,050 | 1.38765x |
| 最大 | +398,474 | 1.38975x |

### 按问题类型

| `problem_type` | 条数 | V4P 平均 | OpenPangu 平均 | 平均差值 | 总量增幅 |
|---|---:|---:|---:|---:|---:|
| `bfs` | 550 | 257,723 | 356,864 | +99,141 | +38.47% |
| `parents` | 600 | 236,364 | 327,278 | +90,913 | +38.46% |

两个问题类型的总量倍率几乎一致，说明这个差异在该数据上主要由 tokenizer 产生，
而不是由 `bfs` 和 `parents` 的文本结构差异产生。

### 上下文阈值

| 阈值 | V4P 超过数 | OpenPangu 超过数 | 由更换 tokenizer 新增跨越数 |
|---|---:|---:|---:|
| 128k | 500 | 500 | 0 |
| 256k | 400 | 400 | 0 |
| 512k | 200 | 200 | 0 |
| 1M | 200 | 200 | 0 |

这批 GraphWalks 数据的长度分布较离散。虽然 OpenPangu token 更多，
但没有额外样本因此跨过上述主要阈值。

全体样本的线性拟合为：

```text
OpenPangu tokens ≈ 1.38523 × V4P tokens - 142
R² ≈ 0.9999996
```

该拟合主要用于长上下文的容量预算，不应代替对实际输入的 tokenizer 编码。

## 使用结论

- 只统计原始 GraphWalks prompt：直接复用 Rust `Tokenizer::from_file` 和 `encode(..., false)`。
- 统计普通 Transformers 文本输入：需要考虑默认 BOS，相比裸文本多 1 token。
- 统计真实聊天请求：必须先应用 chat template，不能只对 message content 编码。
- 已渲染的 chat template 需使用 `add_special_tokens=false`，避免重复 BOS。
- GraphWalks 长上下文容量规划可暂按 V4P token 数的约 1.385 倍预估，最终仍以 OpenPangu 实际编码为准。
