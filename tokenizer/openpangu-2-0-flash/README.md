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
