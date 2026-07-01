use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use clap::Parser;
use deepseek_graphwalks::token_counter;
use tokenizers::Tokenizer;

#[derive(Parser, Debug)]
struct Args {
    /// tokenizer.json 路径。
    #[arg(long, default_value = "tokenizer/deepseek-v4-pro/tokenizer.json")]
    tokenizer_json: PathBuf,

    /// 输出标识，用于生成列名和文件后缀。
    #[arg(long, default_value = "deepseek_v4")]
    output_tag: String,

    /// 要统计 token 数的文本列名。GraphWalks 数据集用 prompt。
    #[arg(long, default_value = "prompt")]
    text_col: String,

    /// 每个 parquet batch 的行数。长上下文行可能很大，建议用小值。
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// 输入的 parquet 文件。
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tokenizer = Tokenizer::from_file(&args.tokenizer_json)
        .map_err(|e| anyhow!("加载 tokenizer 失败 {:?}: {e}", args.tokenizer_json))?;
    let tokenizer = Arc::new(tokenizer);

    for input in &args.inputs {
        token_counter::process_file(
            input,
            tokenizer.clone(),
            &args.text_col,
            args.batch_size,
            &args.output_tag,
        )?;
    }

    Ok(())
}
