use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{anyhow, Result};
use clap::Parser;
use deepseek_graphwalks::utils;

use arrow::array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

#[derive(Parser, Debug)]
struct Args {
    /// 输入的 parquet 文件。
    #[arg(short, long)]
    input: PathBuf,

    /// 要检查的 token 数列。
    #[arg(long, default_value = "deepseek_v4_input_tokens")]
    token_col: String,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file =
        std::fs::File::open(&args.input).map_err(|e| anyhow!("无法打开文件 {}: {e}", args.input.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| anyhow!("无法读取 parquet: {e}"))?;

    let schema = builder.schema().clone();
    let problem_type_idx = schema
        .index_of("problem_type")
        .map_err(|_| anyhow!("缺少 'problem_type' 列"))?;
    let token_idx = schema.index_of(&args.token_col).ok();

    let reader = builder.with_batch_size(64).build()?;

    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;
    let mut token_ranges: Vec<(String, i32)> = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let problem_types = utils::read_string_column(&batch, problem_type_idx)?;
        let token_col = token_idx.map(|idx| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap()
        });

        for row in 0..problem_types.len() {
            let pt = problem_types.value(row).to_owned();
            *type_counts.entry(pt.clone()).or_default() += 1;

            if let Some(tokens) = token_col {
                token_ranges.push((pt, tokens.value(row)));
            }
            total += 1;
        }
    }

    println!("文件: {}", args.input.display());
    println!("总样本数: {total}\n");

    println!("problem_type 分布:");
    let mut sorted: Vec<_> = type_counts.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));
    for (pt, count) in &sorted {
        println!("  {pt}: {count} ({:.2}%)", *count as f64 / total as f64 * 100.0);
    }

    // 打印前 5 条和后 5 条的 problem_type + token 数
    if !token_ranges.is_empty() {
        println!("\n前 5 条样本:");
        for (i, (pt, tokens)) in token_ranges.iter().take(5).enumerate() {
            println!("  [{i}] {pt}  tokens={tokens}");
        }
        println!("\n后 5 条样本:");
        for (i, (pt, tokens)) in token_ranges.iter().rev().take(5).rev().enumerate() {
            let idx = total - 5 + i;
            println!("  [{idx}] {pt}  tokens={tokens}");
        }
    }

    Ok(())
}
