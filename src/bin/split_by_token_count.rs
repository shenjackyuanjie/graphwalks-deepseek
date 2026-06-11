//! 按 token 数阈值把一个 parquet 文件拆成两个：
//!   低档（<= threshold）→ output_low
//!   高档（>  threshold）→ output_high

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Int32Array};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use deepseek_graphwalks::stats::TOKEN_COL;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;

#[derive(Parser, Debug)]
#[command(about = "按 token 数阈值把一个 parquet 文件拆成两个")]
struct Args {
    /// 输入的 parquet 文件（须包含 deepseek_v4_input_tokens 列）。
    #[arg(long)]
    input: PathBuf,

    /// token 数分割阈值（含）：<= threshold 进 output_low，> threshold 进 output_high。
    #[arg(long, default_value_t = 1_000_000)]
    threshold: i32,

    /// token 数 <= threshold 的行写入此文件。
    #[arg(long)]
    output_low: PathBuf,

    /// token 数 > threshold 的行写入此文件。
    #[arg(long)]
    output_high: PathBuf,

    /// 每个 parquet batch 的行数。
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let file = File::open(&args.input)
        .with_context(|| format!("无法打开输入 parquet: {}", args.input.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| "无法创建 parquet reader")?;

    let schema = builder.schema().clone();
    let token_idx = schema
        .index_of(TOKEN_COL)
        .with_context(|| format!("找不到列 {TOKEN_COL:?}"))?;

    let reader = builder
        .with_batch_size(args.batch_size)
        .build()
        .with_context(|| "无法构建 parquet reader")?;

    let file_low = File::create(&args.output_low)
        .with_context(|| format!("无法创建输出文件: {}", args.output_low.display()))?;
    let file_high = File::create(&args.output_high)
        .with_context(|| format!("无法创建输出文件: {}", args.output_high.display()))?;

    let mut writer_low: Option<ArrowWriter<File>> = None;
    let mut writer_high: Option<ArrowWriter<File>> = None;
    let mut count_low = 0u64;
    let mut count_high = 0u64;

    for batch_result in reader {
        let batch = batch_result.with_context(|| "读取 parquet batch 失败")?;

        let tokens = batch
            .column(token_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| anyhow!("列 {TOKEN_COL:?} 不是 Int32 类型"))?;

        // 收集每行属于哪侧
        let mut low_indices: Vec<usize> = Vec::new();
        let mut high_indices: Vec<usize> = Vec::new();
        for i in 0..tokens.len() {
            let v = if tokens.is_null(i) { 0 } else { tokens.value(i) };
            if v <= args.threshold {
                low_indices.push(i);
            } else {
                high_indices.push(i);
            }
        }

        if !low_indices.is_empty() {
            let sub = filter_batch(&batch, &low_indices)?;
            count_low += sub.num_rows() as u64;
            let w = writer_low.get_or_insert_with(|| {
                ArrowWriter::try_new(file_low.try_clone().unwrap(), Arc::clone(&sub.schema()), None)
                    .expect("无法创建 low 端 parquet writer")
            });
            w.write(&sub).with_context(|| "写入 low 端 batch 失败")?;
        }

        if !high_indices.is_empty() {
            let sub = filter_batch(&batch, &high_indices)?;
            count_high += sub.num_rows() as u64;
            let w = writer_high.get_or_insert_with(|| {
                ArrowWriter::try_new(
                    file_high.try_clone().unwrap(),
                    Arc::clone(&sub.schema()),
                    None,
                )
                .expect("无法创建 high 端 parquet writer")
            });
            w.write(&sub).with_context(|| "写入 high 端 batch 失败")?;
        }
    }

    if let Some(w) = writer_low {
        w.close().with_context(|| "关闭 low 端 parquet writer 失败")?;
    }
    if let Some(w) = writer_high {
        w.close().with_context(|| "关闭 high 端 parquet writer 失败")?;
    }

    println!(
        "完成。threshold={}",
        args.threshold
    );
    println!(
        "  <= threshold: {} 行 → {}",
        count_low,
        args.output_low.display()
    );
    println!(
        "  >  threshold: {} 行 → {}",
        count_high,
        args.output_high.display()
    );

    Ok(())
}

/// 从 batch 中按行下标提取子 RecordBatch。
fn filter_batch(batch: &RecordBatch, indices: &[usize]) -> Result<RecordBatch> {
    use arrow::array::UInt64Array;
    use arrow::compute::take;

    let idx_arr = UInt64Array::from(indices.iter().map(|&i| i as u64).collect::<Vec<_>>());
    let columns: Vec<_> = batch
        .columns()
        .iter()
        .map(|col| take(col.as_ref(), &idx_arr, None).with_context(|| "take 列失败"))
        .collect::<Result<_>>()?;

    RecordBatch::try_new(Arc::clone(&batch.schema()), columns)
        .with_context(|| "构建过滤后的 RecordBatch 失败")
}
