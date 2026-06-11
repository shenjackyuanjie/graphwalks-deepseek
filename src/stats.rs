//! Token 统计：从 parquet 的 token 数列读取并计算分布。

use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Int32Array, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

pub const TOKEN_COL: &str = "deepseek_v4_input_tokens";
pub const PROBLEM_TYPE_COL: &str = "problem_type";

#[derive(Clone, Copy)]
pub struct Bucket {
    pub label: &'static str,
    pub min_inclusive: i32,
    pub max_exclusive: Option<i32>,
}

#[derive(Default)]
pub struct Stats {
    pub total: u64,
    pub sum_tokens: u64,
    pub min_tokens: Option<i32>,
    pub max_tokens: Option<i32>,
    pub bucket_counts: Vec<u64>,
    pub over_1m: u64,
}

pub fn buckets() -> Vec<Bucket> {
    vec![
        Bucket { label: "<=8k", min_inclusive: 0, max_exclusive: Some(8_001) },
        Bucket { label: "8k-16k", min_inclusive: 8_001, max_exclusive: Some(16_001) },
        Bucket { label: "16k-32k", min_inclusive: 16_001, max_exclusive: Some(32_001) },
        Bucket { label: "32k-64k", min_inclusive: 32_001, max_exclusive: Some(64_001) },
        Bucket { label: "64k-128k", min_inclusive: 64_001, max_exclusive: Some(128_001) },
        Bucket { label: "128k-256k", min_inclusive: 128_001, max_exclusive: Some(256_001) },
        Bucket { label: "256k-512k", min_inclusive: 256_001, max_exclusive: Some(512_001) },
        Bucket { label: "512k-1M", min_inclusive: 512_001, max_exclusive: Some(1_000_001) },
        Bucket { label: ">1M", min_inclusive: 1_000_001, max_exclusive: None },
    ]
}

/// 按文件整体及按 problem_type 分组的统计结果。
pub struct StatsReport {
    pub overall: Stats,
    /// key = problem_type 值（如 "bfs" / "parents"）。
    pub by_type: HashMap<String, Stats>,
}

/// 从 parquet 文件读取 TOKEN_COL 列并计算统计信息，同时按 problem_type 分组。
pub fn read_stats(path: &Path, batch_size: usize, buckets: &[Bucket]) -> Result<StatsReport> {
    let file = File::open(path)
        .with_context(|| format!("无法打开输入 parquet: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法创建 parquet reader: {}", path.display()))?;
    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .with_context(|| format!("无法构建 parquet reader: {}", path.display()))?;

    let mut overall = Stats {
        bucket_counts: vec![0; buckets.len()],
        ..Stats::default()
    };
    let mut by_type: HashMap<String, Stats> = HashMap::new();

    for batch in reader {
        let batch = batch.with_context(|| "读取 parquet batch 失败")?;
        let schema = batch.schema();
        let token_idx = schema
            .index_of(TOKEN_COL)
            .with_context(|| format!("找不到列 {TOKEN_COL:?}"))?;
        let arr = batch
            .column(token_idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| anyhow!("列 {TOKEN_COL:?} 不是 Int32 类型"))?;

        let type_arr: Option<&StringArray> = schema.index_of(PROBLEM_TYPE_COL).ok().map(|idx| {
            batch
                .column(idx)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("problem_type 列不是 String 类型")
        });

        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let n = arr.value(i);
            update_stats(&mut overall, n, buckets);

            if let Some(types) = type_arr {
                if !types.is_null(i) {
                    let pt = types.value(i).to_owned();
                    let entry = by_type.entry(pt).or_insert_with(|| Stats {
                        bucket_counts: vec![0; buckets.len()],
                        ..Stats::default()
                    });
                    update_stats(entry, n, buckets);
                }
            }
        }
    }

    Ok(StatsReport { overall, by_type })
}

fn update_stats(stats: &mut Stats, tokens: i32, buckets: &[Bucket]) {
    stats.total += 1;
    stats.sum_tokens += tokens.max(0) as u64;
    stats.min_tokens = Some(stats.min_tokens.map_or(tokens, |min| min.min(tokens)));
    stats.max_tokens = Some(stats.max_tokens.map_or(tokens, |max| max.max(tokens)));
    if tokens > 1_000_000 {
        stats.over_1m += 1;
    }

    for (idx, bucket) in buckets.iter().enumerate() {
        let below_max = bucket
            .max_exclusive
            .is_none_or(|max_exclusive| tokens < max_exclusive);
        if tokens >= bucket.min_inclusive && below_max {
            stats.bucket_counts[idx] += 1;
            return;
        }
    }
}

pub fn merge_report(dst: &mut StatsReport, src: &StatsReport, buckets: &[Bucket]) {
    merge_stats(&mut dst.overall, &src.overall);
    for (pt, src_st) in &src.by_type {
        let entry = dst.by_type.entry(pt.clone()).or_insert_with(|| Stats {
            bucket_counts: vec![0; buckets.len()],
            ..Stats::default()
        });
        merge_stats(entry, src_st);
    }
}

pub fn merge_stats(dst: &mut Stats, src: &Stats) {
    dst.total += src.total;
    dst.sum_tokens += src.sum_tokens;
    dst.over_1m += src.over_1m;
    dst.min_tokens = match (dst.min_tokens, src.min_tokens) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (None, b) => b,
        (a, None) => a,
    };
    dst.max_tokens = match (dst.max_tokens, src.max_tokens) {
        (Some(a), Some(b)) => Some(a.max(b)),
        (None, b) => b,
        (a, None) => a,
    };

    if dst.bucket_counts.is_empty() {
        dst.bucket_counts = vec![0; src.bucket_counts.len()];
    }
    for (dst_count, src_count) in dst.bucket_counts.iter_mut().zip(&src.bucket_counts) {
        *dst_count += src_count;
    }
}

pub fn print_table(title: &str, stats: &Stats, buckets: &[Bucket]) {
    println!("## {title}\n");
    println!(
        "行数: {} | 最小值: {} | 最大值: {} | 均值: {:.1} | >1M: {}",
        stats.total,
        stats.min_tokens.unwrap_or_default(),
        stats.max_tokens.unwrap_or_default(),
        mean(stats),
        stats.over_1m
    );
    println!();
    println!("| Token 区间 | 行数 | 占比 |");
    println!("|---|---:|---:|");
    for (bucket, count) in buckets.iter().zip(&stats.bucket_counts) {
        println!(
            "| {} | {} | {:.2}% |",
            bucket.label,
            count,
            percent(*count, stats.total)
        );
    }
}

fn mean(stats: &Stats) -> f64 {
    if stats.total == 0 {
        0.0
    } else {
        stats.sum_tokens as f64 / stats.total as f64
    }
}

fn percent(count: u64, total: u64) -> f64 {
    if total == 0 {
        0.0
    } else {
        count as f64 * 100.0 / total as f64
    }
}
