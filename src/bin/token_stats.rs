use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use arrow::array::{Array, Int32Array};
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

const TOKEN_COL: &str = "deepseek_v4_input_tokens";

#[derive(Parser, Debug)]
struct Args {
    /// 每个 parquet batch 的行数。
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// 输入的 parquet 文件（须包含 deepseek_v4_input_tokens 列）。
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

#[derive(Clone, Copy)]
struct Bucket {
    label: &'static str,
    min_inclusive: i32,
    max_exclusive: Option<i32>,
}

#[derive(Default)]
struct Stats {
    total: u64,
    sum_tokens: u64,
    min_tokens: Option<i32>,
    max_tokens: Option<i32>,
    bucket_counts: Vec<u64>,
    over_1m: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let buckets = buckets();
    let mut per_file = Vec::new();
    let mut all = Stats {
        bucket_counts: vec![0; buckets.len()],
        ..Stats::default()
    };

    for input in &args.inputs {
        let stats = read_stats(input, args.batch_size, &buckets)?;
        merge_stats(&mut all, &stats);
        per_file.push((input.clone(), stats));
    }

    println!("# DeepSeek V4 Token 数分布\n");
    println!("统计列: `{TOKEN_COL}`");
    println!("区间为 `[下限, 上限)`，最后一个区间除外。\n");

    print_table("全部文件", &all, &buckets);
    for (path, stats) in per_file {
        println!();
        print_table(&path.to_string_lossy(), &stats, &buckets);
    }

    Ok(())
}

fn buckets() -> Vec<Bucket> {
    vec![
        Bucket {
            label: "<=8k",
            min_inclusive: 0,
            max_exclusive: Some(8_001),
        },
        Bucket {
            label: "8k-16k",
            min_inclusive: 8_001,
            max_exclusive: Some(16_001),
        },
        Bucket {
            label: "16k-32k",
            min_inclusive: 16_001,
            max_exclusive: Some(32_001),
        },
        Bucket {
            label: "32k-64k",
            min_inclusive: 32_001,
            max_exclusive: Some(64_001),
        },
        Bucket {
            label: "64k-128k",
            min_inclusive: 64_001,
            max_exclusive: Some(128_001),
        },
        Bucket {
            label: "128k-256k",
            min_inclusive: 128_001,
            max_exclusive: Some(256_001),
        },
        Bucket {
            label: "256k-512k",
            min_inclusive: 256_001,
            max_exclusive: Some(512_001),
        },
        Bucket {
            label: "512k-1M",
            min_inclusive: 512_001,
            max_exclusive: Some(1_000_001),
        },
        Bucket {
            label: ">1M",
            min_inclusive: 1_000_001,
            max_exclusive: None,
        },
    ]
}

fn read_stats(path: &Path, batch_size: usize, buckets: &[Bucket]) -> Result<Stats> {
    let file = File::open(path)
        .with_context(|| format!("无法打开输入 parquet: {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法创建 parquet reader: {}", path.display()))?;
    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .with_context(|| format!("无法构建 parquet reader: {}", path.display()))?;

    let mut stats = Stats {
        bucket_counts: vec![0; buckets.len()],
        ..Stats::default()
    };

    for batch in reader {
        let batch = batch.with_context(|| "读取 parquet batch 失败")?;
        let idx = batch
            .schema()
            .index_of(TOKEN_COL)
            .with_context(|| format!("找不到列 {TOKEN_COL:?}"))?;
        let arr = batch
            .column(idx)
            .as_any()
            .downcast_ref::<Int32Array>()
            .ok_or_else(|| anyhow!("列 {TOKEN_COL:?} 不是 Int32 类型"))?;

        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let n = arr.value(i);
            update_stats(&mut stats, n, buckets);
        }
    }

    Ok(stats)
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

fn merge_stats(dst: &mut Stats, src: &Stats) {
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

fn print_table(title: &str, stats: &Stats, buckets: &[Bucket]) {
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
