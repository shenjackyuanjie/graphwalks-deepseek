use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use deepseek_graphwalks::stats;

#[derive(Parser, Debug)]
struct Args {
    /// 每个 parquet batch 的行数。
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,

    /// 输入的 parquet 文件（须包含 deepseek_v4_input_tokens 列）。
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let buckets = stats::buckets();
    let mut per_file = Vec::new();
    let mut all = stats::Stats {
        bucket_counts: vec![0; buckets.len()],
        ..stats::Stats::default()
    };

    for input in &args.inputs {
        let st = stats::read_stats(input, args.batch_size, &buckets)?;
        stats::merge_stats(&mut all, &st);
        per_file.push((input.clone(), st));
    }

    println!("# DeepSeek V4 Token 数分布\n");
    println!("统计列: `{}`", stats::TOKEN_COL);
    println!("区间为 `[下限, 上限)`，最后一个区间除外。\n");

    stats::print_table("全部文件", &all, &buckets);
    for (path, st) in per_file {
        println!();
        stats::print_table(&path.to_string_lossy(), &st, &buckets);
    }

    Ok(())
}
