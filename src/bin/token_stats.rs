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
    let mut per_file: Vec<(PathBuf, stats::StatsReport)> = Vec::new();
    let mut all = stats::StatsReport {
        overall: stats::Stats {
            bucket_counts: vec![0; buckets.len()],
            ..stats::Stats::default()
        },
        by_type: Default::default(),
    };

    for input in &args.inputs {
        let report = stats::read_stats(input, args.batch_size, &buckets)?;
        stats::merge_report(&mut all, &report, &buckets);
        per_file.push((input.clone(), report));
    }

    println!("# DeepSeek V4 Token 数分布\n");
    println!("统计列: `{}`", stats::TOKEN_COL);
    println!("区间为 `[下限, 上限)`，最后一个区间除外。\n");

    print_report("全部文件", &all, &buckets);

    for (path, report) in &per_file {
        println!();
        print_report(&path.to_string_lossy(), report, &buckets);
    }

    Ok(())
}

fn print_report(title: &str, report: &stats::StatsReport, buckets: &[stats::Bucket]) {
    stats::print_table(&format!("{title} — 合计"), &report.overall, buckets);

    // 按 problem_type 排序后逐一输出
    let mut sorted_types: Vec<&String> = report.by_type.keys().collect();
    sorted_types.sort();
    for pt in sorted_types {
        println!();
        stats::print_table(&format!("{title} — {pt}"), &report.by_type[pt], buckets);
    }
}
