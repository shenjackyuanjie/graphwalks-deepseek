use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use deepseek_graphwalks::eval;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;

// ── 命令行参数 ────────────────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(name = "run_graphwalks")]
struct Args {
    /// OpenAI 兼容的 API 地址。
    #[arg(long, default_value = "https://api.deepseek.com/v1")]
    base_url: String,

    /// API 密钥。未传时从 DEEPSEEK_API_KEY 环境变量读取。
    #[arg(long)]
    api_key: Option<String>,

    /// 模型名称。
    #[arg(long, default_value = "deepseek-chat")]
    model: String,

    /// 输入的 parquet 文件（graphwalks 数据集）。
    #[arg(short, long)]
    input: PathBuf,

    /// 最多测试多少条样本，默认全部。
    #[arg(long)]
    max_samples: Option<usize>,

    /// 并发请求数。
    #[arg(long, default_value_t = 1)]
    concurrency: usize,

    /// 思考强度。传入 high 或 max 开启思考模式，不传则关闭。
    #[arg(long)]
    thinking_effort: Option<String>,

    /// 输出 CSV 文件，写入每条样本的评分结果。
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ── 主函数 ───────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    let api_key = match &args.api_key {
        Some(k) => k.clone(),
        None => std::env::var("DEEPSEEK_API_KEY")
            .context("缺少 API 密钥：请通过 --api-key 传入或设置 DEEPSEEK_API_KEY 环境变量")?,
    };

    let samples = eval::load_samples(&args.input, args.max_samples)?;
    println!("从 {} 加载了 {} 条样本", args.input.display(), samples.len());

    let client = Arc::new(reqwest::Client::new());
    let base_url = Arc::new(args.base_url.trim_end_matches('/').to_owned());
    let model = Arc::new(args.model);
    let api_key = Arc::new(api_key);

    let pb = Arc::new(ProgressBar::new(samples.len() as u64));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let concurrency = args.concurrency;
    let thinking_effort = Arc::new(args.thinking_effort);
    let extract_re = Arc::new(Regex::new("Final Answer: ?\\[(.*?)\\]").unwrap());

    let results: Vec<eval::EvalResult> = stream::iter(samples.into_iter().map(|sample| {
        let client = client.clone();
        let base_url = base_url.clone();
        let model = model.clone();
        let api_key = api_key.clone();
        let pb = pb.clone();
        let thinking_effort = thinking_effort.clone();
        let extract_re = extract_re.clone();
        async move {
            let result = eval::eval_one(
                &client,
                &base_url,
                &model,
                &api_key,
                thinking_effort.as_deref(),
                &extract_re,
                &sample,
            )
            .await;
            pb.inc(1);
            pb.set_message(format!("#{}", sample.index));
            result
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;

    pb.finish_and_clear();

    if let Some(ref out_path) = args.output {
        eval::write_csv(out_path, &results)?;
        println!("结果已写入 {}", out_path.display());
    }

    eval::print_summary(&results);

    Ok(())
}
