use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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

    /// 输出 CSV 文件路径。
    #[arg(short, long, default_value = "results/eval_result.csv")]
    output: PathBuf,
}

// ── 实时统计 ──────────────────────────────────────────────────────────────

/// 滑动 TPS 窗口时长。
const TPS_WINDOW: Duration = Duration::from_secs(60);

struct LiveStats {
    completed: AtomicUsize,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    /// 滑动窗口中每个样本完成时的 (时间戳, 本次增量 token 数)。
    tps_events: Mutex<VecDeque<(Instant, u64)>>,
    /// 最后一个完成的样本 F1（用于进度条展示）。
    last_f1: Mutex<Option<String>>,
}

impl LiveStats {
    fn new() -> Self {
        Self {
            completed: AtomicUsize::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            tps_events: Mutex::new(VecDeque::new()),
            last_f1: Mutex::new(None),
        }
    }

    /// 记录一个样本的完成。
    fn record(&self, prompt_tokens: u64, completion_tokens: u64, f1_label: String) {
        let now = Instant::now();
        let delta = prompt_tokens + completion_tokens;

        self.completed.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);

        // 更新滑动窗口
        {
            let mut events = self.tps_events.lock().unwrap();
            events.push_back((now, delta));
            while events
                .front()
                .map_or(false, |(t, _)| now - *t > TPS_WINDOW)
            {
                events.pop_front();
            }
        }

        *self.last_f1.lock().unwrap() = Some(f1_label);
    }

    /// 格式化进度条消息。
    fn progress_msg(&self) -> String {
        let now = Instant::now();
        let completed = self.completed.load(Ordering::Relaxed);
        let prompt = self.total_prompt_tokens.load(Ordering::Relaxed);
        let completion = self.total_completion_tokens.load(Ordering::Relaxed);
        let total_tokens = prompt + completion;

        let avg_tokens = if completed > 0 {
            total_tokens / completed as u64
        } else {
            0
        };

        let tps = self.sliding_tps(now);

        let last = self.last_f1.lock().unwrap().clone().unwrap_or_default();

        format!(
            "tokens:{total_tokens} avg:{avg_tokens}/s TPS:{tps:.0} {last}",
        )
    }

    fn sliding_tps(&self, now: Instant) -> f64 {
        let mut events = self.tps_events.lock().unwrap();
        while events
            .front()
            .map_or(false, |(t, _)| now - *t > TPS_WINDOW)
        {
            events.pop_front();
        }
        if events.len() < 2 {
            return 0.0;
        }
        let first_time = events.front().unwrap().0;
        let window_tokens: u64 = events.iter().map(|(_, tokens)| tokens).sum();
        let duration = (now - first_time).as_secs_f64();
        if duration > 0.0 {
            window_tokens as f64 / duration
        } else {
            0.0
        }
    }
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
    println!("并发: {}  |  输出: {}", args.concurrency, args.output.display());
    println!();

    let client = Arc::new(reqwest::Client::new());
    let base_url = Arc::new(args.base_url.trim_end_matches('/').to_owned());
    let model = Arc::new(args.model);
    let api_key = Arc::new(api_key);

    let total = samples.len() as u64;
    let pb = Arc::new(ProgressBar::new(total));
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}",
        )
        .unwrap()
        .progress_chars("#>-"),
    );

    let stats = Arc::new(LiveStats::new());
    let concurrency = args.concurrency;
    let thinking_effort = Arc::new(args.thinking_effort);
    let extract_re = Arc::new(Regex::new("Final Answer: ?\\[(.*?)\\]").unwrap());

    let results: Vec<eval::EvalResult> = stream::iter(samples.into_iter().map(|sample| {
        let client = client.clone();
        let base_url = base_url.clone();
        let model = model.clone();
        let api_key = api_key.clone();
        let pb = pb.clone();
        let stats = stats.clone();
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

            let prompt_tokens = result
                .usage
                .as_ref()
                .map_or(0, |u| u.prompt_tokens as u64);
            let completion_tokens = result
                .usage
                .as_ref()
                .map_or(0, |u| u.completion_tokens as u64);

            let f1_label = if result.error.is_some() {
                format!("#{} ERR", sample.index)
            } else {
                format!("#{} F1={:.2}", sample.index, result.f1)
            };

            stats.record(prompt_tokens, completion_tokens, f1_label);
            pb.inc(1);
            pb.set_message(stats.progress_msg());

            // 实时打印非完美样本
            if result.f1 < 0.999 || result.error.is_some() {
                let msg = if let Some(ref e) = result.error {
                    format!("#{} 错误: {e}", result.index)
                } else {
                    format!(
                        "#{} F1={:.4} R={:.4} P={:.4} | pred={:?} truth={:?}",
                        result.index,
                        result.f1,
                        result.recall,
                        result.precision,
                        result.predicted,
                        result.ground_truth,
                    )
                };
                pb.println(msg);
            }

            result
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;

    pb.finish_and_clear();

    // 确保输出目录存在
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("无法创建目录: {}", parent.display()))?;
    }
    eval::write_csv(&args.output, &results)?;
    println!("结果已写入 {}", args.output.display());

    eval::print_summary(&results);

    Ok(())
}
