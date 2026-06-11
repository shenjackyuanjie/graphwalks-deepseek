use std::collections::VecDeque;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use clap::Parser;
use deepseek_graphwalks::{api::StreamTick, eval};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use tokio::sync::mpsc;

// 命令行参数

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
    #[arg(long, default_value = "deepseek-v4-flash")]
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

// 实时统计

/// 滑动窗口时长（用于 streaming chars/s）。
const WINDOW: Duration = Duration::from_secs(30);

struct LiveStats {
    completed: AtomicUsize,
    total_prompt_tokens: AtomicU64,
    total_completion_tokens: AtomicU64,
    /// 累计 streaming 字符数（content + reasoning）。
    stream_chars: AtomicU64,
    /// streaming 字符增量事件 (时间, 增量字符数)。
    stream_ticks: Mutex<VecDeque<(Instant, usize)>>,
    /// 最后一个完成的样本标签。
    last_done: Mutex<Option<String>>,
}

impl LiveStats {
    fn new() -> Self {
        Self {
            completed: AtomicUsize::new(0),
            total_prompt_tokens: AtomicU64::new(0),
            total_completion_tokens: AtomicU64::new(0),
            stream_chars: AtomicU64::new(0),
            stream_ticks: Mutex::new(VecDeque::new()),
            last_done: Mutex::new(None),
        }
    }

    fn record_stream_tick(&self, tick: &StreamTick, now: Instant) {
        let delta = tick.content_delta_chars + tick.reasoning_delta_chars;
        if delta == 0 {
            return;
        }
        self.stream_chars.fetch_add(delta as u64, Ordering::Relaxed);
        let mut ticks = self.stream_ticks.lock().unwrap();
        ticks.push_back((now, delta));
        while ticks
            .front()
            .map_or(false, |(t, _)| now - *t > WINDOW)
        {
            ticks.pop_front();
        }
    }

    fn record_done(&self, prompt_tokens: u64, completion_tokens: u64, label: String) {
        self.completed.fetch_add(1, Ordering::Relaxed);
        self.total_prompt_tokens
            .fetch_add(prompt_tokens, Ordering::Relaxed);
        self.total_completion_tokens
            .fetch_add(completion_tokens, Ordering::Relaxed);
        *self.last_done.lock().unwrap() = Some(label);
    }

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

        let (stream_chars, chars_per_sec) = self.stream_speed(now);

        let mut parts = vec![format!("total:{total_tokens} avg:{avg_tokens}/s")];

        if stream_chars > 0 {
            parts.push(format!("out:{stream_chars}c {chars_per_sec}c/s"));
        }

        if let Some(ref last) = *self.last_done.lock().unwrap() {
            parts.push(last.clone());
        }

        parts.join(" ")
    }

    /// 返回 (累计字符数, 窗口内 chars/sec)。
    fn stream_speed(&self, now: Instant) -> (u64, u64) {
        let mut ticks = self.stream_ticks.lock().unwrap();
        while ticks
            .front()
            .map_or(false, |(t, _)| now - *t > WINDOW)
        {
            ticks.pop_front();
        }
        if ticks.len() < 2 {
            return (0, 0);
        }
        let first_time = ticks.front().unwrap().0;
        let window_chars: usize = ticks.iter().map(|(_, n)| n).sum();
        let duration = (now - first_time).as_secs_f64();
        if duration > 0.3 {
            let chars_per_sec = window_chars as f64 / duration;
            (self.stream_chars.load(Ordering::Relaxed), chars_per_sec as u64)
        } else {
            (0, 0)
        }
    }
}

// 主函数

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
    println!(
        "并发: {}  |  输出: {}",
        args.concurrency,
        args.output.display()
    );
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

    // tick channel: streaming delta -> 后台刷新
    let (tick_tx, mut tick_rx) = mpsc::unbounded_channel::<StreamTick>();

    // 后台任务：消费 stream tick + 定时刷新进度条
    let stats_bg = stats.clone();
    let pb_bg = pb.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_millis(250));
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    pb_bg.set_message(stats_bg.progress_msg());
                    let _ = std::io::stderr().flush();
                }
                tick = tick_rx.recv() => {
                    match tick {
                        Some(t) => {
                            stats_bg.record_stream_tick(&t, Instant::now());
                            pb_bg.set_message(stats_bg.progress_msg());
                            let _ = std::io::stderr().flush();
                        }
                        None => break,
                    }
                }
            }
        }
        pb_bg.set_message(stats_bg.progress_msg());
        let _ = std::io::stderr().flush();
    });

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
        let tick_tx = tick_tx.clone();
        async move {
            let t0 = Instant::now();
            let result = eval::eval_one_streaming(
                &client,
                &base_url,
                &model,
                &api_key,
                thinking_effort.as_deref(),
                &extract_re,
                &sample,
                &tick_tx,
            )
            .await;
            let elapsed = t0.elapsed();

            let prompt_tokens = result
                .usage
                .as_ref()
                .map_or(0, |u| u.prompt_tokens as u64);
            let completion_tokens = result
                .usage
                .as_ref()
                .map_or(0, |u| u.completion_tokens as u64);
            let tps = if elapsed.as_secs_f64() > 0.0 {
                completion_tokens as f64 / elapsed.as_secs_f64()
            } else {
                0.0
            };

            let label = if result.error.is_some() {
                format!("#{} ERR", sample.index)
            } else {
                format!("#{} F1={:.2}", sample.index, result.f1)
            };

            stats.record_done(prompt_tokens, completion_tokens, label);
            pb.inc(1);

            // 每个样本完成后打印一行
            let msg = if let Some(ref e) = result.error {
                format!("#{idx} ERR: {e}", idx = result.index)
            } else {
                let u = result.usage.as_ref();
                let mut extra = String::new();
                if let Some(u) = u {
                    if u.prompt_cache_hit_tokens > 0 || u.prompt_cache_miss_tokens > 0 {
                        extra.push_str(&format!(
                            " cache:{}/{}",
                            u.prompt_cache_hit_tokens, u.prompt_cache_miss_tokens
                        ));
                    }
                    if let Some(ref d) = u.completion_tokens_details {
                        if d.reasoning_tokens > 0 {
                            extra.push_str(&format!(" reason:{}", d.reasoning_tokens));
                        }
                    }
                }
                format!(
                    "#{idx} F1={f1:.4} R={recall:.4} P={precision:.4} | in:{input_tok} out:{output_tok} {tps:.0}t/s{extra} | pred={pred:?} truth={truth:?}",
                    idx = result.index,
                    f1 = result.f1,
                    recall = result.recall,
                    precision = result.precision,
                    input_tok = u.map_or(0, |u| u.prompt_tokens),
                    output_tok = u.map_or(0, |u| u.completion_tokens),
                    tps = tps,
                    extra = extra,
                    pred = result.predicted,
                    truth = result.ground_truth,
                )
            };
            pb.println(msg);

            result
        }
    }))
    .buffer_unordered(concurrency)
    .collect()
    .await;

    // drop tick_tx 以关闭后台任务
    drop(tick_tx);
    tokio::time::sleep(Duration::from_millis(100)).await;

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
