use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};
use graphwalks::utils;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};

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

    /// 输出 CSV 文件，写入每条样本的评分结果。
    #[arg(short, long)]
    output: Option<PathBuf>,
}

// ── API 类型定义 ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
}

// ── 样本 ─────────────────────────────────────────────────────────────────

struct Sample {
    /// 在 parquet 文件中的行号（从 0 开始）。
    index: usize,
    prompt: String,
    ground_truth: HashSet<String>,
    problem_type: String,
}

// ── 评分结果 ──────────────────────────────────────────────────────────────

struct EvalResult {
    index: usize,
    problem_type: String,
    predicted: Vec<String>,
    ground_truth: HashSet<String>,
    recall: f64,
    precision: f64,
    f1: f64,
    response: String,
    error: Option<String>,
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

    let samples = load_samples(&args)?;
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
    let extract_re = Arc::new(Regex::new(r"Final Answer: ?\[(.*?)\]").unwrap());

    let results: Vec<EvalResult> = stream::iter(samples.into_iter().map(|sample| {
        let client = client.clone();
        let base_url = base_url.clone();
        let model = model.clone();
        let api_key = api_key.clone();
        let pb = pb.clone();
        let extract_re = extract_re.clone();
        async move {
            let result = eval_one(&client, &base_url, &model, &api_key, &extract_re, &sample).await;
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
        write_csv(out_path, &results)?;
        println!("结果已写入 {}", out_path.display());
    }

    print_summary(&results);

    Ok(())
}

// ── 加载样本 ─────────────────────────────────────────────────────────────

fn load_samples(args: &Args) -> Result<Vec<Sample>> {
    let file =
        File::open(&args.input).with_context(|| format!("无法打开文件: {}", args.input.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法读取 parquet: {}", args.input.display()))?;

    let schema = builder.schema().clone();

    let prompt_idx = schema
        .index_of("prompt")
        .map_err(|_| anyhow!("缺少 'prompt' 列"))?;
    let answer_idx = schema
        .index_of("answer_nodes")
        .map_err(|_| anyhow!("缺少 'answer_nodes' 列"))?;
    let problem_type_idx = schema
        .index_of("problem_type")
        .map_err(|_| anyhow!("缺少 'problem_type' 列"))?;

    let reader = builder.with_batch_size(32).build()?;
    let mut samples = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let prompts = utils::read_string_column(&batch, prompt_idx)?;
        let problem_types = utils::read_string_column(&batch, problem_type_idx)?;
        let answer_lists = utils::read_list_column(&batch, answer_idx)?;

        for row in 0..batch.num_rows() {
            if let Some(max) = args.max_samples {
                if samples.len() >= max {
                    return Ok(samples);
                }
            }
            let prompt = prompts.value(row).to_owned();
            let problem_type = problem_types.value(row).to_owned();
            let ground_truth: HashSet<String> = answer_lists[row]
                .iter()
                .map(|s| s.to_string())
                .collect();

            samples.push(Sample {
                index: samples.len(),
                prompt,
                ground_truth,
                problem_type,
            });
        }
    }

    Ok(samples)
}

// ── 评估单条样本 ──────────────────────────────────────────────────────────

async fn eval_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    extract_re: &Regex,
    sample: &Sample,
) -> EvalResult {
    let response = match call_api(client, base_url, model, api_key, &sample.prompt).await {
        Ok(content) => content,
        Err(e) => {
            return EvalResult {
                index: sample.index,
                problem_type: sample.problem_type.clone(),
                predicted: vec![],
                ground_truth: sample.ground_truth.clone(),
                recall: 0.0,
                precision: 0.0,
                f1: 0.0,
                response: String::new(),
                error: Some(format!("{e:#}")),
            };
        }
    };

    let predicted = utils::extract_final_answer(extract_re, &response);
    let (recall, precision, f1) = utils::score(&predicted, &sample.ground_truth);

    EvalResult {
        index: sample.index,
        problem_type: sample.problem_type.clone(),
        predicted: predicted.into_iter().collect(),
        ground_truth: sample.ground_truth.clone(),
        recall,
        precision,
        f1,
        response,
        error: None,
    }
}

async fn call_api(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    prompt: &str,
) -> Result<String> {
    let req_body = ChatRequest {
        model: model.to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: prompt.to_owned(),
        }],
        temperature: Some(0.0),
    };

    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .header("Authorization", format!("Bearer {api_key}"))
        .json(&req_body)
        .send()
        .await
        .context("HTTP 请求失败")?;

    let status = resp.status();
    let body_text = resp.text().await.context("读取响应内容失败")?;

    if !status.is_success() {
        return Err(anyhow!("API 返回错误 {status}: {body_text}"));
    }

    let chat_resp: ChatResponse =
        serde_json::from_str(&body_text).context("解析 API 响应失败")?;

    let content = chat_resp
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    Ok(content)
}

// ── 输出 CSV ──────────────────────────────────────────────────────────────

fn write_csv(path: &PathBuf, results: &[EvalResult]) -> Result<()> {
    let mut w = File::create(path)
        .with_context(|| format!("无法创建文件: {}", path.display()))?;

    writeln!(w, "index,problem_type,recall,precision,f1,error,predicted,ground_truth")?;

    for r in results {
        let predicted = r.predicted.join(";");
        let truth: Vec<String> = r.ground_truth.iter().cloned().collect();
        let truth = truth.join(";");
        let error = r.error.as_deref().unwrap_or("");
        writeln!(
            w,
            "{},{},{:.4},{:.4},{:.4},{},{},{}",
            r.index,
            r.problem_type,
            r.recall,
            r.precision,
            r.f1,
            error,
            predicted,
            truth
        )?;
    }

    w.flush()?;
    Ok(())
}

// ── 汇总报告 ──────────────────────────────────────────────────────────────

fn print_summary(results: &[EvalResult]) {
    let total = results.len();
    let errors: Vec<_> = results.iter().filter(|r| r.error.is_some()).collect();
    let ok: Vec<_> = results.iter().filter(|r| r.error.is_none()).collect();

    let sum_recall: f64 = ok.iter().map(|r| r.recall).sum();
    let sum_precision: f64 = ok.iter().map(|r| r.precision).sum();
    let sum_f1: f64 = ok.iter().map(|r| r.f1).sum();
    let n_ok = ok.len();

    // 完全正确（F1 == 1.0）的样本数
    let exact_matches = ok.iter().filter(|r| r.f1 >= 0.999).count();

    println!("\n══════════════════════════════════════════");
    println!("  GraphWalks 评测汇总");
    println!("══════════════════════════════════════════");
    println!("  样本总数:       {total}");
    println!("  出错数:         {}", errors.len());
    println!("  完全正确:       {exact_matches}/{n_ok} ({:.2}%)",
        exact_matches as f64 / n_ok.max(1) as f64 * 100.0);

    if n_ok > 0 {
        println!("──────────────────────────────────────────");
        println!("  平均召回率:     {:.4}", sum_recall / n_ok as f64);
        println!("  平均精确率:     {:.4}", sum_precision / n_ok as f64);
        println!("  平均 F1:        {:.4}", sum_f1 / n_ok as f64);
    }

    // F1 分布
    let mut buckets = [0usize; 5]; // 0, (0,0.5), [0.5,0.8), [0.8,1.0), 1.0
    for r in &ok {
        if r.f1 < 0.001 {
            buckets[0] += 1;
        } else if r.f1 < 0.5 {
            buckets[1] += 1;
        } else if r.f1 < 0.8 {
            buckets[2] += 1;
        } else if r.f1 < 0.999 {
            buckets[3] += 1;
        } else {
            buckets[4] += 1;
        }
    }
    println!("──────────────────────────────────────────");
    println!("  F1 分布:");
    println!("    = 0:           {buckets0}", buckets0 = buckets[0]);
    println!("    (0, 0.5):      {buckets1}", buckets1 = buckets[1]);
    println!("    [0.5, 0.8):    {buckets2}", buckets2 = buckets[2]);
    println!("    [0.8, 1.0):    {buckets3}", buckets3 = buckets[3]);
    println!("    = 1.0:         {buckets4}", buckets4 = buckets[4]);

    // 按问题类型分组
    let mut types: std::collections::HashMap<&str, (usize, f64)> = std::collections::HashMap::new();
    for r in &ok {
        let entry = types.entry(&r.problem_type).or_default();
        entry.0 += 1;
        entry.1 += r.f1;
    }
    if !types.is_empty() {
        println!("──────────────────────────────────────────");
        println!("  按问题类型分组:");
        for (ptype, (count, sum_f1)) in &types {
            println!("    [{ptype}] 数量={count}  平均F1={:.4}", sum_f1 / *count as f64);
        }
    }

    // 出错和未完全正确的样本详情
    if errors.len() + (ok.len() - exact_matches) <= 50 {
        println!("\n──────────────────────────────────────────");
        println!("  出错/未完全正确的样本详情:");
        for r in results {
            if r.error.is_some() || r.f1 < 0.999 {
                println!();
                println!("  ── #{index} ──", index = r.index);
                if let Some(e) = &r.error {
                    println!("  错误: {e}");
                } else {
                    println!(
                        "  F1={f1:.4}  召回率={recall:.4}  精确率={precision:.4}",
                        f1 = r.f1,
                        recall = r.recall,
                        precision = r.precision
                    );
                    println!("  预测结果:  {predicted:?}", predicted = r.predicted);
                    println!(
                        "  正确结果:  {ground_truth:?}",
                        ground_truth = r.ground_truth
                    );
                }
            }
        }
    }

    println!("\n══════════════════════════════════════════");
}
