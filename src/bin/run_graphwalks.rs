use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use clap::Parser;
use futures::stream::{self, StreamExt};
use deepseek_graphwalks::utils;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;
use serde::{Deserialize, Serialize};
use arrow::array::Array;

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

// ── API 类型定义 ──────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<Thinking>,
}

#[derive(Serialize)]
struct Thinking {
    #[serde(rename = "type")]
    type_: String,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize, Debug)]
struct ChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<Usage>,
}

#[derive(Deserialize, Debug, Clone)]
struct Usage {
    completion_tokens: u32,
    prompt_tokens: u32,
    #[serde(default)]
    prompt_cache_hit_tokens: u32,
    #[serde(default)]
    prompt_cache_miss_tokens: u32,
    total_tokens: u32,
    #[serde(default)]
    completion_tokens_details: Option<CompletionTokensDetails>,
}

#[derive(Deserialize, Debug, Clone)]
struct CompletionTokensDetails {
    #[serde(default)]
    reasoning_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct Choice {
    message: ChoiceMessage,
}

#[derive(Deserialize, Debug)]
struct ChoiceMessage {
    content: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

// ── 样本 ─────────────────────────────────────────────────────────────────

struct Sample {
    /// 在 parquet 文件中的行号（从 0 开始）。
    index: usize,
    prompt: String,
    ground_truth: HashSet<String>,
    problem_type: String,
    /// 本地 tokenizer 预计算的输入 token 数（可能不存在）。
    local_input_tokens: Option<i32>,
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
    reasoning_content: Option<String>,
    usage: Option<Usage>,
    local_input_tokens: Option<i32>,
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
    let thinking_effort = Arc::new(args.thinking_effort);
    let extract_re = Arc::new(Regex::new("Final Answer: ?\\[(.*?)\\]").unwrap());

    let results: Vec<EvalResult> = stream::iter(samples.into_iter().map(|sample| {
        let client = client.clone();
        let base_url = base_url.clone();
        let model = model.clone();
        let api_key = api_key.clone();
        let pb = pb.clone();
        let thinking_effort = thinking_effort.clone();
        let extract_re = extract_re.clone();
        async move {
            let result = eval_one(&client, &base_url, &model, &api_key, thinking_effort.as_deref(), &extract_re, &sample).await;
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
    let token_idx = schema.index_of("deepseek_v4_input_tokens").ok();
    let mut samples = Vec::new();

    for batch_result in reader {
        let batch = batch_result?;
        let prompts = utils::read_string_column(&batch, prompt_idx)?;
        let problem_types = utils::read_string_column(&batch, problem_type_idx)?;
        let answer_lists = utils::read_list_column(&batch, answer_idx)?;
        let token_col = token_idx.map(|idx| {
            let col = batch.column(idx);
            col.as_any().downcast_ref::<arrow::array::Int32Array>()
                .ok_or_else(|| anyhow!("deepseek_v4_input_tokens 列不是 Int32 类型"))
        }).transpose()?;

        for (row, answer_list) in answer_lists.iter().enumerate() {
            if let Some(max) = args.max_samples {
                if samples.len() >= max {
                    return Ok(samples);
                }
            }
            let prompt = prompts.value(row).to_owned();
            let problem_type = problem_types.value(row).to_owned();
            let ground_truth: HashSet<String> = answer_list
                .iter()
                .map(|s| s.to_string())
                .collect();

            let local_input_tokens = token_col.and_then(|arr| if arr.is_null(row) { None } else { Some(arr.value(row)) });

            samples.push(Sample {
                index: samples.len(),
                prompt,
                ground_truth,
                problem_type,
                local_input_tokens,
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
    thinking_effort: Option<&str>,
    extract_re: &Regex,
    sample: &Sample,
) -> EvalResult {
    let (content, reasoning_content, usage) = match call_api(client, base_url, model, api_key, thinking_effort, &sample.prompt).await {
        Ok(r) => r,
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
                reasoning_content: None,
                usage: None,
                local_input_tokens: sample.local_input_tokens,
                error: Some(format!("{e:#}")),
            };
        }
    };

    let predicted = utils::extract_final_answer(extract_re, &content);
    let (recall, precision, f1) = utils::score(&predicted, &sample.ground_truth);

    EvalResult {
        index: sample.index,
        problem_type: sample.problem_type.clone(),
        predicted: predicted.into_iter().collect(),
        ground_truth: sample.ground_truth.clone(),
        recall,
        precision,
        f1,
        response: content,
        reasoning_content,
        usage,
        local_input_tokens: sample.local_input_tokens,
        error: None,
    }
}

async fn call_api(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    thinking_effort: Option<&str>,
    prompt: &str,
) -> Result<(String, Option<String>, Option<Usage>)> {
    let (reasoning_effort, thinking) = thinking_effort.map(|e| {
        (Some(e.to_owned()), Some(Thinking { type_: "enabled".to_owned() }))
    }).unwrap_or((None, None));

    let req_body = ChatRequest {
        model: model.to_owned(),
        messages: vec![Message {
            role: "user".to_owned(),
            content: prompt.to_owned(),
        }],
        temperature: Some(0.0),
        reasoning_effort,
        thinking,
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

    let choice = chat_resp
        .choices
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("API 返回了空的 choices"));

    Ok((
        choice.message.content.unwrap_or_default(),
        choice.message.reasoning_content,
        chat_resp.usage,
    ))
}

// ── 输出 CSV ──────────────────────────────────────────────────────────────

fn write_csv(path: &PathBuf, results: &[EvalResult]) -> Result<()> {
    let mut w = File::create(path)
        .with_context(|| format!("无法创建文件: {}", path.display()))?;

    writeln!(w, "index,problem_type,recall,precision,f1,error,response,reasoning_content,predicted,ground_truth,api_prompt_tokens,api_completion_tokens,api_total_tokens,api_cache_hit_tokens,api_cache_miss_tokens,api_reasoning_tokens,local_input_tokens")?;

    for r in results {
        let predicted = r.predicted.join(";");
        let truth: Vec<String> = r.ground_truth.iter().cloned().collect();
        let truth = truth.join(";");
        let error = r.error.as_deref().unwrap_or("");
        let response = csv_escape(&r.response);
        let reasoning = r.reasoning_content.as_deref().map(csv_escape).unwrap_or_default();
        let u = r.usage.as_ref();
        writeln!(
            w,
            "{},{},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{},{},{},{},{}",
            r.index,
            r.problem_type,
            r.recall,
            r.precision,
            r.f1,
            error,
            response,
            reasoning,
            predicted,
            truth,
            u.map_or(0, |u| u.prompt_tokens),
            u.map_or(0, |u| u.completion_tokens),
            u.map_or(0, |u| u.total_tokens),
            u.map_or(0, |u| u.prompt_cache_hit_tokens),
            u.map_or(0, |u| u.prompt_cache_miss_tokens),
            u.and_then(|u| u.completion_tokens_details.as_ref()).map_or(0, |d| d.reasoning_tokens),
            r.local_input_tokens.unwrap_or(-1),
        )?;
    }

    w.flush()?;
    Ok(())
}

/// 简单的 CSV 字段转义：用双引号包裹，内部换行替换为空格。
fn csv_escape(s: &str) -> String {
    let escaped = s.replace('\n', " ").replace('"', "\"\"");
    format!("\"{}\"", escaped)
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
