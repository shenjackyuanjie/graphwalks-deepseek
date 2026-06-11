//! GraphWalks 评估流程：样本加载、单条评估、CSV 输出、汇总报告。

use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{anyhow, Context, Result};
use arrow::array::Array;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use regex::Regex;

use crate::api::{call_api, Usage};
use crate::utils;

// ── 样本 ─────────────────────────────────────────────────────────────────

pub struct Sample {
    /// 在 parquet 文件中的行号（从 0 开始）。
    pub index: usize,
    pub prompt: String,
    pub ground_truth: HashSet<String>,
    pub problem_type: String,
    /// 本地 tokenizer 预计算的输入 token 数（可能不存在）。
    pub local_input_tokens: Option<i32>,
}

// ── 评分结果 ──────────────────────────────────────────────────────────────

pub struct EvalResult {
    pub index: usize,
    pub problem_type: String,
    pub predicted: Vec<String>,
    pub ground_truth: HashSet<String>,
    pub recall: f64,
    pub precision: f64,
    pub f1: f64,
    pub response: String,
    pub reasoning_content: Option<String>,
    pub usage: Option<Usage>,
    pub local_input_tokens: Option<i32>,
    pub error: Option<String>,
}

// ── 加载样本 ─────────────────────────────────────────────────────────────

/// 从 GraphWalks 格式的 parquet 文件加载样本。
pub fn load_samples(input: &Path, max_samples: Option<usize>) -> Result<Vec<Sample>> {
    let file =
        File::open(input).with_context(|| format!("无法打开文件: {}", input.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法读取 parquet: {}", input.display()))?;

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
        let token_col = token_idx
            .map(|idx| {
                let col = batch.column(idx);
                col.as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .ok_or_else(|| anyhow!("deepseek_v4_input_tokens 列不是 Int32 类型"))
            })
            .transpose()?;

        for (row, answer_list) in answer_lists.iter().enumerate() {
            if let Some(max) = max_samples {
                if samples.len() >= max {
                    return Ok(samples);
                }
            }
            let prompt = prompts.value(row).to_owned();
            let problem_type = problem_types.value(row).to_owned();
            let ground_truth: HashSet<String> =
                answer_list.iter().map(|s| s.to_string()).collect();

            let local_input_tokens = token_col.and_then(|arr| {
                if arr.is_null(row) {
                    None
                } else {
                    Some(arr.value(row))
                }
            });

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

pub async fn eval_one(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    api_key: &str,
    thinking_effort: Option<&str>,
    extract_re: &Regex,
    sample: &Sample,
) -> EvalResult {
    let (content, reasoning_content, usage) = match call_api(
        client,
        base_url,
        model,
        api_key,
        thinking_effort,
        &sample.prompt,
    )
    .await
    {
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

// ── 输出 CSV ──────────────────────────────────────────────────────────────

pub fn write_csv(path: &Path, results: &[EvalResult]) -> Result<()> {
    let mut w = File::create(path)
        .with_context(|| format!("无法创建文件: {}", path.display()))?;

    writeln!(w, "index,problem_type,recall,precision,f1,error,response,reasoning_content,predicted,ground_truth,api_prompt_tokens,api_completion_tokens,api_total_tokens,api_cache_hit_tokens,api_cache_miss_tokens,api_reasoning_tokens,local_input_tokens")?;

    for r in results {
        let predicted = r.predicted.join(";");
        let truth: Vec<String> = r.ground_truth.iter().cloned().collect();
        let truth = truth.join(";");
        let error = r.error.as_deref().unwrap_or("");
        let response = csv_escape(&r.response);
        let reasoning = r
            .reasoning_content
            .as_deref()
            .map(csv_escape)
            .unwrap_or_default();
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
            u.and_then(|u| u.completion_tokens_details.as_ref())
                .map_or(0, |d| d.reasoning_tokens),
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

pub fn print_summary(results: &[EvalResult]) {
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
    println!(
        "  完全正确:       {exact_matches}/{n_ok} ({:.2}%)",
        exact_matches as f64 / n_ok.max(1) as f64 * 100.0
    );

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
    let mut types: std::collections::HashMap<&str, (usize, f64)> =
        std::collections::HashMap::new();
    for r in &ok {
        let entry = types.entry(&r.problem_type).or_default();
        entry.0 += 1;
        entry.1 += r.f1;
    }
    if !types.is_empty() {
        println!("──────────────────────────────────────────");
        println!("  按问题类型分组:");
        for (ptype, (count, sum_f1)) in &types {
            println!(
                "    [{ptype}] 数量={count}  平均F1={:.4}",
                sum_f1 / *count as f64
            );
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
