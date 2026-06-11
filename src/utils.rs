//! GraphWalks 评测工具箱
//!
//! 提供 parquet 列读取、评分计算等三个 bin 共用的工具函数。

use std::collections::HashSet;

use anyhow::{anyhow, Result};
use arrow::array::{Array, ListArray, StringArray};
use arrow::record_batch::RecordBatch;
use regex::Regex;

// ── parquet 列读取 ────────────────────────────────────────────────────────

/// 从 RecordBatch 中读取 Utf8 字符串列。
pub fn read_string_column(batch: &RecordBatch, idx: usize) -> Result<&StringArray> {
    let col = batch.column(idx);
    col
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("第 {idx} 列不是 Utf8 类型"))
}

/// 从 RecordBatch 中读取 `List<String>` 列，返回每行的字符串列表。
pub fn read_list_column(batch: &RecordBatch, idx: usize) -> Result<Vec<Vec<String>>> {
    let col = batch.column(idx);
    let list_arr = col
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| anyhow!("第 {idx} 列不是 List 类型"))?;

    let values_arr = list_arr.values();
    let values_str = values_arr
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| anyhow!("List 的值不是 Utf8 类型"))?;

    let mut result = Vec::with_capacity(list_arr.len());
    for i in 0..list_arr.len() {
        if list_arr.is_null(i) {
            result.push(Vec::new());
        } else {
            let start = list_arr.value_offsets()[i] as usize;
            let end = list_arr.value_offsets()[i + 1] as usize;
            let items: Vec<String> = (start..end)
                .map(|j| values_str.value(j).to_owned())
                .collect();
            result.push(items);
        }
    }
    Ok(result)
}

// ── 评分 ──────────────────────────────────────────────────────────────────

/// 计算 precision、recall、F1。
///
/// 完全按照 GraphWalks 官方公式：
/// - recall = |预测 ∩ 真实| / |真实|
/// - precision = |预测 ∩ 真实| / |预测|
/// - F1 = 2 * recall * precision / (recall + precision)
///
/// 当预测和真实均为空时，认为完全正确（F1 = 1.0）。
pub fn score(predicted: &HashSet<String>, ground_truth: &HashSet<String>) -> (f64, f64, f64) {
    let n_overlap = predicted.intersection(ground_truth).count();
    let n_pred = predicted.len();
    let n_truth = ground_truth.len();

    let recall = if n_truth > 0 {
        n_overlap as f64 / n_truth as f64
    } else {
        1.0 // 真实为空，不管预测什么都是完美的
    };

    let precision = if n_pred > 0 {
        n_overlap as f64 / n_pred as f64
    } else {
        1.0 // 预测为空，只有当真实也为空时才对（已由 recall 处理）
    };

    let f1 = if recall + precision > 0.0 {
        2.0 * recall * precision / (recall + precision)
    } else {
        0.0
    };

    (recall, precision, f1)
}

/// 从模型回复的最后一行提取 `Final Answer: [...]` 中的节点集合。
///
/// 与 GraphWalks 官方 Python 提取逻辑一致：
/// 只看最后一行，用正则 `Final Answer: ?[(.*?)]` 匹配。
pub fn extract_final_answer(re: &Regex, response: &str) -> HashSet<String> {
    let line = response.lines().last().unwrap_or("");
    if !line.contains("Final Answer:") {
        return HashSet::new();
    }

    if let Some(caps) = re.captures(line) {
        let list_str = caps.get(1).map_or("", |m| m.as_str());
        if list_str.trim().is_empty() {
            return HashSet::new();
        }
        list_str
            .split(',')
            .map(|s| s.trim().trim_matches(|c| c == '"' || c == '\'').to_owned())
            .filter(|s| !s.is_empty())
            .collect()
    } else {
        HashSet::new()
    }
}
