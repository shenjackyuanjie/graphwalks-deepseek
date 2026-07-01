use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use arrow::array::{Array, Int32Array, LargeStringArray, StringArray};
use arrow::datatypes::DataType;
use clap::Parser;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;

#[derive(Clone, Debug)]
struct DatasetSpec {
    label: String,
    v4p: PathBuf,
    v32: PathBuf,
    openpangu: PathBuf,
}

impl FromStr for DatasetSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (label, paths) = value
            .split_once('=')
            .context("数据集参数格式应为 LABEL=V4P,V32,OPENPANGU")?;
        let paths: Vec<_> = paths.split(',').collect();
        if label.is_empty() || paths.len() != 3 || paths.iter().any(|path| path.is_empty()) {
            bail!("数据集参数格式应为 LABEL=V4P,V32,OPENPANGU，实际为 {value:?}");
        }
        Ok(Self {
            label: label.to_owned(),
            v4p: paths[0].into(),
            v32: paths[1].into(),
            openpangu: paths[2].into(),
        })
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// 数据集及三份计数文件，格式为 LABEL=V4P,V32,OPENPANGU；可重复传入。
    #[arg(long = "dataset", required = true)]
    datasets: Vec<DatasetSpec>,

    #[arg(long)]
    title: String,

    #[arg(long)]
    output: PathBuf,

    /// 可选分组列，从 V4P 文件读取并按原始行顺序对齐。
    #[arg(long)]
    group_col: Option<String>,

    #[arg(long, default_value = "deepseek_v4_input_tokens")]
    v4p_col: String,

    #[arg(long, default_value = "deepseek_v32_input_tokens")]
    v32_col: String,

    #[arg(long, default_value = "openpangu_2_0_flash_input_tokens")]
    openpangu_col: String,

    #[arg(long, default_value_t = 8192)]
    batch_size: usize,
}

#[derive(Debug)]
struct Item {
    dataset: String,
    group: Option<String>,
    v4p: i64,
    v32: i64,
    openpangu: i64,
}

#[derive(Clone, Copy)]
enum CountKind {
    V4p,
    V32,
    OpenPangu,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut items = Vec::new();
    for dataset in &args.datasets {
        items.extend(read_dataset(dataset, &args)?);
    }
    if items.is_empty() {
        bail!("输入数据集没有记录");
    }

    let report = render_report(&args, &items);
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建报告目录: {}", parent.display()))?;
    }
    fs::write(&args.output, report)
        .with_context(|| format!("无法写入报告: {}", args.output.display()))?;
    println!(
        "已对齐统计 {} 条记录，报告写入 {}",
        items.len(),
        args.output.display()
    );
    Ok(())
}

fn read_dataset(dataset: &DatasetSpec, args: &Args) -> Result<Vec<Item>> {
    let v4p = read_i32_column(&dataset.v4p, &args.v4p_col, args.batch_size)?;
    let v32 = read_i32_column(&dataset.v32, &args.v32_col, args.batch_size)?;
    let openpangu = read_i32_column(&dataset.openpangu, &args.openpangu_col, args.batch_size)?;
    let groups = args
        .group_col
        .as_deref()
        .map(|column| read_string_column(&dataset.v4p, column, args.batch_size))
        .transpose()?;

    let expected = v4p.len();
    if v32.len() != expected
        || openpangu.len() != expected
        || groups
            .as_ref()
            .is_some_and(|values| values.len() != expected)
    {
        bail!(
            "数据集 {:?} 行数不一致: V4P={}, V3.2={}, OpenPangu={}, group={:?}",
            dataset.label,
            v4p.len(),
            v32.len(),
            openpangu.len(),
            groups.as_ref().map(Vec::len)
        );
    }

    Ok((0..expected)
        .map(|index| Item {
            dataset: dataset.label.clone(),
            group: groups.as_ref().map(|values| values[index].clone()),
            v4p: i64::from(v4p[index]),
            v32: i64::from(v32[index]),
            openpangu: i64::from(openpangu[index]),
        })
        .collect())
}

fn read_i32_column(path: &Path, column: &str, batch_size: usize) -> Result<Vec<i32>> {
    let file = File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法读取 parquet schema: {}", path.display()))?;
    let index = builder
        .schema()
        .index_of(column)
        .with_context(|| format!("{} 中缺少列 {column:?}", path.display()))?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [index]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()?;
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch?;
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .with_context(|| format!("{} 的 {column:?} 不是 Int32", path.display()))?;
        for index in 0..array.len() {
            if array.is_null(index) {
                bail!(
                    "{} 的 {column:?} 第 {} 行为空",
                    path.display(),
                    values.len()
                );
            }
            let value = array.value(index);
            if value < 0 {
                bail!("{} 的 {column:?} 包含无效计数 {value}", path.display());
            }
            values.push(value);
        }
    }
    Ok(values)
}

fn read_string_column(path: &Path, column: &str, batch_size: usize) -> Result<Vec<String>> {
    let file = File::open(path).with_context(|| format!("无法打开 {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("无法读取 parquet schema: {}", path.display()))?;
    let index = builder
        .schema()
        .index_of(column)
        .with_context(|| format!("{} 中缺少列 {column:?}", path.display()))?;
    let mask = ProjectionMask::roots(builder.parquet_schema(), [index]);
    let reader = builder
        .with_projection(mask)
        .with_batch_size(batch_size)
        .build()?;
    let mut values = Vec::new();
    for batch in reader {
        let batch = batch?;
        let array = batch.column(0);
        match array.data_type() {
            DataType::Utf8 => append_strings(
                array
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .context("Utf8 类型转换失败")?,
                path,
                column,
                &mut values,
            )?,
            DataType::LargeUtf8 => append_large_strings(
                array
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .context("LargeUtf8 类型转换失败")?,
                path,
                column,
                &mut values,
            )?,
            actual => bail!(
                "{} 的 {column:?} 不是字符串列，而是 {actual}",
                path.display()
            ),
        }
    }
    Ok(values)
}

fn append_strings(
    array: &StringArray,
    path: &Path,
    column: &str,
    output: &mut Vec<String>,
) -> Result<()> {
    for index in 0..array.len() {
        if array.is_null(index) {
            bail!(
                "{} 的 {column:?} 第 {} 行为空",
                path.display(),
                output.len()
            );
        }
        output.push(array.value(index).to_owned());
    }
    Ok(())
}

fn append_large_strings(
    array: &LargeStringArray,
    path: &Path,
    column: &str,
    output: &mut Vec<String>,
) -> Result<()> {
    for index in 0..array.len() {
        if array.is_null(index) {
            bail!(
                "{} 的 {column:?} 第 {} 行为空",
                path.display(),
                output.len()
            );
        }
        output.push(array.value(index).to_owned());
    }
    Ok(())
}

fn render_report(args: &Args, items: &[Item]) -> String {
    let mut output = String::new();
    writeln!(output, "# {}\n", args.title).unwrap();
    render_method(args, items, &mut output);
    render_overall(items, &mut output);
    render_datasets(items, &mut output);
    if args.group_col.is_some() {
        render_groups(items, &mut output);
    }
    render_buckets(items, &mut output);
    render_quantiles(items, &mut output);
    render_thresholds(items, &mut output);
    render_directions(items, &mut output);
    render_regressions(items, &mut output);
    output
}

fn render_method(args: &Args, items: &[Item], output: &mut String) {
    writeln!(output, "## 统计口径\n").unwrap();
    writeln!(output, "| 项目 | 值 |").unwrap();
    writeln!(output, "|---|---|").unwrap();
    writeln!(output, "| 对齐记录数 | {} |", items.len()).unwrap();
    writeln!(
        output,
        "| 对齐方式 | 同一数据集三份计数文件按原始行顺序对齐，并校验行数 |"
    )
    .unwrap();
    writeln!(
        output,
        "| 编码方式 | `Tokenizer::encode(text, false)`，不添加 special tokens |"
    )
    .unwrap();
    writeln!(output, "| V4P 计数列 | `{}` |", escape(&args.v4p_col)).unwrap();
    writeln!(output, "| V3.2 计数列 | `{}` |", escape(&args.v32_col)).unwrap();
    writeln!(
        output,
        "| OpenPangu 计数列 | `{}` |",
        escape(&args.openpangu_col)
    )
    .unwrap();
    writeln!(
        output,
        "| 分组列 | {} |\n",
        args.group_col
            .as_deref()
            .map(|column| format!("`{}`", escape(column)))
            .unwrap_or_else(|| "无".to_owned())
    )
    .unwrap();
}

fn render_overall(items: &[Item], output: &mut String) {
    let v4p_total = total(items, CountKind::V4p);
    writeln!(output, "## 总体结果\n").unwrap();
    writeln!(output, "| Tokenizer | 总 token 数 | 平均每条 | 中位数 | 最小 | 最大 | 相对 V4P 差值 | 相对 V4P 增幅 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, kind) in tokenizer_kinds() {
        let values = counts(items, kind);
        let sum: i64 = values.iter().sum();
        writeln!(
            output,
            "| {name} | {} | {:.2} | {:.0} | {} | {} | {:+} | {:+.2}% |",
            integer(sum),
            sum as f64 / items.len() as f64,
            percentile(&values, 0.5),
            integer(*values.iter().min().unwrap()),
            integer(*values.iter().max().unwrap()),
            sum - v4p_total,
            percent(sum - v4p_total, v4p_total),
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_datasets(items: &[Item], output: &mut String) {
    writeln!(output, "## 按数据集\n").unwrap();
    writeln!(
        output,
        "| 数据集 | 条数 | V4P 总量 | V3.2 总量 | OpenPangu 总量 | V3.2 / V4P | OpenPangu / V4P |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for (dataset, subset) in group_by(items, |item| item.dataset.clone()) {
        render_group_row(&dataset, &subset, output);
    }
    render_group_row("全部", &items.iter().collect::<Vec<_>>(), output);
    output.push('\n');
}

fn render_groups(items: &[Item], output: &mut String) {
    writeln!(output, "## 按分组列\n").unwrap();
    writeln!(
        output,
        "| 分组 | 条数 | V4P 总量 | V3.2 总量 | OpenPangu 总量 | V3.2 / V4P | OpenPangu / V4P |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for (group, subset) in group_by(items, |item| item.group.clone().unwrap_or_default()) {
        render_group_row(&group, &subset, output);
    }
    output.push('\n');
}

fn render_group_row(label: &str, items: &[&Item], output: &mut String) {
    let v4p: i64 = items.iter().map(|item| item.v4p).sum();
    let v32: i64 = items.iter().map(|item| item.v32).sum();
    let openpangu: i64 = items.iter().map(|item| item.openpangu).sum();
    writeln!(
        output,
        "| {} | {} | {} | {} | {} | {:.5}x | {:.5}x |",
        escape(label),
        items.len(),
        integer(v4p),
        integer(v32),
        integer(openpangu),
        ratio(v32, v4p),
        ratio(openpangu, v4p),
    )
    .unwrap();
}

fn render_buckets(items: &[Item], output: &mut String) {
    const BOUNDS: [i64; 9] = [
        2_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000,
    ];
    writeln!(output, "## 按 V4P token 长度\n").unwrap();
    writeln!(output, "| V4P 区间 | 条数 | V4P 平均 | V3.2 平均 | OpenPangu 平均 | V3.2 / V4P | OpenPangu / V4P |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    for bucket in 0..=BOUNDS.len() {
        let subset: Vec<_> = items
            .iter()
            .filter(|item| bucket_index(item.v4p, &BOUNDS) == bucket)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let v4p: i64 = subset.iter().map(|item| item.v4p).sum();
        let v32: i64 = subset.iter().map(|item| item.v32).sum();
        let openpangu: i64 = subset.iter().map(|item| item.openpangu).sum();
        writeln!(
            output,
            "| {} | {} | {:.0} | {:.0} | {:.0} | {:.5}x | {:.5}x |",
            bucket_label(bucket, &BOUNDS),
            subset.len(),
            v4p as f64 / subset.len() as f64,
            v32 as f64 / subset.len() as f64,
            openpangu as f64 / subset.len() as f64,
            ratio(v32, v4p),
            ratio(openpangu, v4p),
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_quantiles(items: &[Item], output: &mut String) {
    let v32_deltas: Vec<i64> = items.iter().map(|item| item.v32 - item.v4p).collect();
    let v32_ratios: Vec<f64> = items.iter().map(|item| ratio(item.v32, item.v4p)).collect();
    let pangu_deltas: Vec<i64> = items.iter().map(|item| item.openpangu - item.v4p).collect();
    let pangu_ratios: Vec<f64> = items
        .iter()
        .map(|item| ratio(item.openpangu, item.v4p))
        .collect();
    writeln!(output, "## 逐条差异分位数\n").unwrap();
    writeln!(
        output,
        "| 分位数 | V3.2 - V4P | V3.2 / V4P | OpenPangu - V4P | OpenPangu / V4P |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|").unwrap();
    for (label, quantile) in quantiles() {
        writeln!(
            output,
            "| {label} | {:+.0} | {:.5}x | {:+.0} | {:.5}x |",
            percentile(&v32_deltas, quantile),
            percentile(&v32_ratios, quantile),
            percentile(&pangu_deltas, quantile),
            percentile(&pangu_ratios, quantile),
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_thresholds(items: &[Item], output: &mut String) {
    writeln!(output, "## 上下文阈值\n").unwrap();
    writeln!(output, "| 阈值 | V4P 超过数 | V3.2 超过数 | OpenPangu 超过数 | V3.2 相对 V4P 新增 | OpenPangu 相对 V4P 新增 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|").unwrap();
    for threshold in [128_000, 256_000, 512_000, 1_000_000] {
        let v4p = items.iter().filter(|item| item.v4p > threshold).count();
        let v32 = items.iter().filter(|item| item.v32 > threshold).count();
        let openpangu = items
            .iter()
            .filter(|item| item.openpangu > threshold)
            .count();
        writeln!(
            output,
            "| {} | {v4p} | {v32} | {openpangu} | {:+} | {:+} |",
            integer(threshold),
            v32 as i64 - v4p as i64,
            openpangu as i64 - v4p as i64
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_directions(items: &[Item], output: &mut String) {
    writeln!(output, "## 逐条大小关系\n").unwrap();
    writeln!(output, "| 对比 | 小于 V4P | 等于 V4P | 大于 V4P |").unwrap();
    writeln!(output, "|---|---:|---:|---:|").unwrap();
    for (name, kind) in [
        ("V3.2", CountKind::V32),
        ("OpenPangu", CountKind::OpenPangu),
    ] {
        let values = counts(items, kind);
        let baselines = counts(items, CountKind::V4p);
        let less = values
            .iter()
            .zip(&baselines)
            .filter(|(value, base)| value < base)
            .count();
        let equal = values
            .iter()
            .zip(&baselines)
            .filter(|(value, base)| value == base)
            .count();
        let greater = values.len() - less - equal;
        writeln!(output, "| {name} | {less} | {equal} | {greater} |").unwrap();
    }
    output.push('\n');
}

fn render_regressions(items: &[Item], output: &mut String) {
    let x: Vec<f64> = items.iter().map(|item| item.v4p as f64).collect();
    writeln!(output, "## 相对 V4P 线性拟合\n").unwrap();
    writeln!(output, "| Tokenizer | 斜率 | 截距 | R² | 估算式 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---|").unwrap();
    for (name, kind) in [
        ("V3.2", CountKind::V32),
        ("OpenPangu", CountKind::OpenPangu),
    ] {
        let y: Vec<f64> = counts(items, kind)
            .into_iter()
            .map(|value| value as f64)
            .collect();
        let (slope, intercept, r_squared) = linear_regression(&x, &y);
        writeln!(output, "| {name} | {slope:.8} | {intercept:+.2} | {r_squared:.9} | `{name} ≈ {slope:.5} × V4P {intercept:+.0}` |").unwrap();
    }
}

fn tokenizer_kinds() -> [(&'static str, CountKind); 3] {
    [
        ("V4P", CountKind::V4p),
        ("V3.2", CountKind::V32),
        ("OpenPangu 2.0 Flash", CountKind::OpenPangu),
    ]
}

fn counts(items: &[Item], kind: CountKind) -> Vec<i64> {
    items
        .iter()
        .map(|item| match kind {
            CountKind::V4p => item.v4p,
            CountKind::V32 => item.v32,
            CountKind::OpenPangu => item.openpangu,
        })
        .collect()
}

fn total(items: &[Item], kind: CountKind) -> i64 {
    counts(items, kind).iter().sum()
}

fn group_by<F>(items: &[Item], key: F) -> BTreeMap<String, Vec<&Item>>
where
    F: Fn(&Item) -> String,
{
    let mut groups = BTreeMap::new();
    for item in items {
        groups.entry(key(item)).or_insert_with(Vec::new).push(item);
    }
    groups
}

fn bucket_index(value: i64, bounds: &[i64]) -> usize {
    bounds
        .iter()
        .position(|bound| value < *bound)
        .unwrap_or(bounds.len())
}

fn bucket_label(index: usize, bounds: &[i64]) -> String {
    if index == 0 {
        format!("<{}", compact(bounds[0]))
    } else if index == bounds.len() {
        format!(">={}", compact(bounds[index - 1]))
    } else {
        format!("{}–{}", compact(bounds[index - 1]), compact(bounds[index]))
    }
}

fn compact(value: i64) -> String {
    if value % 1_000_000 == 0 {
        format!("{}M", value / 1_000_000)
    } else if value % 1_000 == 0 {
        format!("{}k", value / 1_000)
    } else {
        value.to_string()
    }
}

fn quantiles() -> [(&'static str, f64); 8] {
    [
        ("最小", 0.0),
        ("P10", 0.1),
        ("P25", 0.25),
        ("P50", 0.5),
        ("P75", 0.75),
        ("P90", 0.9),
        ("P99", 0.99),
        ("最大", 1.0),
    ]
}

trait IntoF64 {
    fn into_f64(self) -> f64;
}
impl IntoF64 for i64 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}
impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

fn percentile<T: Copy + IntoF64>(values: &[T], quantile: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().map(|value| (*value).into_f64()).collect();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower as f64)
}

fn ratio(value: i64, baseline: i64) -> f64 {
    if baseline == 0 {
        f64::NAN
    } else {
        value as f64 / baseline as f64
    }
}

fn percent(difference: i64, baseline: i64) -> f64 {
    if baseline == 0 {
        f64::NAN
    } else {
        difference as f64 / baseline as f64 * 100.0
    }
}

fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64, f64) {
    let count = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / count;
    let y_mean = y.iter().sum::<f64>() / count;
    let covariance: f64 = x
        .iter()
        .zip(y)
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum();
    let x_variance: f64 = x.iter().map(|x| (x - x_mean).powi(2)).sum();
    let slope = covariance / x_variance;
    let intercept = y_mean - slope * x_mean;
    let residual: f64 = x
        .iter()
        .zip(y)
        .map(|(x, y)| (y - (slope * x + intercept)).powi(2))
        .sum();
    let total: f64 = y.iter().map(|y| (y - y_mean).powi(2)).sum();
    (slope, intercept, 1.0 - residual / total)
}

fn integer(value: i64) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let mut output = String::new();
    if negative {
        output.push('-');
    }
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            output.push(',');
        }
        output.push(character);
    }
    output
}

fn escape(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_dataset_spec() {
        let spec: DatasetSpec = "short=a.parquet,b.parquet,c.parquet".parse().unwrap();
        assert_eq!(spec.label, "short");
        assert_eq!(spec.v32, PathBuf::from("b.parquet"));
    }

    #[test]
    fn rejects_incomplete_dataset_spec() {
        assert!("short=a.parquet,b.parquet".parse::<DatasetSpec>().is_err());
    }

    #[test]
    fn labels_length_buckets() {
        let bounds = [2_000, 8_000];
        assert_eq!(bucket_label(0, &bounds), "<2k");
        assert_eq!(bucket_label(1, &bounds), "2k–8k");
        assert_eq!(bucket_label(2, &bounds), ">=8k");
    }

    #[test]
    fn regression_recovers_line() {
        let (slope, intercept, r_squared) = linear_regression(&[1.0, 2.0, 3.0], &[3.0, 5.0, 7.0]);
        assert!((slope - 2.0).abs() < 1e-10);
        assert!((intercept - 1.0).abs() < 1e-10);
        assert!((r_squared - 1.0).abs() < 1e-10);
    }
}
