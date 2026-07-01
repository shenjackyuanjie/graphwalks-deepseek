use std::collections::{BTreeMap, HashSet};
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
struct TokenizerSpec {
    name: String,
    column: String,
}

impl FromStr for TokenizerSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (name, column) = value
            .split_once('=')
            .context("tokenizer 参数格式应为 NAME=TOKEN_COUNT_COLUMN")?;
        if name.trim().is_empty() || column.is_empty() {
            bail!("tokenizer 参数格式应为 NAME=TOKEN_COUNT_COLUMN，实际为 {value:?}");
        }
        Ok(Self {
            name: name.trim().to_owned(),
            column: column.to_owned(),
        })
    }
}

#[derive(Clone, Debug)]
struct DatasetSpec {
    label: String,
    paths: Vec<PathBuf>,
}

impl FromStr for DatasetSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (label, paths) = value
            .split_once('=')
            .context("数据集参数格式应为 LABEL=TOKENIZER_FILE,...")?;
        let paths: Vec<PathBuf> = paths.split(',').map(PathBuf::from).collect();
        if label.trim().is_empty()
            || paths.is_empty()
            || paths.iter().any(|path| path.as_os_str().is_empty())
        {
            bail!("数据集参数格式应为 LABEL=TOKENIZER_FILE,...，实际为 {value:?}");
        }
        Ok(Self {
            label: label.trim().to_owned(),
            paths,
        })
    }
}

#[derive(Parser, Debug)]
struct Args {
    /// Tokenizer 与计数列定义，可重复传入，格式为 NAME=TOKEN_COUNT_COLUMN。
    #[arg(long = "tokenizer", required = true)]
    tokenizers: Vec<TokenizerSpec>,

    /// 每个数据集按 --tokenizer 的顺序列出计数文件。
    #[arg(long = "dataset", required = true)]
    datasets: Vec<DatasetSpec>,

    /// 主对比 tokenizer 名称；省略时使用第一个 --tokenizer。
    #[arg(long)]
    primary: Option<String>,

    #[arg(long)]
    title: String,

    #[arg(long)]
    output: PathBuf,

    #[arg(long)]
    group_col: Option<String>,

    #[arg(long, default_value_t = 8192)]
    batch_size: usize,
}

#[derive(Debug)]
struct Item {
    dataset: String,
    group: Option<String>,
    counts: Vec<i64>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let primary = validate_args(&args)?;
    let mut items = Vec::new();
    for dataset in &args.datasets {
        items.extend(read_dataset(dataset, &args)?);
    }
    if items.is_empty() {
        bail!("输入数据集没有记录");
    }

    let report = render_report(&args, &items, primary);
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建报告目录: {}", parent.display()))?;
    }
    fs::write(&args.output, report)
        .with_context(|| format!("无法写入报告: {}", args.output.display()))?;
    println!(
        "已使用 {} 个 tokenizer 对齐统计 {} 条记录，报告写入 {}",
        args.tokenizers.len(),
        items.len(),
        args.output.display()
    );
    Ok(())
}

fn validate_args(args: &Args) -> Result<usize> {
    if args.tokenizers.len() < 2 {
        bail!("至少需要传入两个 --tokenizer");
    }
    let mut names = HashSet::new();
    let mut columns = HashSet::new();
    for spec in &args.tokenizers {
        if !names.insert(&spec.name) {
            bail!("tokenizer 名称重复: {:?}", spec.name);
        }
        if !columns.insert(&spec.column) {
            bail!("token count 列名重复: {:?}", spec.column);
        }
    }
    for dataset in &args.datasets {
        if dataset.paths.len() != args.tokenizers.len() {
            bail!(
                "数据集 {:?} 有 {} 个文件，但定义了 {} 个 tokenizer",
                dataset.label,
                dataset.paths.len(),
                args.tokenizers.len()
            );
        }
    }
    match args.primary.as_deref() {
        Some(name) => args
            .tokenizers
            .iter()
            .position(|spec| spec.name == name)
            .with_context(|| format!("primary {name:?} 不在 tokenizer 列表中")),
        None => Ok(0),
    }
}

fn read_dataset(dataset: &DatasetSpec, args: &Args) -> Result<Vec<Item>> {
    let columns: Vec<Vec<i32>> = dataset
        .paths
        .iter()
        .zip(&args.tokenizers)
        .map(|(path, spec)| read_i32_column(path, &spec.column, args.batch_size))
        .collect::<Result<_>>()?;
    let expected = columns[0].len();
    for (index, values) in columns.iter().enumerate().skip(1) {
        if values.len() != expected {
            bail!(
                "数据集 {:?} 行数不一致: {}={}, {}={}",
                dataset.label,
                args.tokenizers[0].name,
                expected,
                args.tokenizers[index].name,
                values.len()
            );
        }
    }
    let groups = args
        .group_col
        .as_deref()
        .map(|column| read_string_column(&dataset.paths[0], column, args.batch_size))
        .transpose()?;
    if groups
        .as_ref()
        .is_some_and(|values| values.len() != expected)
    {
        bail!("数据集 {:?} 的分组列行数不一致", dataset.label);
    }

    Ok((0..expected)
        .map(|row| Item {
            dataset: dataset.label.clone(),
            group: groups.as_ref().map(|values| values[row].clone()),
            counts: columns
                .iter()
                .map(|values| i64::from(values[row]))
                .collect(),
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

fn render_report(args: &Args, items: &[Item], primary: usize) -> String {
    let mut output = String::new();
    writeln!(output, "# {}\n", args.title).unwrap();
    render_method(args, items, primary, &mut output);
    render_overall(&args.tokenizers, items, primary, &mut output);
    render_dataset_groups(
        "按数据集",
        &args.tokenizers,
        items,
        primary,
        |item| item.dataset.clone(),
        &mut output,
    );
    if args.group_col.is_some() {
        render_dataset_groups(
            "按分组列",
            &args.tokenizers,
            items,
            primary,
            |item| item.group.clone().unwrap_or_default(),
            &mut output,
        );
    }
    render_buckets(&args.tokenizers, items, primary, &mut output);
    render_quantiles(&args.tokenizers, items, primary, &mut output);
    render_thresholds(&args.tokenizers, items, primary, &mut output);
    render_directions(&args.tokenizers, items, primary, &mut output);
    render_regressions(&args.tokenizers, items, primary, &mut output);
    render_perspectives(&args.tokenizers, items, &mut output);
    output
}

fn render_method(args: &Args, items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 统计口径\n").unwrap();
    writeln!(output, "| 项目 | 值 |").unwrap();
    writeln!(output, "|---|---|").unwrap();
    writeln!(output, "| 对齐记录数 | {} |", items.len()).unwrap();
    writeln!(output, "| Tokenizer 数 | {} |", args.tokenizers.len()).unwrap();
    writeln!(
        output,
        "| 主对比 tokenizer | {} |",
        escape(&args.tokenizers[primary].name)
    )
    .unwrap();
    writeln!(
        output,
        "| 对齐方式 | 同一数据集各计数文件按原始行顺序对齐，并校验行数 |"
    )
    .unwrap();
    writeln!(
        output,
        "| 编码方式 | `Tokenizer::encode(text, false)`，不添加 special tokens |"
    )
    .unwrap();
    for spec in &args.tokenizers {
        writeln!(
            output,
            "| {} 计数列 | `{}` |",
            escape(&spec.name),
            escape(&spec.column)
        )
        .unwrap();
    }
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

fn render_overall(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    let primary_total = total(items, primary);
    writeln!(output, "## 总体结果\n").unwrap();
    writeln!(output, "| Tokenizer | 总 token 数 | 平均每条 | 中位数 | 最小 | 最大 | 相对主 tokenizer 差值 | 相对主 tokenizer 增幅 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (index, spec) in specs.iter().enumerate() {
        let values = counts(items, index);
        let sum: i64 = values.iter().sum();
        writeln!(
            output,
            "| {} | {} | {:.2} | {:.0} | {} | {} | {:+} | {:+.2}% |",
            escape(&spec.name),
            integer(sum),
            sum as f64 / items.len() as f64,
            percentile(&values, 0.5),
            integer(*values.iter().min().unwrap()),
            integer(*values.iter().max().unwrap()),
            sum - primary_total,
            percent(sum - primary_total, primary_total)
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_dataset_groups<F>(
    title: &str,
    specs: &[TokenizerSpec],
    items: &[Item],
    primary: usize,
    key: F,
    output: &mut String,
) where
    F: Fn(&Item) -> String,
{
    writeln!(output, "## {title}\n").unwrap();
    write!(output, "| 分组 | 条数").unwrap();
    for spec in specs {
        write!(output, " | {} 总量", escape(&spec.name)).unwrap();
    }
    for (index, spec) in specs.iter().enumerate() {
        if index != primary {
            write!(
                output,
                " | {} / {}",
                escape(&spec.name),
                escape(&specs[primary].name)
            )
            .unwrap();
        }
    }
    writeln!(output, " |").unwrap();
    write!(output, "|---|---:").unwrap();
    for _ in specs {
        write!(output, "|---:").unwrap();
    }
    for index in 0..specs.len() {
        if index != primary {
            write!(output, "|---:").unwrap();
        }
    }
    writeln!(output, "|").unwrap();

    for (label, subset) in group_by(items, key) {
        render_group_row(&label, specs.len(), &subset, primary, output);
    }
    render_group_row(
        "全部",
        specs.len(),
        &items.iter().collect::<Vec<_>>(),
        primary,
        output,
    );
    output.push('\n');
}

fn render_group_row(
    label: &str,
    tokenizer_count: usize,
    items: &[&Item],
    primary: usize,
    output: &mut String,
) {
    let totals: Vec<i64> = (0..tokenizer_count)
        .map(|index| items.iter().map(|item| item.counts[index]).sum())
        .collect();
    write!(output, "| {} | {}", escape(label), items.len()).unwrap();
    for value in &totals {
        write!(output, " | {}", integer(*value)).unwrap();
    }
    for (index, value) in totals.iter().enumerate() {
        if index != primary {
            write!(output, " | {:.5}x", ratio(*value, totals[primary])).unwrap();
        }
    }
    writeln!(output, " |").unwrap();
}

fn render_buckets(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    const BOUNDS: [i64; 9] = [
        2_000, 8_000, 16_000, 32_000, 64_000, 128_000, 256_000, 512_000, 1_000_000,
    ];
    writeln!(
        output,
        "## 按 {} token 长度\n",
        escape(&specs[primary].name)
    )
    .unwrap();
    write!(output, "| {} 区间 | 条数", escape(&specs[primary].name)).unwrap();
    for spec in specs {
        write!(output, " | {} 平均", escape(&spec.name)).unwrap();
    }
    for (index, spec) in specs.iter().enumerate() {
        if index != primary {
            write!(
                output,
                " | {} / {}",
                escape(&spec.name),
                escape(&specs[primary].name)
            )
            .unwrap();
        }
    }
    writeln!(output, " |").unwrap();
    write!(output, "|---|---:").unwrap();
    for _ in specs {
        write!(output, "|---:").unwrap();
    }
    for index in 0..specs.len() {
        if index != primary {
            write!(output, "|---:").unwrap();
        }
    }
    writeln!(output, "|").unwrap();

    for bucket in 0..=BOUNDS.len() {
        let subset: Vec<_> = items
            .iter()
            .filter(|item| bucket_index(item.counts[primary], &BOUNDS) == bucket)
            .collect();
        if subset.is_empty() {
            continue;
        }
        let totals: Vec<i64> = (0..specs.len())
            .map(|index| subset.iter().map(|item| item.counts[index]).sum())
            .collect();
        write!(
            output,
            "| {} | {}",
            bucket_label(bucket, &BOUNDS),
            subset.len()
        )
        .unwrap();
        for total in &totals {
            write!(output, " | {:.0}", *total as f64 / subset.len() as f64).unwrap();
        }
        for (index, total) in totals.iter().enumerate() {
            if index != primary {
                write!(output, " | {:.5}x", ratio(*total, totals[primary])).unwrap();
            }
        }
        writeln!(output, " |").unwrap();
    }
    output.push('\n');
}

fn render_quantiles(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 逐条差异分位数\n").unwrap();
    writeln!(
        output,
        "| Tokenizer | 分位数 | 相对主 tokenizer 差值 | 相对主 tokenizer 比例 |"
    )
    .unwrap();
    writeln!(output, "|---|---|---:|---:|").unwrap();
    for (index, spec) in specs.iter().enumerate() {
        if index == primary {
            continue;
        }
        let deltas: Vec<i64> = items
            .iter()
            .map(|item| item.counts[index] - item.counts[primary])
            .collect();
        let ratios: Vec<f64> = items
            .iter()
            .map(|item| ratio(item.counts[index], item.counts[primary]))
            .collect();
        for (label, quantile) in quantiles() {
            writeln!(
                output,
                "| {} | {label} | {:+.0} | {:.5}x |",
                escape(&spec.name),
                percentile(&deltas, quantile),
                percentile(&ratios, quantile)
            )
            .unwrap();
        }
    }
    output.push('\n');
}

fn render_thresholds(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 上下文阈值\n").unwrap();
    writeln!(
        output,
        "| 阈值 | Tokenizer | 超过数 | 相对主 tokenizer 增减 |"
    )
    .unwrap();
    writeln!(output, "|---:|---|---:|---:|").unwrap();
    for threshold in [128_000, 256_000, 512_000, 1_000_000] {
        let primary_count = items
            .iter()
            .filter(|item| item.counts[primary] > threshold)
            .count();
        for (index, spec) in specs.iter().enumerate() {
            let count = items
                .iter()
                .filter(|item| item.counts[index] > threshold)
                .count();
            writeln!(
                output,
                "| {} | {} | {count} | {:+} |",
                integer(threshold),
                escape(&spec.name),
                count as i64 - primary_count as i64
            )
            .unwrap();
        }
    }
    output.push('\n');
}

fn render_directions(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 相对主 tokenizer 的逐条大小关系\n").unwrap();
    writeln!(
        output,
        "| Tokenizer | 小于主 tokenizer | 等于主 tokenizer | 大于主 tokenizer |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|").unwrap();
    for (index, spec) in specs.iter().enumerate() {
        if index == primary {
            continue;
        }
        let (less, equal, more) = directions(items, index, primary);
        writeln!(
            output,
            "| {} | {less} | {equal} | {more} |",
            escape(&spec.name)
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_regressions(
    specs: &[TokenizerSpec],
    items: &[Item],
    primary: usize,
    output: &mut String,
) {
    let x: Vec<f64> = items
        .iter()
        .map(|item| item.counts[primary] as f64)
        .collect();
    writeln!(output, "## 相对主 tokenizer 线性拟合\n").unwrap();
    writeln!(output, "| Tokenizer | 斜率 | 截距 | R² | 估算式 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---|").unwrap();
    for (index, spec) in specs.iter().enumerate() {
        if index == primary {
            continue;
        }
        let y: Vec<f64> = items.iter().map(|item| item.counts[index] as f64).collect();
        let (slope, intercept, r_squared) = linear_regression(&x, &y);
        writeln!(
            output,
            "| {} | {slope:.8} | {intercept:+.2} | {r_squared:.9} | `{} ≈ {slope:.5} × {} {intercept:+.0}` |",
            escape(&spec.name),
            escape(&spec.name),
            escape(&specs[primary].name)
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_perspectives(specs: &[TokenizerSpec], items: &[Item], output: &mut String) {
    writeln!(output, "## 分 tokenizer 视角结论\n").unwrap();
    let totals: Vec<i64> = (0..specs.len()).map(|index| total(items, index)).collect();
    for (core, core_spec) in specs.iter().enumerate() {
        writeln!(output, "### 以 {} 为核心\n", escape(&core_spec.name)).unwrap();
        writeln!(output, "| 对比对象 | 核心总量 | 对象总量 | 对象 - 核心 | 对象 / 核心 | 逐条对象更少 | 逐条相同 | 逐条对象更多 | 结论 |").unwrap();
        writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|---|").unwrap();
        for (other, other_spec) in specs.iter().enumerate() {
            if other == core {
                continue;
            }
            let (less, equal, more) = directions(items, other, core);
            writeln!(
                output,
                "| {} | {} | {} | {:+} | {:.5}x | {less} | {equal} | {more} | {} |",
                escape(&other_spec.name),
                integer(totals[core]),
                integer(totals[other]),
                totals[other] - totals[core],
                ratio(totals[other], totals[core]),
                perspective_summary(
                    &core_spec.name,
                    &other_spec.name,
                    totals[core],
                    totals[other]
                )
            )
            .unwrap();
        }
        output.push('\n');
    }
}

fn counts(items: &[Item], index: usize) -> Vec<i64> {
    items.iter().map(|item| item.counts[index]).collect()
}

fn total(items: &[Item], index: usize) -> i64 {
    items.iter().map(|item| item.counts[index]).sum()
}

fn directions(items: &[Item], other: usize, core: usize) -> (usize, usize, usize) {
    let less = items
        .iter()
        .filter(|item| item.counts[other] < item.counts[core])
        .count();
    let equal = items
        .iter()
        .filter(|item| item.counts[other] == item.counts[core])
        .count();
    (less, equal, items.len() - less - equal)
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

fn perspective_summary(
    core_name: &str,
    other_name: &str,
    core_total: i64,
    other_total: i64,
) -> String {
    let difference = other_total - core_total;
    if difference == 0 {
        format!("{other_name} 与 {core_name} 总量相同")
    } else if difference > 0 {
        format!(
            "{other_name} 比 {core_name} 多 {:.2}%，{core_name} 更节省 token",
            percent(difference, core_total)
        )
    } else {
        format!(
            "{other_name} 比 {core_name} 少 {:.2}%，{core_name} 使用更多 token",
            -percent(difference, core_total)
        )
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

    fn args() -> Args {
        Args {
            tokenizers: vec![
                "A=a_tokens".parse().unwrap(),
                "B=b_tokens".parse().unwrap(),
                "C=c_tokens".parse().unwrap(),
            ],
            datasets: vec!["short=a.parquet,b.parquet,c.parquet".parse().unwrap()],
            primary: Some("C".to_owned()),
            title: "test".to_owned(),
            output: "test.md".into(),
            group_col: None,
            batch_size: 8,
        }
    }

    #[test]
    fn validates_dynamic_tokenizer_count_and_primary() {
        assert_eq!(validate_args(&args()).unwrap(), 2);
    }

    #[test]
    fn rejects_dataset_file_count_mismatch() {
        let mut args = args();
        args.datasets[0].paths.pop();
        assert!(validate_args(&args).is_err());
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
