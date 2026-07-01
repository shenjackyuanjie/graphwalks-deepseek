use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GroupBy {
    TopLevel,
    Parent,
}

#[derive(Clone, Debug)]
struct TokenizerSpec {
    name: String,
    path: PathBuf,
}

impl FromStr for TokenizerSpec {
    type Err = anyhow::Error;

    fn from_str(value: &str) -> Result<Self> {
        let (name, path) = value
            .split_once('=')
            .context("tokenizer 参数格式应为 NAME=TOKENIZER_JSON")?;
        if name.trim().is_empty() || path.is_empty() {
            bail!("tokenizer 参数格式应为 NAME=TOKENIZER_JSON，实际为 {value:?}");
        }
        Ok(Self {
            name: name.trim().to_owned(),
            path: path.into(),
        })
    }
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    root: PathBuf,

    #[arg(long)]
    title: String,

    #[arg(long)]
    output: PathBuf,

    /// Tokenizer 定义，可重复传入，格式为 NAME=TOKENIZER_JSON。
    #[arg(long = "tokenizer", required = true)]
    tokenizers: Vec<TokenizerSpec>,

    /// 主对比 tokenizer 名称；省略时使用第一个 --tokenizer。
    #[arg(long)]
    primary: Option<String>,

    #[arg(long = "extension")]
    extensions: Vec<String>,

    #[arg(long = "exclude")]
    excludes: Vec<PathBuf>,

    #[arg(long, value_enum, default_value_t = GroupBy::TopLevel)]
    group_by: GroupBy,

    /// 在每个分组内按自然文件顺序每 N 项汇总一次。
    #[arg(long)]
    chunk_size: Option<NonZeroUsize>,
}

#[derive(Debug)]
struct Item {
    path: String,
    group: String,
    chars: usize,
    counts: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let primary = validate_specs(&args.tokenizers, args.primary.as_deref())?;
    let root = args
        .root
        .canonicalize()
        .with_context(|| format!("无法解析根目录: {}", args.root.display()))?;
    if !root.is_dir() {
        bail!("root 不是目录: {}", root.display());
    }

    let tokenizers: Vec<Tokenizer> = args
        .tokenizers
        .iter()
        .map(|spec| load_tokenizer(&spec.path))
        .collect::<Result<_>>()?;
    let extensions = normalize_extensions(&args.extensions);
    let excludes: Vec<PathBuf> = args
        .excludes
        .iter()
        .map(|path| normalize_relative_path(path))
        .collect();

    let mut files = Vec::new();
    collect_files(&root, &root, &extensions, &excludes, &mut files)?;
    files.sort_by_key(|path| natural_path_key(path));
    if files.is_empty() {
        bail!("没有找到符合条件的文本文件: {}", root.display());
    }

    let mut items: Vec<Item> = files
        .par_iter()
        .map(|relative| analyze_file(&root, relative, args.group_by, &tokenizers))
        .collect::<Result<Vec<_>>>()?;
    items.sort_by_key(|item| natural_path_key(Path::new(&item.path)));

    let report = render_report(&args, &root, &items, primary);
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建报告目录: {}", parent.display()))?;
    }
    fs::write(&args.output, report)
        .with_context(|| format!("无法写入报告: {}", args.output.display()))?;
    println!(
        "已使用 {} 个 tokenizer 统计 {} 个文件，报告写入 {}",
        args.tokenizers.len(),
        items.len(),
        args.output.display()
    );
    Ok(())
}

fn validate_specs(specs: &[TokenizerSpec], primary: Option<&str>) -> Result<usize> {
    if specs.len() < 2 {
        bail!("至少需要传入两个 --tokenizer");
    }
    let mut names = HashSet::new();
    for spec in specs {
        if !names.insert(&spec.name) {
            bail!("tokenizer 名称重复: {:?}", spec.name);
        }
    }
    match primary {
        Some(name) => specs
            .iter()
            .position(|spec| spec.name == name)
            .with_context(|| format!("primary {name:?} 不在 tokenizer 列表中")),
        None => Ok(0),
    }
}

fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    Tokenizer::from_file(path)
        .map_err(|error| anyhow::anyhow!("加载 tokenizer 失败 {}: {error}", path.display()))
}

fn normalize_extensions(extensions: &[String]) -> Vec<String> {
    extensions
        .iter()
        .map(|extension| extension.trim_start_matches('.').to_ascii_lowercase())
        .collect()
}

fn normalize_relative_path(path: &Path) -> PathBuf {
    path.components()
        .filter_map(|component| match component {
            std::path::Component::Normal(part) => Some(part),
            _ => None,
        })
        .collect()
}

fn collect_files(
    root: &Path,
    directory: &Path,
    extensions: &[String],
    excludes: &[PathBuf],
    output: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in
        fs::read_dir(directory).with_context(|| format!("无法读取目录: {}", directory.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        let relative = path.strip_prefix(root).expect("递归路径应位于 root 下");
        if excludes
            .iter()
            .any(|excluded| relative.starts_with(excluded))
        {
            continue;
        }
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            collect_files(root, &path, extensions, excludes, output)?;
        } else if file_type.is_file() && extension_matches(&path, extensions) {
            output.push(relative.to_path_buf());
        }
    }
    Ok(())
}

fn extension_matches(path: &Path, extensions: &[String]) -> bool {
    extensions.is_empty()
        || path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extensions
                    .iter()
                    .any(|item| item.eq_ignore_ascii_case(extension))
            })
}

fn analyze_file(
    root: &Path,
    relative: &Path,
    group_by: GroupBy,
    tokenizers: &[Tokenizer],
) -> Result<Item> {
    let path = root.join(relative);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("文件不是有效 UTF-8 文本: {}", path.display()))?;
    let counts = tokenizers
        .iter()
        .map(|tokenizer| {
            tokenizer
                .encode(text.as_str(), false)
                .map(|encoding| encoding.len())
                .map_err(|error| anyhow::anyhow!("tokenize 失败 {}: {error}", path.display()))
        })
        .collect::<Result<_>>()?;
    Ok(Item {
        path: relative.to_string_lossy().replace('\\', "/"),
        group: group_for(relative, group_by),
        chars: text.chars().count(),
        counts,
    })
}

fn group_for(path: &Path, group_by: GroupBy) -> String {
    match group_by {
        GroupBy::TopLevel => path
            .components()
            .next()
            .map(|part| part.as_os_str().to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_owned()),
        GroupBy::Parent => path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .map(|parent| parent.to_string_lossy().replace('\\', "/"))
            .unwrap_or_else(|| ".".to_owned()),
    }
}

fn render_report(args: &Args, root: &Path, items: &[Item], primary: usize) -> String {
    let mut output = String::new();
    writeln!(output, "# {}\n", args.title).unwrap();
    render_method(args, root, items, primary, &mut output);
    render_overall(&args.tokenizers, items, primary, &mut output);
    render_groups(&args.tokenizers, items, primary, &mut output);
    if let Some(chunk_size) = args.chunk_size {
        render_chunks(
            &args.tokenizers,
            items,
            primary,
            chunk_size.get(),
            &mut output,
        );
    }
    render_quantiles(&args.tokenizers, items, primary, &mut output);
    render_perspectives(&args.tokenizers, items, &mut output);
    render_details(&args.tokenizers, items, primary, &mut output);
    output
}

fn render_method(args: &Args, root: &Path, items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 统计口径\n").unwrap();
    writeln!(output, "| 项目 | 值 |").unwrap();
    writeln!(output, "|---|---|").unwrap();
    writeln!(
        output,
        "| 输入目录 | `{}` |",
        escape(&root.to_string_lossy())
    )
    .unwrap();
    writeln!(output, "| 文件数 | {} |", items.len()).unwrap();
    writeln!(output, "| Tokenizer 数 | {} |", args.tokenizers.len()).unwrap();
    writeln!(
        output,
        "| 分块汇总 | {} |",
        args.chunk_size
            .map(|size| format!("每 {} 项", size.get()))
            .unwrap_or_else(|| "关闭".to_owned())
    )
    .unwrap();
    writeln!(
        output,
        "| 主对比 tokenizer | {} |",
        escape(&args.tokenizers[primary].name)
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
            "| {} tokenizer | `{}` |",
            escape(&spec.name),
            escape(&spec.path.to_string_lossy())
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_overall(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    let primary_total = total(items, primary);
    writeln!(output, "## 总体结果\n").unwrap();
    writeln!(output, "| Tokenizer | 总 token 数 | 平均每项 | 中位数 | 最小 | 最大 | 相对主 tokenizer 差值 | 相对主 tokenizer 增幅 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (index, spec) in specs.iter().enumerate() {
        let values = counts(items, index);
        let sum: usize = values.iter().sum();
        writeln!(
            output,
            "| {} | {} | {:.2} | {:.0} | {} | {} | {:+} | {:+.2}% |",
            escape(&spec.name),
            integer(sum),
            sum as f64 / items.len() as f64,
            percentile(&values, 0.5),
            integer(*values.iter().min().unwrap()),
            integer(*values.iter().max().unwrap()),
            sum as i128 - primary_total as i128,
            percent(sum as i128 - primary_total as i128, primary_total)
        )
        .unwrap();
    }
    output.push('\n');
}

fn render_groups(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 按分组汇总\n").unwrap();
    write!(output, "| 分组 | 项数 | 字符数").unwrap();
    for spec in specs {
        write!(output, " | {}", escape(&spec.name)).unwrap();
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
    write!(output, "|---|---:|---:").unwrap();
    for _ in specs {
        write!(output, "|---:").unwrap();
    }
    for index in 0..specs.len() {
        if index != primary {
            write!(output, "|---:").unwrap();
        }
    }
    writeln!(output, "|").unwrap();

    for (group, group_items) in grouped(items) {
        let totals: Vec<usize> = (0..specs.len())
            .map(|index| group_items.iter().map(|item| item.counts[index]).sum())
            .collect();
        let chars: usize = group_items.iter().map(|item| item.chars).sum();
        write!(
            output,
            "| {} | {} | {}",
            escape(&group),
            group_items.len(),
            integer(chars)
        )
        .unwrap();
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
    output.push('\n');
}

fn render_chunks(
    specs: &[TokenizerSpec],
    items: &[Item],
    primary: usize,
    chunk_size: usize,
    output: &mut String,
) {
    writeln!(output, "## 每 {chunk_size} 项汇总\n").unwrap();
    write!(output, "| 分组 | 范围 | 项数 | 字符数").unwrap();
    for spec in specs {
        write!(output, " | {}", escape(&spec.name)).unwrap();
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
    write!(output, "|---|---|---:|---:").unwrap();
    for _ in specs {
        write!(output, "|---:").unwrap();
    }
    for index in 0..specs.len() {
        if index != primary {
            write!(output, "|---:").unwrap();
        }
    }
    writeln!(output, "|").unwrap();

    for (group, group_items) in grouped(items) {
        for chunk in group_items.chunks(chunk_size) {
            let totals: Vec<usize> = (0..specs.len())
                .map(|index| chunk.iter().map(|item| item.counts[index]).sum())
                .collect();
            let chars: usize = chunk.iter().map(|item| item.chars).sum();
            let range = format!(
                "{}–{}",
                file_label(&chunk[0].path),
                file_label(&chunk[chunk.len() - 1].path)
            );
            write!(
                output,
                "| {} | {} | {} | {}",
                escape(&group),
                escape(&range),
                chunk.len(),
                integer(chars)
            )
            .unwrap();
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
    }
    output.push('\n');
}

fn file_label(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn render_quantiles(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 逐项差异分位数\n").unwrap();
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
        let deltas: Vec<i128> = items
            .iter()
            .map(|item| item.counts[index] as i128 - item.counts[primary] as i128)
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

fn render_perspectives(specs: &[TokenizerSpec], items: &[Item], output: &mut String) {
    writeln!(output, "## 分 tokenizer 视角结论\n").unwrap();
    let totals: Vec<usize> = (0..specs.len()).map(|index| total(items, index)).collect();
    for (core, core_spec) in specs.iter().enumerate() {
        writeln!(output, "### 以 {} 为核心\n", escape(&core_spec.name)).unwrap();
        writeln!(output, "| 对比对象 | 核心总量 | 对象总量 | 对象 - 核心 | 对象 / 核心 | 逐项对象更少 | 逐项相同 | 逐项对象更多 | 结论 |").unwrap();
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
                totals[other] as i128 - totals[core] as i128,
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

fn render_details(specs: &[TokenizerSpec], items: &[Item], primary: usize, output: &mut String) {
    writeln!(output, "## 逐项结果\n").unwrap();
    for (group, group_items) in grouped(items) {
        writeln!(output, "### {}\n", escape(&group)).unwrap();
        write!(output, "| 文件 | 字符数").unwrap();
        for spec in specs {
            write!(output, " | {}", escape(&spec.name)).unwrap();
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
        for item in group_items {
            write!(
                output,
                "| `{}` | {}",
                escape(&item.path),
                integer(item.chars)
            )
            .unwrap();
            for value in &item.counts {
                write!(output, " | {}", integer(*value)).unwrap();
            }
            for (index, value) in item.counts.iter().enumerate() {
                if index != primary {
                    write!(output, " | {:.5}x", ratio(*value, item.counts[primary])).unwrap();
                }
            }
            writeln!(output, " |").unwrap();
        }
        output.push('\n');
    }
}

fn grouped(items: &[Item]) -> BTreeMap<String, Vec<&Item>> {
    let mut groups: BTreeMap<String, Vec<&Item>> = BTreeMap::new();
    for item in items {
        groups.entry(item.group.clone()).or_default().push(item);
    }
    groups
}

fn counts(items: &[Item], index: usize) -> Vec<usize> {
    items.iter().map(|item| item.counts[index]).collect()
}

fn total(items: &[Item], index: usize) -> usize {
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

fn perspective_summary(
    core_name: &str,
    other_name: &str,
    core_total: usize,
    other_total: usize,
) -> String {
    let difference = other_total as i128 - core_total as i128;
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

fn ratio(value: usize, baseline: usize) -> f64 {
    if baseline == 0 {
        f64::NAN
    } else {
        value as f64 / baseline as f64
    }
}

fn percent(difference: i128, baseline: usize) -> f64 {
    if baseline == 0 {
        f64::NAN
    } else {
        difference as f64 / baseline as f64 * 100.0
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

trait IntoF64 {
    fn into_f64(self) -> f64;
}

impl IntoF64 for usize {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl IntoF64 for i128 {
    fn into_f64(self) -> f64 {
        self as f64
    }
}

impl IntoF64 for f64 {
    fn into_f64(self) -> f64 {
        self
    }
}

fn integer(value: usize) -> String {
    let digits = value.to_string();
    let mut output = String::with_capacity(digits.len() + digits.len() / 3);
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

fn natural_path_key(path: &Path) -> Vec<NaturalPart> {
    let value = path.to_string_lossy().to_ascii_lowercase();
    let mut parts = Vec::new();
    let mut buffer = String::new();
    let mut digit_mode = None;
    for character in value.chars() {
        let is_digit = character.is_ascii_digit();
        if digit_mode.is_some_and(|mode| mode != is_digit) {
            parts.push(natural_part(&buffer, digit_mode.unwrap()));
            buffer.clear();
        }
        digit_mode = Some(is_digit);
        buffer.push(character);
    }
    if let Some(mode) = digit_mode {
        parts.push(natural_part(&buffer, mode));
    }
    parts
}

#[derive(Debug, Eq, Ord, PartialEq, PartialOrd)]
enum NaturalPart {
    Text(String),
    Number(u128, usize),
}

fn natural_part(value: &str, digits: bool) -> NaturalPart {
    if digits {
        NaturalPart::Number(value.parse().unwrap_or(u128::MAX), value.len())
    } else {
        NaturalPart::Text(value.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tokenizer_spec() {
        let spec: TokenizerSpec = "GLM 5.2=tokenizer/glm/tokenizer.json".parse().unwrap();
        assert_eq!(spec.name, "GLM 5.2");
        assert_eq!(spec.path, PathBuf::from("tokenizer/glm/tokenizer.json"));
    }

    #[test]
    fn validates_unique_names_and_primary() {
        let specs = vec![
            "A=a.json".parse().unwrap(),
            "B=b.json".parse().unwrap(),
            "C=c.json".parse().unwrap(),
        ];
        assert_eq!(validate_specs(&specs, Some("C")).unwrap(), 2);
        assert!(validate_specs(&specs, Some("D")).is_err());
    }

    #[test]
    fn natural_sort_orders_numbered_chapters() {
        let mut paths = [
            PathBuf::from("10.md"),
            PathBuf::from("2.md"),
            PathBuf::from("1.md"),
        ];
        paths.sort_by_key(|path| natural_path_key(path));
        assert_eq!(
            paths,
            [
                PathBuf::from("1.md"),
                PathBuf::from("2.md"),
                PathBuf::from("10.md")
            ]
        );
    }

    #[test]
    fn perspective_is_relative_to_selected_core() {
        assert!(perspective_summary("A", "B", 100, 120).contains("B 比 A 多 20.00%"));
        assert!(perspective_summary("A", "B", 100, 80).contains("B 比 A 少 20.00%"));
    }
}
