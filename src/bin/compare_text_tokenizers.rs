use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, ValueEnum};
use rayon::prelude::*;
use tokenizers::Tokenizer;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GroupBy {
    /// 使用相对于 root 的第一级目录名分组。
    TopLevel,
    /// 使用文件所在的相对目录分组。
    Parent,
}

#[derive(Parser, Debug)]
struct Args {
    /// 待分析文本的根目录。
    #[arg(long)]
    root: PathBuf,

    /// 报告标题。
    #[arg(long)]
    title: String,

    /// Markdown 报告输出路径。
    #[arg(long)]
    output: PathBuf,

    /// 纳入统计的扩展名，可重复传入；不传则接受所有文件。
    #[arg(long = "extension")]
    extensions: Vec<String>,

    /// 相对 root 排除的目录或文件前缀，可重复传入。
    #[arg(long = "exclude")]
    excludes: Vec<PathBuf>,

    /// 报告中逐项表格的分组方式。
    #[arg(long, value_enum, default_value_t = GroupBy::TopLevel)]
    group_by: GroupBy,

    #[arg(long, default_value = "tokenizer/deepseek-v4-pro/tokenizer.json")]
    v4p_tokenizer: PathBuf,

    #[arg(long, default_value = "tokenizer/deepseek-v3-2/tokenizer.json")]
    v32_tokenizer: PathBuf,

    #[arg(long, default_value = "tokenizer/openpangu-2-0-flash/tokenizer.json")]
    openpangu_tokenizer: PathBuf,
}

#[derive(Debug)]
struct Item {
    path: String,
    group: String,
    chars: usize,
    v4p: usize,
    v32: usize,
    openpangu: usize,
}

#[derive(Clone, Copy)]
enum CountKind {
    V4p,
    V32,
    OpenPangu,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let root = args
        .root
        .canonicalize()
        .with_context(|| format!("无法解析根目录: {}", args.root.display()))?;
    if !root.is_dir() {
        bail!("root 不是目录: {}", root.display());
    }

    let tokenizers = [
        load_tokenizer(&args.v4p_tokenizer)?,
        load_tokenizer(&args.v32_tokenizer)?,
        load_tokenizer(&args.openpangu_tokenizer)?,
    ];
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
    items.sort_by(|left, right| {
        natural_path_key(Path::new(&left.path)).cmp(&natural_path_key(Path::new(&right.path)))
    });

    let report = render_report(&args, &root, &items);
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("无法创建报告目录: {}", parent.display()))?;
    }
    fs::write(&args.output, report)
        .with_context(|| format!("无法写入报告: {}", args.output.display()))?;

    println!(
        "已统计 {} 个文件，报告写入 {}",
        items.len(),
        args.output.display()
    );
    Ok(())
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
    let mut normalized = PathBuf::new();
    for component in path.components() {
        if let std::path::Component::Normal(part) = component {
            normalized.push(part);
        }
    }
    normalized
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
            .map(|extension| {
                extensions
                    .iter()
                    .any(|item| item.eq_ignore_ascii_case(extension))
            })
            .unwrap_or(false)
}

fn analyze_file(
    root: &Path,
    relative: &Path,
    group_by: GroupBy,
    tokenizers: &[Tokenizer; 3],
) -> Result<Item> {
    let path = root.join(relative);
    let text = fs::read_to_string(&path)
        .with_context(|| format!("文件不是有效 UTF-8 文本: {}", path.display()))?;
    let path_label = relative.to_string_lossy().replace('\\', "/");
    let group = group_for(relative, group_by);
    Ok(Item {
        path: path_label,
        group,
        chars: text.chars().count(),
        v4p: count_tokens(&tokenizers[0], &text, &path)?,
        v32: count_tokens(&tokenizers[1], &text, &path)?,
        openpangu: count_tokens(&tokenizers[2], &text, &path)?,
    })
}

fn count_tokens(tokenizer: &Tokenizer, text: &str, path: &Path) -> Result<usize> {
    tokenizer
        .encode(text, false)
        .map(|encoding| encoding.len())
        .map_err(|error| anyhow::anyhow!("tokenize 失败 {}: {error}", path.display()))
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

fn render_report(args: &Args, root: &Path, items: &[Item]) -> String {
    let mut output = String::new();
    writeln!(output, "# {}\n", args.title).unwrap();
    writeln!(output, "## 统计口径\n").unwrap();
    writeln!(output, "| 项目 | 值 |").unwrap();
    writeln!(output, "|---|---|").unwrap();
    writeln!(
        output,
        "| 输入目录 | `{}` |",
        markdown_escape(&root.to_string_lossy())
    )
    .unwrap();
    writeln!(output, "| 文件数 | {} |", items.len()).unwrap();
    writeln!(
        output,
        "| 文件扩展名 | {} |",
        if args.extensions.is_empty() {
            "全部".to_owned()
        } else {
            args.extensions.join(", ")
        }
    )
    .unwrap();
    writeln!(
        output,
        "| 排除路径 | {} |",
        if args.excludes.is_empty() {
            "无".to_owned()
        } else {
            args.excludes
                .iter()
                .map(|p| format!("`{}`", markdown_escape(&p.to_string_lossy())))
                .collect::<Vec<_>>()
                .join(", ")
        }
    )
    .unwrap();
    writeln!(
        output,
        "| 编码方式 | `Tokenizer::encode(text, false)`，不添加 special tokens |"
    )
    .unwrap();
    writeln!(
        output,
        "| V4P tokenizer | `{}` |",
        markdown_escape(&args.v4p_tokenizer.to_string_lossy())
    )
    .unwrap();
    writeln!(
        output,
        "| V3.2 tokenizer | `{}` |",
        markdown_escape(&args.v32_tokenizer.to_string_lossy())
    )
    .unwrap();
    writeln!(
        output,
        "| OpenPangu tokenizer | `{}` |\n",
        markdown_escape(&args.openpangu_tokenizer.to_string_lossy())
    )
    .unwrap();

    writeln!(output, "## 总体结果\n").unwrap();
    writeln!(output, "| Tokenizer | 总 token 数 | 平均每项 | 中位数 | 最小 | 最大 | 相对 V4P 差值 | 相对 V4P 增幅 |").unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, kind) in [
        ("V4P", CountKind::V4p),
        ("V3.2", CountKind::V32),
        ("OpenPangu 2.0 Flash", CountKind::OpenPangu),
    ] {
        let counts = counts(items, kind);
        let total: usize = counts.iter().sum();
        let v4p_total: usize = items.iter().map(|item| item.v4p).sum();
        let difference = total as i128 - v4p_total as i128;
        writeln!(
            output,
            "| {name} | {} | {:.2} | {:.0} | {} | {} | {:+} | {:+.2}% |",
            format_integer(total),
            total as f64 / items.len() as f64,
            percentile(&counts, 0.5),
            format_integer(*counts.iter().min().unwrap()),
            format_integer(*counts.iter().max().unwrap()),
            difference,
            percent(difference, v4p_total),
        )
        .unwrap();
    }

    writeln!(output, "\n## 按分组汇总\n").unwrap();
    writeln!(
        output,
        "| 分组 | 项数 | 字符数 | V4P | V3.2 | OpenPangu | V3.2 / V4P | OpenPangu / V4P |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    let groups = grouped(items);
    for (group, group_items) in &groups {
        let chars: usize = group_items.iter().map(|item| item.chars).sum();
        let v4p: usize = group_items.iter().map(|item| item.v4p).sum();
        let v32: usize = group_items.iter().map(|item| item.v32).sum();
        let openpangu: usize = group_items.iter().map(|item| item.openpangu).sum();
        writeln!(
            output,
            "| {} | {} | {} | {} | {} | {} | {:.5}x | {:.5}x |",
            markdown_escape(group),
            group_items.len(),
            format_integer(chars),
            format_integer(v4p),
            format_integer(v32),
            format_integer(openpangu),
            ratio(v32, v4p),
            ratio(openpangu, v4p)
        )
        .unwrap();
    }

    writeln!(output, "\n## 逐项差异分位数\n").unwrap();
    writeln!(
        output,
        "| 分位数 | V3.2 - V4P | V3.2 / V4P | OpenPangu - V4P | OpenPangu / V4P |"
    )
    .unwrap();
    writeln!(output, "|---|---:|---:|---:|---:|").unwrap();
    let v32_deltas: Vec<i128> = items
        .iter()
        .map(|item| item.v32 as i128 - item.v4p as i128)
        .collect();
    let v32_ratios: Vec<f64> = items.iter().map(|item| ratio(item.v32, item.v4p)).collect();
    let pangu_deltas: Vec<i128> = items
        .iter()
        .map(|item| item.openpangu as i128 - item.v4p as i128)
        .collect();
    let pangu_ratios: Vec<f64> = items
        .iter()
        .map(|item| ratio(item.openpangu, item.v4p))
        .collect();
    for (label, quantile) in [
        ("最小", 0.0),
        ("P10", 0.1),
        ("P25", 0.25),
        ("P50", 0.5),
        ("P75", 0.75),
        ("P90", 0.9),
        ("P99", 0.99),
        ("最大", 1.0),
    ] {
        writeln!(
            output,
            "| {label} | {:+.0} | {:.5}x | {:+.0} | {:.5}x |",
            percentile(&v32_deltas, quantile),
            percentile(&v32_ratios, quantile),
            percentile(&pangu_deltas, quantile),
            percentile(&pangu_ratios, quantile)
        )
        .unwrap();
    }

    writeln!(output, "\n## 逐项结果\n").unwrap();
    for (group, group_items) in groups {
        writeln!(output, "### {}\n", markdown_escape(&group)).unwrap();
        writeln!(output, "| 文件 | 字符数 | V4P | V3.2 | OpenPangu | V3.2 - V4P | V3.2 / V4P | OpenPangu - V4P | OpenPangu / V4P |").unwrap();
        writeln!(output, "|---|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        for item in group_items {
            writeln!(
                output,
                "| `{}` | {} | {} | {} | {} | {:+} | {:.5}x | {:+} | {:.5}x |",
                markdown_escape(&item.path),
                format_integer(item.chars),
                format_integer(item.v4p),
                format_integer(item.v32),
                format_integer(item.openpangu),
                item.v32 as i128 - item.v4p as i128,
                ratio(item.v32, item.v4p),
                item.openpangu as i128 - item.v4p as i128,
                ratio(item.openpangu, item.v4p)
            )
            .unwrap();
        }
        output.push('\n');
    }
    output
}

fn grouped(items: &[Item]) -> BTreeMap<String, Vec<&Item>> {
    let mut groups: BTreeMap<String, Vec<&Item>> = BTreeMap::new();
    for item in items {
        groups.entry(item.group.clone()).or_default().push(item);
    }
    groups
}

fn counts(items: &[Item], kind: CountKind) -> Vec<usize> {
    items
        .iter()
        .map(|item| match kind {
            CountKind::V4p => item.v4p,
            CountKind::V32 => item.v32,
            CountKind::OpenPangu => item.openpangu,
        })
        .collect()
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

fn percentile<T>(values: &[T], quantile: f64) -> f64
where
    T: Copy + PartialOrd + IntoF64,
{
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

fn format_integer(value: usize) -> String {
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

fn markdown_escape(value: &str) -> String {
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
    fn integer_format_has_grouping() {
        assert_eq!(format_integer(12), "12");
        assert_eq!(format_integer(1_234_567), "1,234,567");
    }

    #[test]
    fn percentile_interpolates() {
        assert_eq!(percentile(&[0usize, 10], 0.5), 5.0);
        assert_eq!(percentile(&[0usize, 10, 20], 0.5), 10.0);
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
    fn exclusions_match_path_prefixes() {
        assert!(Path::new("engine/test/case.rs").starts_with(Path::new("engine/test")));
        assert!(!Path::new("engine/tester.rs").starts_with(Path::new("engine/test")));
    }
}
