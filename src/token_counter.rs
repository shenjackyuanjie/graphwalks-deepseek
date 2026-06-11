//! Token 计数器：从 parquet 文件中读取文本列，使用 tokenizer 编码，
//! 追加 token 数列和超长标记列。

use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, LargeStringArray, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use tokenizers::Tokenizer;

pub const TOKEN_COUNT_COL: &str = "deepseek_v4_input_tokens";
pub const OVER_1M_COL: &str = "deepseek_v4_over_1m";

/// 处理单个 parquet 文件：读 text_col 列 → tokenize → 追加列 → 写出。
pub fn process_file(
    input_path: &Path,
    tokenizer: Arc<Tokenizer>,
    text_col: &str,
    batch_size: usize,
) -> Result<()> {
    let input_file = File::open(input_path)
        .with_context(|| format!("无法打开输入 parquet: {}", input_path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .with_context(|| format!("无法创建 parquet reader: {}", input_path.display()))?;

    let input_schema = builder.schema().clone();
    if input_schema.index_of(TOKEN_COUNT_COL).is_ok() || input_schema.index_of(OVER_1M_COL).is_ok()
    {
        return Err(anyhow!(
            "输入文件已包含 {TOKEN_COUNT_COL:?} 或 {OVER_1M_COL:?} 列: {}",
            input_path.display()
        ));
    }

    let total_rows = builder.metadata().file_metadata().num_rows() as u64;
    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .with_context(|| format!("无法构建 parquet reader: {}", input_path.display()))?;

    let output_path = output_path_for(input_path);
    let output_file = File::create(&output_path)
        .with_context(|| format!("无法创建输出 parquet: {}", output_path.display()))?;

    let pb = ProgressBar::new(total_rows);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} {msg} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows",
        )
        .unwrap()
        .progress_chars("#>-"),
    );
    pb.set_message(
        input_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
    );

    let mut writer: Option<ArrowWriter<File>> = None;

    for batch_result in reader {
        let batch = batch_result.with_context(|| "读取 parquet batch 失败")?;

        let texts: Vec<String> = {
            let text_array = get_text_array(&batch, text_col)?;
            (0..text_array.len())
                .map(|i| {
                    if text_array.is_null(i) {
                        String::new()
                    } else {
                        text_array.value(i).to_owned()
                    }
                })
                .collect()
        };

        let token_counts: Vec<i32> = texts
            .par_iter()
            .map(|text| {
                tokenizer
                    .encode(text.as_str(), false)
                    .map(|enc| enc.get_ids().len() as i32)
                    .unwrap_or(-1)
            })
            .collect();

        let over_1m: Vec<bool> = token_counts.iter().map(|&n| n > 1_000_000).collect();
        let new_batch = append_columns(batch, token_counts, over_1m)?;

        if writer.is_none() {
            writer = Some(
                ArrowWriter::try_new(output_file.try_clone()?, new_batch.schema(), None)
                    .with_context(|| "无法创建 parquet writer")?,
            );
        }

        writer
            .as_mut()
            .unwrap()
            .write(&new_batch)
            .with_context(|| "写入 parquet batch 失败")?;

        pb.inc(new_batch.num_rows() as u64);
    }

    if let Some(writer) = writer {
        writer
            .close()
            .with_context(|| "关闭 parquet writer 失败")?;
    }

    pb.finish_with_message(format!("已写入 {}", output_path.display()));
    Ok(())
}

fn output_path_for(input: &Path) -> PathBuf {
    let stem = input.file_stem().unwrap_or_default().to_string_lossy();
    input.with_file_name(format!("{stem}.deepseek_v4_tokens.parquet"))
}

fn get_text_array<'a>(batch: &'a RecordBatch, text_col: &str) -> Result<Box<dyn TextAccessor + 'a>> {
    let idx = batch
        .schema()
        .index_of(text_col)
        .with_context(|| format!("找不到列 {text_col:?}"))?;

    let col = batch.column(idx);

    match col.data_type() {
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("Utf8 列类型转换失败"))?;
            Ok(Box::new(Utf8Accessor { arr }))
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| anyhow!("LargeUtf8 列类型转换失败"))?;
            Ok(Box::new(LargeUtf8Accessor { arr }))
        }
        other => Err(anyhow!(
            "列 {text_col:?} 必须是 Utf8 或 LargeUtf8 类型，实际为 {other:?}"
        )),
    }
}

trait TextAccessor {
    fn len(&self) -> usize;
    fn is_null(&self, i: usize) -> bool;
    fn value(&self, i: usize) -> &str;
}

struct Utf8Accessor<'a> {
    arr: &'a StringArray,
}

impl TextAccessor for Utf8Accessor<'_> {
    fn len(&self) -> usize { self.arr.len() }
    fn is_null(&self, i: usize) -> bool { self.arr.is_null(i) }
    fn value(&self, i: usize) -> &str { self.arr.value(i) }
}

struct LargeUtf8Accessor<'a> {
    arr: &'a LargeStringArray,
}

impl TextAccessor for LargeUtf8Accessor<'_> {
    fn len(&self) -> usize { self.arr.len() }
    fn is_null(&self, i: usize) -> bool { self.arr.is_null(i) }
    fn value(&self, i: usize) -> &str { self.arr.value(i) }
}

fn append_columns(
    batch: RecordBatch,
    token_counts: Vec<i32>,
    over_1m: Vec<bool>,
) -> Result<RecordBatch> {
    if token_counts.len() != batch.num_rows() || over_1m.len() != batch.num_rows() {
        return Err(anyhow!(
            "新列长度不匹配: 行数={}, tokens={}, over_1m={}",
            batch.num_rows(),
            token_counts.len(),
            over_1m.len()
        ));
    }

    let old_schema = batch.schema();
    let mut fields = old_schema.fields().to_vec();
    fields.push(Arc::new(Field::new(
        TOKEN_COUNT_COL,
        DataType::Int32,
        false,
    )));
    fields.push(Arc::new(Field::new(OVER_1M_COL, DataType::Boolean, false)));

    let new_schema = Arc::new(Schema::new(fields));

    let mut columns: Vec<ArrayRef> = batch.columns().to_vec();
    columns.push(Arc::new(Int32Array::from(token_counts)) as ArrayRef);
    columns.push(Arc::new(BooleanArray::from(over_1m)) as ArrayRef);

    RecordBatch::try_new(new_schema, columns)
        .with_context(|| "构建输出 RecordBatch 失败")
}
