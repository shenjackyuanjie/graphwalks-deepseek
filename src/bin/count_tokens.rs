use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
    Array, ArrayRef, BooleanArray, Int32Array, LargeStringArray, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ArrowWriter;
use rayon::prelude::*;
use tokenizers::Tokenizer;

const TOKEN_COUNT_COL: &str = "deepseek_v4_input_tokens";
const OVER_1M_COL: &str = "deepseek_v4_over_1m";

#[derive(Parser, Debug)]
struct Args {
    /// Path to DeepSeek V4 tokenizer.json.
    #[arg(long, default_value = "tokenizer/deepseek-v4-pro/tokenizer.json")]
    tokenizer_json: PathBuf,

    /// Text column to count. GraphWalks uses prompt.
    #[arg(long, default_value = "prompt")]
    text_col: String,

    /// Rows per parquet batch. Long-context rows can be very large.
    #[arg(long, default_value_t = 32)]
    batch_size: usize,

    /// Input parquet files.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let tokenizer = Tokenizer::from_file(&args.tokenizer_json)
        .map_err(|e| anyhow!("failed to load tokenizer {:?}: {e}", args.tokenizer_json))?;
    let tokenizer = Arc::new(tokenizer);

    for input in &args.inputs {
        process_file(input, tokenizer.clone(), &args.text_col, args.batch_size)?;
    }

    Ok(())
}

fn process_file(
    input_path: &Path,
    tokenizer: Arc<Tokenizer>,
    text_col: &str,
    batch_size: usize,
) -> Result<()> {
    let input_file = File::open(input_path)
        .with_context(|| format!("failed to open input parquet: {}", input_path.display()))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(input_file)
        .with_context(|| format!("failed to create parquet reader: {}", input_path.display()))?;

    let input_schema = builder.schema().clone();
    if input_schema.index_of(TOKEN_COUNT_COL).is_ok() || input_schema.index_of(OVER_1M_COL).is_ok()
    {
        return Err(anyhow!(
            "input already contains {TOKEN_COUNT_COL:?} or {OVER_1M_COL:?}: {}",
            input_path.display()
        ));
    }

    let total_rows = builder.metadata().file_metadata().num_rows() as u64;
    let reader = builder
        .with_batch_size(batch_size)
        .build()
        .with_context(|| format!("failed to build parquet reader: {}", input_path.display()))?;

    let output_path = output_path_for(input_path);
    let output_file = File::create(&output_path)
        .with_context(|| format!("failed to create output parquet: {}", output_path.display()))?;

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
        let batch = batch_result.with_context(|| "failed to read parquet batch")?;

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
                    .with_context(|| "failed to create parquet writer")?,
            );
        }

        writer
            .as_mut()
            .unwrap()
            .write(&new_batch)
            .with_context(|| "failed to write parquet batch")?;

        pb.inc(new_batch.num_rows() as u64);
    }

    if let Some(writer) = writer {
        writer
            .close()
            .with_context(|| "failed to close parquet writer")?;
    }

    pb.finish_with_message(format!("wrote {}", output_path.display()));
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
        .with_context(|| format!("column {text_col:?} not found"))?;

    let col = batch.column(idx);

    match col.data_type() {
        DataType::Utf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| anyhow!("failed to downcast Utf8 column"))?;
            Ok(Box::new(Utf8Accessor { arr }))
        }
        DataType::LargeUtf8 => {
            let arr = col
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| anyhow!("failed to downcast LargeUtf8 column"))?;
            Ok(Box::new(LargeUtf8Accessor { arr }))
        }
        other => Err(anyhow!(
            "column {text_col:?} must be Utf8 or LargeUtf8, got {other:?}"
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
    fn len(&self) -> usize {
        self.arr.len()
    }

    fn is_null(&self, i: usize) -> bool {
        self.arr.is_null(i)
    }

    fn value(&self, i: usize) -> &str {
        self.arr.value(i)
    }
}

struct LargeUtf8Accessor<'a> {
    arr: &'a LargeStringArray,
}

impl TextAccessor for LargeUtf8Accessor<'_> {
    fn len(&self) -> usize {
        self.arr.len()
    }

    fn is_null(&self, i: usize) -> bool {
        self.arr.is_null(i)
    }

    fn value(&self, i: usize) -> &str {
        self.arr.value(i)
    }
}

fn append_columns(
    batch: RecordBatch,
    token_counts: Vec<i32>,
    over_1m: Vec<bool>,
) -> Result<RecordBatch> {
    if token_counts.len() != batch.num_rows() || over_1m.len() != batch.num_rows() {
        return Err(anyhow!(
            "new column length mismatch: rows={}, tokens={}, over_1m={}",
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
        .with_context(|| "failed to build output record batch")
}
