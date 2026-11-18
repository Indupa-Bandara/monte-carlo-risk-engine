use arrow_array::{Date32Array, Date64Array, Float32Array, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use risk_engine::data::{AlignedReturns, ParquetReturnsError};
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_parquet_path() -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("risk-engine-returns-{unique}.parquet"))
}

fn write_batch(path: &Path, batch: &RecordBatch) {
    let file = File::create(path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(batch).unwrap();
    writer.close().unwrap();
}

#[test]
fn reads_wide_parquet_returns_in_asset_column_order() {
    let path = temp_parquet_path();
    let schema = Arc::new(Schema::new(vec![
        Field::new("B", DataType::Float32, true),
        Field::new("date", DataType::Utf8, false),
        Field::new("A", DataType::Float64, true),
    ]));
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Float32Array::from(vec![0.01])),
            Arc::new(StringArray::from(vec!["2026-01-01"])),
            Arc::new(Float64Array::from(vec![0.03])),
        ],
    )
    .unwrap();
    let second = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float32Array::from(vec![0.02])),
            Arc::new(StringArray::from(vec!["2026-01-02"])),
            Arc::new(Float64Array::from(vec![0.04])),
        ],
    )
    .unwrap();

    let file = File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, first.schema(), None).unwrap();
    writer.write(&first).unwrap();
    writer.write(&second).unwrap();
    writer.close().unwrap();

    let result = AlignedReturns::from_parquet_path(&path);
    fs::remove_file(&path).unwrap();
    let returns = result.unwrap();

    assert_eq!(returns.asset_names(), &["B", "A"]);
    assert_eq!(returns.dates(), &["2026-01-01", "2026-01-02"]);
    for (row, expected) in returns
        .values()
        .iter()
        .zip([vec![0.01, 0.03], vec![0.02, 0.04]])
    {
        for (actual, expected) in row.iter().zip(expected) {
            assert!((*actual - expected).abs() < 1e-8);
        }
    }
}

#[test]
fn reads_native_parquet_date_columns() {
    let path = temp_parquet_path();
    let date32_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("date", DataType::Date32, false),
            Field::new("A", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Date32Array::from(vec![0, -1])),
            Arc::new(Float64Array::from(vec![0.01, 0.02])),
        ],
    )
    .unwrap();
    write_batch(&path, &date32_batch);
    let date32_returns = AlignedReturns::from_parquet_path(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(date32_returns.dates(), &["1970-01-01", "1969-12-31"]);

    let path = temp_parquet_path();
    let date64_batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("date", DataType::Date64, false),
            Field::new("A", DataType::Float64, false),
        ])),
        vec![
            Arc::new(Date64Array::from(vec![86_400_000, -1])),
            Arc::new(Float64Array::from(vec![0.01, 0.02])),
        ],
    )
    .unwrap();
    write_batch(&path, &date64_batch);
    let date64_returns = AlignedReturns::from_parquet_path(&path).unwrap();
    fs::remove_file(&path).unwrap();
    assert_eq!(date64_returns.dates(), &["1970-01-02", "1969-12-31"]);
}

#[test]
fn rejects_null_return_cells() {
    let path = temp_parquet_path();
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("date", DataType::Utf8, true),
            Field::new("A", DataType::Float64, true),
        ])),
        vec![
            Arc::new(StringArray::from(vec![Some("2026-01-01")])),
            Arc::new(Float64Array::from(vec![None])),
        ],
    )
    .unwrap();
    write_batch(&path, &batch);

    let result = AlignedReturns::from_parquet_path(&path);
    fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(ParquetReturnsError::NullValue {
            row: 0,
            column
        }) if column == "A"
    ));
}

#[test]
fn rejects_missing_date_and_unsupported_return_types() {
    let path = temp_parquet_path();
    let no_date = RecordBatch::try_new(
        Arc::new(Schema::new(vec![Field::new("A", DataType::Float64, false)])),
        vec![Arc::new(Float64Array::from(vec![0.01]))],
    )
    .unwrap();
    write_batch(&path, &no_date);
    let result = AlignedReturns::from_parquet_path(&path);
    fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(ParquetReturnsError::MissingDateColumn)
    ));

    let path = temp_parquet_path();
    let unsupported = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("date", DataType::Utf8, false),
            Field::new("A", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["2026-01-01"])),
            Arc::new(arrow_array::Int64Array::from(vec![1])),
        ],
    )
    .unwrap();
    write_batch(&path, &unsupported);
    let result = AlignedReturns::from_parquet_path(&path);
    fs::remove_file(&path).unwrap();
    assert!(matches!(
        result,
        Err(ParquetReturnsError::UnsupportedColumnType { column, .. }) if column == "A"
    ));
}
