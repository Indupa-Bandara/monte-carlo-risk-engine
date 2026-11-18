use arrow_array::{ArrayRef, Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::error::Error;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let input_path = args.next().map(PathBuf::from).ok_or(
        "usage: cargo run --example binance_returns_to_parquet -- <returns.csv> <returns.parquet>",
    )?;
    let output_path = args.next().map(PathBuf::from).ok_or(
        "usage: cargo run --example binance_returns_to_parquet -- <returns.csv> <returns.parquet>",
    )?;
    if args.next().is_some() {
        return Err(
            "usage: cargo run --example binance_returns_to_parquet -- <returns.csv> <returns.parquet>"
                .into(),
        );
    }

    let mut reader = csv::Reader::from_path(&input_path)?;
    let headers = reader.headers()?.clone();
    if headers.len() < 2 {
        return Err(
            "returns CSV must contain a `date` column and at least one asset column".into(),
        );
    }
    let date_header = headers
        .get(0)
        .unwrap_or_default()
        .trim_start_matches('\u{feff}');
    if date_header != "date" {
        return Err("the first returns CSV column must be named `date`".into());
    }
    let asset_names: Vec<String> = headers.iter().skip(1).map(str::to_owned).collect();
    let schema = Arc::new(Schema::new(
        std::iter::once(Field::new("date", DataType::Utf8, false))
            .chain(
                asset_names
                    .iter()
                    .map(|name| Field::new(name, DataType::Float64, false)),
            )
            .collect::<Vec<_>>(),
    ));

    let mut dates = Vec::new();
    let mut values = vec![Vec::new(); asset_names.len()];
    for (row_index, record) in reader.records().enumerate() {
        let record = record?;
        if record.len() != asset_names.len() + 1 {
            return Err(format!("row {} has the wrong number of columns", row_index + 2).into());
        }
        dates.push(record[0].to_owned());
        for (asset_index, value) in values.iter_mut().enumerate() {
            let parsed = record[asset_index + 1].parse::<f64>()?;
            if !parsed.is_finite() {
                return Err(format!(
                    "non-finite return in row {}, asset {}",
                    row_index + 2,
                    asset_names[asset_index]
                )
                .into());
            }
            value.push(parsed);
        }
    }
    if dates.is_empty() {
        return Err("returns CSV contains no observations".into());
    }

    let mut columns: Vec<ArrayRef> = vec![Arc::new(StringArray::from(dates))];
    columns.extend(
        values
            .into_iter()
            .map(|asset_values| Arc::new(Float64Array::from(asset_values)) as ArrayRef),
    );
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(&output_path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    println!("Wrote {}", output_path.display());
    Ok(())
}
