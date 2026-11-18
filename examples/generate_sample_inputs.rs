use arrow_array::{Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::error::Error;
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::Arc;

const OBSERVATION_COUNT: usize = 252;

fn main() -> Result<(), Box<dyn Error>> {
    let output_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("examples/sample_data"));
    fs::create_dir_all(&output_dir)?;

    let returns_path = output_dir.join("returns.parquet");
    let positions_path = output_dir.join("positions.csv");
    let schema = Arc::new(Schema::new(vec![
        Field::new("date", DataType::Utf8, false),
        Field::new("A", DataType::Float64, false),
        Field::new("B", DataType::Float64, false),
    ]));
    let mut dates = Vec::with_capacity(OBSERVATION_COUNT);
    let mut returns_a = Vec::with_capacity(OBSERVATION_COUNT);
    let mut returns_b = Vec::with_capacity(OBSERVATION_COUNT);
    let mut random_state = 42_u64;
    let (mut year, mut month, mut day) = (2026, 1, 1);
    for index in 0..OBSERVATION_COUNT {
        dates.push(next_date(&mut year, &mut month, &mut day));

        let market = (next_unit(&mut random_state) - 0.5) * 0.02;
        let idiosyncratic_a = (next_unit(&mut random_state) - 0.5) * 0.01;
        let idiosyncratic_b = (next_unit(&mut random_state) - 0.5) * 0.012;
        let mut asset_a = market + idiosyncratic_a;
        let mut asset_b = 0.7 * market + idiosyncratic_b;
        if index == 80 || index == 180 {
            asset_a -= 0.05;
            asset_b -= 0.07;
        }
        returns_a.push(asset_a);
        returns_b.push(asset_b);
    }

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(dates)),
            Arc::new(Float64Array::from(returns_a)),
            Arc::new(Float64Array::from(returns_b)),
        ],
    )?;
    let mut writer = ArrowWriter::try_new(File::create(&returns_path)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    fs::write(&positions_path, "asset,notional\nA,10000\nB,20000\n")?;

    println!("Wrote {}", returns_path.display());
    println!("Wrote {}", positions_path.display());
    println!("Generated {OBSERVATION_COUNT} deterministic synthetic observations");
    Ok(())
}

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1);
    ((*state >> 11) as f64) / ((1_u64 << 53) as f64)
}

fn next_date(year: &mut u32, month: &mut u32, day: &mut u32) -> String {
    let date = format!("{year:04}-{month:02}-{day:02}");
    let days_in_month = match *month {
        2 if is_leap_year(*year) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };

    *day += 1;
    if *day > days_in_month {
        *day = 1;
        *month += 1;
        if *month > 12 {
            *month = 1;
            *year += 1;
        }
    }

    date
}

fn is_leap_year(year: u32) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}
