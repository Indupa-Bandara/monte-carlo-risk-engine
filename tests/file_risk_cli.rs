use arrow_array::{Float64Array, RecordBatch, StringArray};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::fs::{self, File};
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_path(extension: &str) -> PathBuf {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!("risk-engine-cli-{unique}.{extension}"))
}

#[test]
fn cli_reads_files_and_reports_var_and_es() {
    let returns_path = temp_path("parquet");
    let positions_path = temp_path("csv");
    let schema = Arc::new(Schema::new(vec![
        Field::new("date", DataType::Utf8, false),
        Field::new("A", DataType::Float64, false),
        Field::new("B", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(vec![
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
                "2026-01-04",
            ])),
            Arc::new(Float64Array::from(vec![0.01, 0.0, -0.01, -0.10])),
            Arc::new(Float64Array::from(vec![0.01, -0.01, 0.0, 0.02])),
        ],
    )
    .unwrap();
    let mut writer =
        ArrowWriter::try_new(File::create(&returns_path).unwrap(), schema, None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();
    fs::write(&positions_path, "asset,notional\nA,10000\nB,20000\n").unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_risk-engine"))
        .arg(&returns_path)
        .arg(&positions_path)
        .args([
            "--scenarios",
            "5000",
            "--seed",
            "10",
            "--chunk-size",
            "500",
            "--contributions",
            "--stress",
        ])
        .output()
        .unwrap();

    let backtest_output = Command::new(env!("CARGO_BIN_EXE_risk-engine"))
        .arg(&returns_path)
        .arg(&positions_path)
        .args([
            "--scenarios",
            "5000",
            "--seed",
            "10",
            "--chunk-size",
            "500",
            "--backtest-window",
            "3",
        ])
        .output()
        .unwrap();

    let comparison_output = Command::new(env!("CARGO_BIN_EXE_risk-engine"))
        .arg(&returns_path)
        .arg(&positions_path)
        .args([
            "--scenarios",
            "5000",
            "--seed",
            "10",
            "--chunk-size",
            "500",
            "--backtest-window",
            "3",
            "--model",
            "all",
        ])
        .output()
        .unwrap();

    fs::remove_file(&returns_path).unwrap();
    fs::remove_file(&positions_path).unwrap();

    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("VaR:"));
    assert!(stdout.contains("ES:"));
    assert!(stdout.contains("VaR 95% CI"));
    assert!(stdout.contains("ES 95% CI"));
    assert!(stdout.contains("Historical observations: 4"));
    assert!(stdout.contains("Historical component ES"));
    assert!(stdout.contains("Worst historical one-day replay scenarios"));
    assert!(stdout.contains("uniform -20% shock"));
    assert!(stdout.contains("uniform -40% shock"));

    assert!(
        backtest_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&backtest_output.stderr)
    );
    let backtest_stdout = String::from_utf8_lossy(&backtest_output.stdout);
    assert!(backtest_stdout.contains("Rolling out-of-sample forecasts"));
    assert!(backtest_stdout.contains("Kupiec POF p-value"));
    assert!(backtest_stdout.contains("2026-01-04"));

    assert!(
        comparison_output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&comparison_output.stderr)
    );
    let comparison_stdout = String::from_utf8_lossy(&comparison_output.stdout);
    assert!(comparison_stdout.contains("Model: Historical simulation"));
    assert!(comparison_stdout.contains("Model: EWMA normal"));
    assert!(comparison_stdout.contains("Model: Normal Monte Carlo"));
    assert!(comparison_stdout.contains("Model: Student-t Monte Carlo"));
}

#[test]
fn cli_rejects_invalid_confidence() {
    let output = Command::new(env!("CARGO_BIN_EXE_risk-engine"))
        .args(["returns.parquet", "positions.csv", "--confidence", "1.0"])
        .output()
        .unwrap();

    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("confidence"));
}

#[test]
fn cli_reports_the_missing_input_path() {
    let returns_path = temp_path("missing.parquet");
    let output = Command::new(env!("CARGO_BIN_EXE_risk-engine"))
        .arg(&returns_path)
        .arg("positions.csv")
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("could not read returns Parquet file"));
    assert!(stderr.contains(&returns_path.display().to_string()));
}
