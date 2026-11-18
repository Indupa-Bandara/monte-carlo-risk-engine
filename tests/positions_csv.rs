use risk_engine::data::{AlignedReturns, DataError, PositionCsvError, Positions};
use std::io::Cursor;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn parses_positions_and_aligns_them_by_asset_name() {
    let csv = "notional,asset\n125.5,\"ACME, Inc.\"\n-20,B\n";
    let positions = Positions::from_csv_reader(Cursor::new(csv)).unwrap();
    let returns = AlignedReturns::new(
        vec!["B".into(), "ACME, Inc.".into()],
        vec!["2026-01-01".into()],
        vec![vec![0.01, 0.02]],
    )
    .unwrap();

    assert_eq!(positions.align_to(&returns).unwrap(), vec![-20.0, 125.5]);
}

#[test]
fn reads_positions_from_a_csv_path() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("risk-engine-positions-{unique}.csv"));
    std::fs::write(&path, "asset,notional\nA,10\n").unwrap();

    let result = Positions::from_csv_path(&path);
    std::fs::remove_file(&path).unwrap();
    let positions = result.unwrap();
    let returns = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into()],
        vec![vec![0.01]],
    )
    .unwrap();

    assert_eq!(positions.align_to(&returns).unwrap(), vec![10.0]);
}

#[test]
fn rejects_missing_or_duplicate_headers() {
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("symbol,value\nA,10\n")),
        Err(PositionCsvError::MissingHeader("asset"))
    ));
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("asset,asset,notional\nA,A,10\n")),
        Err(PositionCsvError::DuplicateHeader("asset"))
    ));
}

#[test]
fn reports_invalid_cells_and_duplicate_assets() {
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("asset,notional\nA,\n")),
        Err(PositionCsvError::MissingValue {
            row: 2,
            column: "notional"
        })
    ));
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("asset,notional\nA,abc\n")),
        Err(PositionCsvError::InvalidNotional { row: 2, .. })
    ));
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("asset,notional\nA,NaN\n")),
        Err(PositionCsvError::NonFiniteNotional { row: 2, .. })
    ));
    assert!(matches!(
        Positions::from_csv_reader(Cursor::new("asset,notional\nA,10\nA,20\n")),
        Err(PositionCsvError::Data(DataError::DuplicateAssetName(name))) if name == "A"
    ));
}
