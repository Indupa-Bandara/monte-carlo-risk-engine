use risk_engine::data::{AlignedReturns, DataError, Positions};

#[test]
fn positions_align_by_asset_name_and_unheld_assets_get_zero() {
    let returns = AlignedReturns::new(
        vec!["A".into(), "B".into(), "C".into()],
        vec!["2026-01-01".into(), "2026-01-02".into()],
        vec![vec![0.01, 0.02, 0.03], vec![-0.01, 0.01, 0.02]],
    )
    .unwrap();
    let positions = Positions::new(vec!["C".into(), "A".into()], vec![30.0, 10.0]).unwrap();

    assert_eq!(positions.align_to(&returns).unwrap(), vec![10.0, 0.0, 30.0]);
}

#[test]
fn rejects_misaligned_or_non_finite_return_data() {
    let assets = vec!["A".into(), "B".into()];
    let dates = vec!["2026-01-01".into()];

    assert_eq!(
        AlignedReturns::new(assets.clone(), vec![], vec![vec![0.01, 0.02]]).unwrap_err(),
        DataError::DateCountMismatch {
            dates: 0,
            observations: 1,
        }
    );
    assert_eq!(
        AlignedReturns::new(assets.clone(), dates.clone(), vec![vec![0.01]]).unwrap_err(),
        DataError::RowWidthMismatch {
            row: 0,
            expected: 2,
            actual: 1,
        }
    );
    assert_eq!(
        AlignedReturns::new(assets, dates, vec![vec![0.01, f64::NAN]]).unwrap_err(),
        DataError::NonFiniteReturn { row: 0, asset: 1 }
    );
}

#[test]
fn rejects_duplicate_dates_and_unknown_position_assets() {
    let duplicate_dates = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into(), "2026-01-01".into()],
        vec![vec![0.01], vec![0.02]],
    );
    assert_eq!(
        duplicate_dates.unwrap_err(),
        DataError::DuplicateDate("2026-01-01".into())
    );

    let returns = AlignedReturns::new(
        vec!["A".into()],
        vec!["2026-01-01".into()],
        vec![vec![0.01]],
    )
    .unwrap();
    let positions = Positions::new(vec!["B".into()], vec![10.0]).unwrap();
    assert_eq!(
        positions.align_to(&returns).unwrap_err(),
        DataError::UnknownPositionAsset("B".into())
    );
}

#[test]
fn rejects_non_finite_and_mismatched_position_notionals() {
    assert_eq!(
        Positions::new(vec!["A".into()], vec![f64::INFINITY]).unwrap_err(),
        DataError::NonFiniteNotional { index: 0 }
    );
    assert_eq!(
        Positions::new(vec!["A".into()], vec![]).unwrap_err(),
        DataError::PositionLengthMismatch {
            assets: 1,
            notionals: 0,
        }
    );
}
