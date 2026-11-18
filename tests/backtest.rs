use risk_engine::backtest::{christoffersen_independence, kupiec_pof};

#[test]
fn kupiec_does_not_reject_correct_model() {
    // n=1000, expected 1% exceptions, observe exactly 10.
    let r = kupiec_pof(1000, 10, 0.01);
    assert!(r.p_value > 0.05, "p_value={}", r.p_value);
    assert!(!r.rejected);
}

#[test]
fn kupiec_rejects_understated_var() {
    // Model says 1% but reality is ~5%: 50 exceptions in 1000.
    let r = kupiec_pof(1000, 50, 0.01);
    assert!(r.p_value < 0.01, "p_value={}", r.p_value);
    assert!(r.rejected);
}

#[test]
fn christoffersen_flags_clustered_exceptions() {
    // Two clusters of exceptions -- classic violation of independence.
    let mut ex = vec![false; 500];
    for exception in ex.iter_mut().take(20) {
        *exception = true;
    }
    for exception in ex.iter_mut().skip(480).take(20) {
        *exception = true;
    }
    let r = christoffersen_independence(&ex);
    assert!(r.rejected, "p_value={}", r.p_value);
}

#[test]
fn christoffersen_accepts_scattered_exceptions() {
    // Scattered singles -- consistent with independence.
    let mut ex = vec![false; 1000];
    for i in (0..1000).step_by(50) {
        ex[i] = true;
    }
    let r = christoffersen_independence(&ex);
    assert!(!r.rejected, "p_value={}", r.p_value);
}
