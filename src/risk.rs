#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskEstimate {
    pub var: f64,
    pub es: f64,
    pub var_ci_95: Option<ConfidenceInterval>,
    pub es_ci_95: Option<ConfidenceInterval>,
}

/// Compute VaR and ES as positive loss magnitudes from a P&L sample.
/// P&L is positive for profit and negative for loss. Sorts `pnls` in place.
pub fn var_es(pnls: &mut [f64], alpha: f64) -> (f64, f64) {
    let (var, es, _, _) = sorted_var_es(pnls, alpha);
    (var, es)
}

/// Compute VaR, ES, and approximate 95% Monte Carlo confidence intervals.
///
/// The VaR interval uses a normal approximation to the order-statistic ranks.
/// The ES interval uses a normal approximation for the mean of the tail losses.
/// Intervals are omitted when fewer than 30 scenarios are in the tail.
pub fn var_es_with_ci(pnls: &mut [f64], alpha: f64) -> RiskEstimate {
    let (var, es, tail_n, n) = sorted_var_es(pnls, alpha);
    if tail_n < 30 {
        return RiskEstimate {
            var,
            es,
            var_ci_95: None,
            es_ci_95: None,
        };
    }

    let p = 1.0 - alpha;
    let n_f = n as f64;
    let mean_rank = n_f * p;
    let rank_std_dev = (n_f * p * alpha).sqrt();
    let z_95 = 1.959963984540054;
    let low_rank = (mean_rank - z_95 * rank_std_dev)
        .floor()
        .clamp(0.0, (n - 1) as f64) as usize;
    let high_rank = (mean_rank + z_95 * rank_std_dev)
        .ceil()
        .clamp(0.0, (n - 1) as f64) as usize;
    let var_ci_95 = Some(ConfidenceInterval {
        lower: -pnls[high_rank],
        upper: -pnls[low_rank],
    });

    let tail_mean_loss = es;
    let squared_deviations = pnls[..tail_n]
        .iter()
        .map(|pnl| {
            let deviation = -pnl - tail_mean_loss;
            deviation * deviation
        })
        .sum::<f64>();
    let tail_variance = squared_deviations / (tail_n - 1) as f64;
    let es_half_width = z_95 * (tail_variance / tail_n as f64).sqrt();
    let es_ci_95 = es_half_width.is_finite().then_some(ConfidenceInterval {
        lower: es - es_half_width,
        upper: es + es_half_width,
    });

    RiskEstimate {
        var,
        es,
        var_ci_95,
        es_ci_95,
    }
}

fn sorted_var_es(pnls: &mut [f64], alpha: f64) -> (f64, f64, usize, usize) {
    assert!((0.0..1.0).contains(&alpha));
    if pnls.is_empty() {
        return (0.0, 0.0, 0, 0);
    }
    pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = pnls.len();
    let tail_n = (((1.0 - alpha) * n as f64).floor() as usize).max(1).min(n);
    let var_idx = tail_n - 1;
    let var = -pnls[var_idx];
    let es = -pnls[..tail_n].iter().sum::<f64>() / tail_n as f64;
    (var, es, tail_n, n)
}
