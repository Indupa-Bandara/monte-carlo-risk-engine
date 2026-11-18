/// Cholesky decomposition of a symmetric positive-definite matrix `a`
/// stored row-major as a flat `n*n` slice. Returns `L` (lower triangular),
/// row-major, such that `a = L L^T`. Returns `None` if `a` is not SPD.
pub fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(a.len(), n * n);
    let mut l = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if sum <= 0.0 {
                    return None;
                }
                l[i * n + j] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Add a small diagonal jitter until Cholesky succeeds. Useful for
/// empirical covariance matrices that are numerically near-singular.
pub fn cholesky_with_jitter(a: &[f64], n: usize, max_jitter: f64) -> Option<Vec<f64>> {
    if let Some(l) = cholesky(a, n) {
        return Some(l);
    }
    let mut a2 = a.to_vec();
    let mut eps = 1e-10f64;
    while eps <= max_jitter {
        for i in 0..n {
            a2[i * n + i] = a[i * n + i] + eps;
        }
        if let Some(l) = cholesky(&a2, n) {
            return Some(l);
        }
        eps *= 10.0;
    }
    None
}
