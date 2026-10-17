"""Inference tools for small event samples (about 190 meetings).

- ols:            OLS with HC3 standard errors
- wild_boot:      restricted wild bootstrap (Rademacher) p-value for one coefficient, HC3 t-statistic
- permutation:    permutation p-value: the regressor of interest is shuffled across events
- paired_boot:    event-resampling bootstrap of the difference between two coefficients
- holm:           Holm step-down adjusted p-values
- newey_west:     OLS with HAC (Newey-West) standard errors, for local projections
All functions take plain numpy arrays; X includes a constant column.
"""
import numpy as np
import statsmodels.api as sm


def _hc3_t(X, y, XtX_inv, j):
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (e / (1 - h)) ** 2
    V = XtX_inv @ (X.T * w) @ X @ XtX_inv
    return b[j], b[j] / np.sqrt(V[j, j])


def ols(y, X, names=None) -> dict:
    res = sm.OLS(y, X).fit(cov_type="HC3")
    names = names or [f"x{i}" for i in range(X.shape[1])]
    return {"n": int(len(y)), "r2": float(res.rsquared), "coef": dict(zip(names, res.params.tolist())),
            "se": dict(zip(names, res.bse.tolist())), "t": dict(zip(names, res.tvalues.tolist())),
            "p": dict(zip(names, res.pvalues.tolist()))}


def wild_boot(y, X, j, B=9999, seed=0, alternative="two-sided") -> float:
    """Restricted wild bootstrap: impose beta_j = 0, resample residuals with Rademacher weights."""
    rng = np.random.default_rng(seed)
    XtX_inv = np.linalg.inv(X.T @ X)
    _, t_obs = _hc3_t(X, y, XtX_inv, j)
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, e_r = Xr @ br, y - Xr @ br
    ts = np.empty(B)
    for b in range(B):
        v = rng.choice([-1.0, 1.0], size=len(y))
        ts[b] = _hc3_t(X, fit_r + e_r * v, XtX_inv, j)[1]
    if alternative == "greater":
        return float((1 + np.sum(ts >= t_obs)) / (B + 1))
    if alternative == "less":
        return float((1 + np.sum(ts <= t_obs)) / (B + 1))
    return float((1 + np.sum(np.abs(ts) >= abs(t_obs))) / (B + 1))


def permutation(y, X, j, B=9999, seed=0, alternative="two-sided") -> float:
    """Shuffle column j across events (breaks any link between it and y, keeps its distribution)."""
    rng = np.random.default_rng(seed)
    XtX_inv = np.linalg.inv(X.T @ X)
    _, t_obs = _hc3_t(X, y, XtX_inv, j)
    ts = np.empty(B)
    Xp = X.copy()
    for b in range(B):
        Xp[:, j] = rng.permutation(X[:, j])
        ts[b] = _hc3_t(Xp, y, np.linalg.inv(Xp.T @ Xp), j)[1]
    if alternative == "greater":
        return float((1 + np.sum(ts >= t_obs)) / (B + 1))
    if alternative == "less":
        return float((1 + np.sum(ts <= t_obs)) / (B + 1))
    return float((1 + np.sum(np.abs(ts) >= abs(t_obs))) / (B + 1))


def freedman_lane(y, X, j, B=9999, seed=0, alternative="two-sided") -> float:
    """Freedman-Lane permutation: permute the residuals of the reduced model (y on the other regressors) and add
    them back to its fitted values; the regressor of interest stays attached to its own event."""
    rng = np.random.default_rng(seed)
    XtX_inv = np.linalg.inv(X.T @ X)
    _, t_obs = _hc3_t(X, y, XtX_inv, j)
    Xr = np.delete(X, j, axis=1)
    br = np.linalg.lstsq(Xr, y, rcond=None)[0]
    fit_r, e_r = Xr @ br, y - Xr @ br
    ts = np.array([_hc3_t(X, fit_r + rng.permutation(e_r), XtX_inv, j)[1] for _ in range(B)])
    if alternative == "greater":
        return float((1 + np.sum(ts >= t_obs)) / (B + 1))
    if alternative == "less":
        return float((1 + np.sum(ts <= t_obs)) / (B + 1))
    return float((1 + np.sum(np.abs(ts) >= abs(t_obs))) / (B + 1))


def paired_boot(y, XA, XB, j, B=9999, seed=0) -> dict:
    """Bootstrap (resampling events) of beta_A[j] - beta_B[j] estimated on the same resampled events."""
    rng = np.random.default_rng(seed)
    n = len(y)
    bA = np.linalg.lstsq(XA, y, rcond=None)[0][j]
    bB = np.linalg.lstsq(XB, y, rcond=None)[0][j]
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = np.linalg.lstsq(XA[i], y[i], rcond=None)[0][j] - np.linalg.lstsq(XB[i], y[i], rcond=None)[0][j]
    return {"diff": float(bA - bB), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, 2 * min(np.mean(d <= 0), np.mean(d >= 0)))),
            "p_greater": float(np.mean(d <= 0))}


def holm(pvals) -> list:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def newey_west(y, X, lags, names=None) -> dict:
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    names = names or [f"x{i}" for i in range(X.shape[1])]
    return {"n": int(len(y)), "coef": dict(zip(names, res.params.tolist())), "se": dict(zip(names, res.bse.tolist())),
            "p": dict(zip(names, res.pvalues.tolist()))}
