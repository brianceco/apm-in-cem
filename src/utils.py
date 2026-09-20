"""Utility functions for estimating OU parameters, carrying out sensitivity analysis of transaction cost parameters, computing SPT functionals, plotting stylized facts, and performing MLE for mean-reverting SDD model."""

from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm


class TargetKernels:
    """Grid-offset arrays shared by every step of the frictionless target E_t[lbda_0(u)].

    Writing the forecast horizon as u = t + p * dt and the inner integration variable as
    s = t + q * dt, every kernel entering the target depends on the grid only through the
    offsets p, q and p - q, never on t itself. They are therefore built once at full size
    (p, q <= N) by _target_kernels and sliced to the P = N + 1 - n points still ahead of
    t = n * dt. Rows are indexed by u, columns by s.
    """

    def __init__(
        self,
        delta_reversion: np.ndarray,
        phi_reversion: np.ndarray,
        log_delta_var: np.ndarray,
        inner_kernel: np.ndarray,
    ) -> None:
        self.delta_reversion = (
            delta_reversion  # (N+1,)      exp(-kappa_delta * (u - t))
        )
        self.phi_reversion = phi_reversion  # (N+1,)        exp(-kappa_phi * (u - t))
        self.log_delta_var = log_delta_var  # (N+1,)        Var_t[log delta(u)]
        self.inner_kernel = (
            inner_kernel  # (N+1, N+1)     weighted kernel of the s integral
        )


def target_kernels(
    kappa_phi: float, kappa_delta: float, nu_delta: float, dt: float, N: int
) -> TargetKernels:
    """Build the grid-offset kernels of _get_target once, at the full horizon of N steps."""
    stationary_var = 0.5 * nu_delta**2 / kappa_delta

    offset = dt * np.arange(N + 1)  # u - t at p steps ahead, (N+1,)
    lag = np.tril(offset[:, None] - offset[None, :])  # u - s on p >= q, else 0

    delta_reversion = np.exp(-kappa_delta * offset)  # (N+1,)

    # Cov_t[log delta(u), log delta(s)] for s <= u, i.e. on the lower triangle
    log_delta_cov = stationary_var * (
        np.exp(-kappa_delta * lag) - delta_reversion[:, None] * delta_reversion[None, :]
    )  # (N+1, N+1)

    # left-endpoint weights for the inner integral over s in [t, u]: q ∈ [0, p)
    inner_weights = np.tri(N + 1, N + 1, -1) * dt

    return TargetKernels(
        delta_reversion=delta_reversion,
        phi_reversion=np.exp(-kappa_phi * offset),
        log_delta_var=stationary_var * (1 - delta_reversion**2),
        inner_kernel=inner_weights
        * np.exp(-(kappa_phi + kappa_delta) * lag - 0.5 * log_delta_cov),
    )


def get_target(
    params: dict[str, float],
    kernels: TargetKernels,
    gamma: float,
    phi_t: np.ndarray | float,
    log_delta_t: np.ndarray | float,
    P: int | None = None,
) -> np.ndarray:
    """Compute the forecasted frictionless target E_t[lbda_0(u)] at times u = t + p*dt for p = 0, ..., P-1 for M sample values of diversity phi_t and dispersion delta_t."""
    # params
    kappa_phi = params["kappa_phi"]
    phi_bar = params["phi_bar"]
    nu_phi = params["nu_phi"]
    log_delta_bar = np.log(params["delta_bar"])
    nu_delta = params["nu_delta"]
    rho = params["rho"]

    if P is None:
        P = kernels.inner_kernel.shape[0]

    # kernels
    delta_reversion = kernels.delta_reversion[:P, None]
    phi_reversion = kernels.phi_reversion[:P, None]
    log_delta_var = kernels.log_delta_var[:P, None]
    inner_kernel = kernels.inner_kernel[:P, :P]

    # convert floats to arrays
    phi_t = np.atleast_1d(phi_t)
    log_delta_t = np.atleast_1d(log_delta_t)

    mu_tu = (
        delta_reversion * log_delta_t[None, :] + (1 - delta_reversion) * log_delta_bar
    )  # E_t[log delta(u)]
    E_delta_inv = np.exp(-mu_tu + 0.5 * log_delta_var)  # E_t[1 / delta(u)]
    E_delta_sqrt = np.exp(0.5 * mu_tu + 0.125 * log_delta_var)

    mr_term = (
        kappa_phi * (phi_bar - phi_t)[None, :] * phi_reversion * E_delta_inv / nu_phi**2
    )

    # the s integral runs over the same grid as u, so E_delta_sqrt is reused columnwise
    corr_term = (
        kappa_phi
        * nu_delta
        * rho
        / nu_phi
        * E_delta_inv
        * (inner_kernel @ E_delta_sqrt)
    )

    const_term = (1 + nu_phi**2) / (2 * nu_phi**2)

    target = (mr_term + corr_term + const_term) / (1 + gamma)

    return target.squeeze()


def calibrate_c_delta(diversity_monthly: pd.Series, rd_eql_monthly: pd.Series, log_V_eql: pd.Series, START_DATE: str, WARMUP_END_DATE: str, dt: float, print_summary: bool = True):
    x_full = 0.5 * rd_eql_monthly.iloc[1:].cumsum() * dt
    div_centered = diversity_monthly.diff().cumsum()
    y = (log_V_eql - div_centered).dropna().loc[START_DATE:WARMUP_END_DATE]
    x = x_full.loc[START_DATE:WARMUP_END_DATE]

    lr = sm.OLS(y, x).fit()
    c_delta = lr.params["rd_eql"]
    fit = div_centered + c_delta * x_full
    if print_summary:
        print(f"Regression summary:\n{lr.summary()}")
    return c_delta, fit

def get_ou_params(data: pd.Series, dt: float) -> tuple[float, float, float]:
    """Estimate the parameters of an Ornstein-Uhlenbeck process from a time series using an AR(1) model."""
    ar1_model = sm.tsa.AutoReg(data, lags=1).fit()
    (_, b1) = ar1_model.params
    var = np.var(ar1_model.resid.to_numpy())

    mu = data.mean()  # use in place of AR(1) intercept

    if b1 > 0:
        kappa = -np.log(b1) / dt
        sigma = np.sqrt(var * 2 * kappa / (1 - b1**2))
    else:
        raise ValueError(
            "AR(1) parameter b1 must be positive to convert to OU parameters."
        )

    return kappa, mu, sigma


#################
# SPT utilities
#################


def get_ew_diversity(mkt_wgts: pd.DataFrame) -> pd.Series:
    """
    Compute the negative cross-entropy between the market weights and the uniform distribution.
    """
    nonzero_wgts = mkt_wgts[mkt_wgts > 0]

    return np.sum(
        np.log(nonzero_wgts).divide(nonzero_wgts.notna().sum(axis=1), axis=0),
        axis=1,
    )


def get_dispersion(ret_df: pd.DataFrame, wgts_lag1: pd.DataFrame) -> pd.Series:
    """cross-sectional variance of log-returns"""
    log_ret_df = np.log1p(ret_df.where(ret_df > -1))
    avg_ret = wgts_lag1.multiply(log_ret_df).sum(axis=1)
    realized_dispersion = wgts_lag1.multiply(
        log_ret_df.sub(avg_ret, axis=0).pow(2)
    ).sum(axis=1)
    return realized_dispersion


def get_TC_penalty_rate(
    lbda: pd.Series | pd.DataFrame,
    diversity_qv_rate: pd.Series | pd.DataFrame,
    dt: float,
    Lbda_1: float,
    Lbda_2: float,
) -> pd.Series | pd.DataFrame:
    """Compute the differential dTC(t) of the trading cost penalty for lambda tilt process."""
    rate = lbda.diff() / dt
    TC_1 = 0.5 * Lbda_1 * diversity_qv_rate.multiply(lbda**2, axis=0)
    TC_2 = 0.5 * Lbda_2 * diversity_qv_rate.multiply(rate**2, axis=0)
    tc_rate = TC_1 + TC_2
    tc_rate.iloc[0] = 0
    return tc_rate


#########################
# Stylized fact utilities
#########################


def l2_log_loss(
    params: np.ndarray,
    f: Callable[[np.ndarray, np.ndarray], np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """Sum of squared differences between log observed and log predicted values, fitting the tail on a log scale."""
    y_pred = f(params, x)
    # floor rather than drop the underflowed predictions: masking them out lets a fit whose
    # predictions all underflow to 0 score a perfect loss of 0
    y_pred = np.clip(y_pred, np.finfo(float).tiny, None)
    loss = np.sum((np.log(y) - np.log(y_pred)) ** 2)
    return loss


def plot_cdf_tail(
    ser: pd.Series,
    fit: bool = True,
    bounds: tuple | None = None,
    x0: tuple | None = None,
) -> dict[str, np.ndarray]:
    """Empirical right-tail cdf, optionally with a fitted normal survival function.

    Fitted parameters are ordered (scale, loc); loc is held fixed at `floc` when given.
    """
    data = {}

    # empirical cdf
    # ser = (ser-ser.mean()) / ser.std()
    sorted_ser = np.sort(ser)
    cdf = np.arange(1, len(sorted_ser) + 1) / len(sorted_ser)

    # drop zero and 1 cdf values for log scale
    sorted_ser = sorted_ser[1:-1]
    cdf = cdf[1:-1]

    q_right = np.max([0.9, 1 - 100.0 / ser.size])

    right_tail_mask = sorted_ser > np.quantile(sorted_ser, q_right)
    data["right_tail_x"] = sorted_ser[right_tail_mask]
    data["right_tail_y"] = 1 - cdf[right_tail_mask]

    if fit:
        if x0 is None:
            x0 = (ser.mean(), ser.std())
        if bounds is None:
            # keep the scale parameter strictly positive
            bounds = [
                (None, None),
                (0.1 * ser.std(), 10 * ser.std()),
            ]  # mean, std deviation

        def norm_cdf(params: tuple, x: np.ndarray, upper: bool) -> np.ndarray:
            loc = params[0]
            scale = params[1]
            p = stats.norm.cdf(x, loc=loc, scale=scale)
            return 1 - p if upper else p

        if "right_tail_x" not in data:
            return data
        f = lambda params, x: norm_cdf(params, x, upper=True)
        res = minimize(
            fun=l2_log_loss,
            x0=x0,
            args=(f, data["right_tail_x"], data["right_tail_y"]),
            bounds=bounds,
        )
        if res.success:
            print("Calibrated right-tail parameters:")
            print(f"mu={res.x[0]:.4f}, sigma={res.x[1]:.4f}")
            data["right_tail_fit"] = f(res.x, data["right_tail_x"])
            data["right_tail_params"] = res.x
    return data


#################
# MLE utilities
#################


# define the OU loss function in Pytorch to automatically compute gradients and vectorize computations
def get_cov_mat_torch(
    X_lagged: torch.Tensor,
    lbda: torch.Tensor,
    nu: torch.Tensor,
    rho: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Per-step 2x2 covariance matrices of the (diversity, dispersion) increments under the mean-reverting SDD model."""
    cov = torch.empty((X_lagged.shape[0], 2, 2), dtype=X_lagged.dtype)
    cov[:, 0, 0] = lbda**2 * X_lagged[:, 1] * dt
    cov[:, 0, 1] = lbda * nu * X_lagged[:, 1] ** 1.5 * rho * dt
    cov[:, 1, 0] = cov[:, 0, 1]
    cov[:, 1, 1] = nu**2 * X_lagged[:, 1] ** 2 * dt
    return cov


# use the fact that inverse of 2x2 matrix is easy to compute
def get_inv_and_dets(cov_mats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Invert a batch of 2x2 matrices in closed form, returning the inverses and their determinants."""
    a = cov_mats[:, 0, 0]
    b = cov_mats[:, 0, 1]
    c = cov_mats[:, 1, 0]
    d = cov_mats[:, 1, 1]

    dets = a * d - b * c
    inv_dets = 1.0 / dets

    inv_cov_mats = torch.empty_like(cov_mats)

    inv_cov_mats[:, 0, 0] = d * inv_dets
    inv_cov_mats[:, 0, 1] = -b * inv_dets
    inv_cov_mats[:, 1, 0] = -c * inv_dets
    inv_cov_mats[:, 1, 1] = a * inv_dets

    return inv_cov_mats, dets


def get_drift_phi(
    X_lagged: torch.Tensor, kappa_phi: torch.Tensor, mu_phi: torch.Tensor
) -> torch.Tensor:
    """Drift of diversity, mean-reverting to mu_phi at rate kappa_phi."""
    drift_phi = kappa_phi * (mu_phi - X_lagged[:, 0])
    return drift_phi


def get_drift_delta(
    X_lagged: torch.Tensor,
    kappa_delta: torch.Tensor,
    mu_delta: torch.Tensor,
    nu: torch.Tensor,
) -> torch.Tensor:
    """Drift of dispersion, including the Ito correction from the log dynamics."""
    drift_delta = (
        kappa_delta * (torch.log(mu_delta) - torch.log(X_lagged[:, 1])) + 0.5 * nu**2
    ) * X_lagged[:, 1]
    return drift_delta


def log_likelihood_torch(
    params: torch.Tensor,
    diversity: torch.Tensor,
    dispersion: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Negative log likelihood of the observed diversity and dispersion path under the Euler-discretized SDD model."""
    kappa_phi, mu_phi, lbda, kappa_delta, mu_delta, nu, rho = params
    X = torch.column_stack((diversity, dispersion))
    X_curr = X[1:]
    X_lagged = X[:-1]

    drift_phi = get_drift_phi(X_lagged, kappa_phi, mu_phi)
    drift_delta = get_drift_delta(X_lagged, kappa_delta, mu_delta, nu)
    drift = torch.column_stack((drift_phi, drift_delta))

    means = X_lagged + drift * dt

    cov_mats = get_cov_mat_torch(X_lagged, lbda, nu, rho, dt)

    inv_cov_mats, determinants = get_inv_and_dets(cov_mats)

    diff = (X_curr - means).unsqueeze(2)  # shape (N-1, 2, 1)
    exponent = (
        -0.5 * torch.bmm(torch.bmm(diff.transpose(1, 2), inv_cov_mats), diff).squeeze()
    )
    log_lik = torch.sum(
        -0.5 * torch.log(determinants)
        + exponent
        - torch.log(torch.full_like(determinants, 2 * torch.pi))
    )

    return -log_lik


def log_likelihood_with_grad(
    params: np.ndarray, diversity: np.ndarray, dispersion: np.ndarray, dt: float
) -> tuple[float, np.ndarray]:
    """Negative log likelihood and its autograd gradient, for use as a scipy objective with jac=True."""
    params_torch = torch.tensor(params, dtype=torch.float64, requires_grad=True)
    diversity_torch = torch.tensor(diversity, dtype=torch.float64)
    dispersion_torch = torch.tensor(dispersion, dtype=torch.float64)

    ll = log_likelihood_torch(params_torch, diversity_torch, dispersion_torch, dt)
    ll.backward()
    grad = params_torch.grad.numpy()
    return ll.item(), grad


def get_ci(
    mle_params: dict[str, float],
    div_array: np.ndarray,
    disp_array: np.ndarray,
    dt: float,
    alpha: float = 0.05,
) -> dict[str, tuple[float, float]]:
    """Compute confidence intervals for MLE parameters of mean-reverting SDD models using the Hessian of the log-likelihood function."""
    param_vals = list(mle_params.values())
    optimal_params = torch.tensor(param_vals, dtype=torch.float64, requires_grad=True)
    phi_torch = torch.tensor(div_array, dtype=torch.float64)
    delta_torch = torch.tensor(disp_array, dtype=torch.float64)
    hessian = torch.autograd.functional.hessian(
        lambda params: log_likelihood_torch(params, phi_torch, delta_torch, dt),
        optimal_params,
    )

    try:
        np.linalg.cholesky(hessian)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "Hessian is not positive definite, cannot compute confidence intervals."
        )

    cov_mat = torch.inverse(hessian)
    std_errors = torch.sqrt(torch.diag(cov_mat)).numpy()
    z = stats.norm.ppf(1 - alpha / 2)
    ci_lower = param_vals - z * std_errors
    ci_upper = param_vals + z * std_errors
    ci = {
        param_name: (lower, upper)
        for param_name, lower, upper in zip(mle_params.keys(), ci_lower, ci_upper)
    }
    return ci
