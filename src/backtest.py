"""Backtesting utilities for optimal lambda-tilt processes. Implement optimal lambda-tilt processes for mean-reverting and trending SDD models and run backtests on real monthly data, comparing to equal-weighted and market portfolios."""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model
from IPython.display import clear_output
from scipy.optimize import minimize, root_scalar

from paths import RESULTS, relative_path
from utils import (
    get_ou_params,
    get_target,
    get_TC_penalty_rate,
    target_kernels,
)

DAYS_PER_MONTH = 21


def ff_regression(
    X: pd.DataFrame, y: pd.Series, save_results: bool = True
) -> pd.DataFrame:
    """Regress portfolio returns y on the factor returns X with an intercept, returning coefficients and t-stats."""
    N = X.shape[1]
    X = X.copy()
    X["alpha"] = np.ones(len(X))
    model = sm.OLS(y, X).fit()
    tstats = model.tvalues
    results = pd.DataFrame({"coef": model.params, "tstat": tstats}, index=X.columns)
    print(f" R-squared: {model.rsquared:.4f}")
    print(f" Annualized alpha: {model.params['alpha'] * 12:.4f}")
    print(results.round(3))
    if save_results:
        os.makedirs(RESULTS / "ff_regression", exist_ok=True)
        save_path = RESULTS / "ff_regression" / f"ff{N}_{y.name}_results.csv"
        results.to_csv(save_path)
        print(f"Saved regression results to {relative_path(save_path)}")
    return results


class LbdaSignal:
    """Base class for the frictionless allocation signal lbda_0. Subclasses estimate the diversity drift and
    variance features over the warmup window; the signal itself is their common drift-over-variance ratio."""

    def __init__(
        self,
        diversity: pd.Series,
        dispersion: pd.Series,
        dt: float,
        warmup_dates: tuple[str, str],
        horizon: int,
    ) -> None:
        self.diversity = diversity
        self.dispersion = dispersion
        self.dt = dt
        self.warmup_dates = warmup_dates
        self.horizon = horizon
        self.features = None
        self.params = None
        self.signal_0 = None

    def _get_features(self) -> tuple[dict, dict]:
        """Estimate the features (and model parameters) the signal is built from."""
        raise NotImplementedError("Subclasses must implement the _get_features method.")

    def _get_signal_0(self) -> pd.Series:
        """Build the frictionless signal lbda_0 as the ratio of active drift to active variance of the calibrated features."""
        if self.features is None:
            raise ValueError(
                "Features have not been estimated yet. Please call calibrate_signal() first."
            )

        b_phi, var_phi, dispersion = (
            self.features["b_phi"],
            self.features["var_phi"],
            self.features["dispersion"],
        )

        signal = (b_phi + 0.5 * (dispersion + var_phi)) / var_phi

        return signal

    def calibrate_signal(self) -> None:
        """Calibrate the features of the signal over the warmup window and construct optimal frictionless lambda tilt process."""
        self.features, self.params = self._get_features()
        self.signal_0 = self._get_signal_0()

    def get_lbda(self, gamma: float, Lbda_1: float, Lbda_2: float) -> pd.Series:
        """Compute the optimal lambda given risk aversion gamma and transaction cost penalty coefficients Lbda_1 and Lbda_2."""
        if not (self.params and self.features):
            raise ValueError(
                "Model parameters must be set by calling .calibrate_signal()."
            )

        if Lbda_2 <= 0:
            raise ValueError("Lbda_2 must be positive.")


class LbdaMeanReversion(LbdaSignal):
    """Lambda signal for mean-reverting SDD model, using OU model of diversity with variance proportional to dispersion"""

    def __init__(
        self,
        diversity: pd.Series,
        dispersion: pd.Series,
        diversity_qv: pd.Series,
        dt: float,
        warmup_dates: tuple[str, str],
        horizon: int,
    ) -> None:
        super().__init__(
            diversity=diversity,
            dispersion=dispersion,
            dt=dt,
            warmup_dates=warmup_dates,
            horizon=horizon,
        )

        self.diversity_qv = diversity_qv

    def _get_features(self) -> tuple[dict, dict]:
        """Fit OU parameters to diversity and to log dispersion over the warmup window, returning the signal features and the fitted parameters."""
        ws, we = self.warmup_dates

        phi_IS = self.diversity[ws:we]
        delta_IS = self.dispersion[ws:we]

        features = {}
        params = {}

        features["dispersion"] = self.dispersion

        kappa_phi, phi_bar, _ = get_ou_params(phi_IS, self.dt)

        features["b_phi"] = kappa_phi * (phi_bar - self.diversity)

        params["kappa_phi"] = kappa_phi
        params["phi_bar"] = phi_bar

        div_qv_IS = self.diversity_qv[ws:we]
        nu_phi = np.sqrt(np.mean(div_qv_IS / delta_IS))

        features["var_phi"] = nu_phi**2 * self.dispersion

        params["nu_phi"] = nu_phi

        params["kappa_delta"], log_delta_bar, params["nu_delta"] = get_ou_params(
            np.log(delta_IS), self.dt
        )

        params["delta_bar"] = np.exp(log_delta_bar)

        params["rho"] = phi_IS.diff().corr(np.log(delta_IS).diff())

        return features, params

    def get_lbda(self, gamma: float, Lbda_1: float, Lbda_2: float) -> pd.Series:
        """Compute the optimal tilt lambda(t) as an exponential weighted average of lambda_0(t) and aim process."""

        phi = self.diversity
        dispersion = self.dispersion

        M = self.horizon

        C_1 = (1 + gamma) / (1 + gamma + Lbda_1)
        C_2 = np.sqrt((1 + gamma + Lbda_1) / Lbda_2)

        N = len(phi)

        aim_lbda = np.zeros(N)

        dt = self.dt

        T = M * dt
        u_grid = np.linspace(0, T - dt, M)  # (0, dt, ..., (M-1)*dt)

        wgts = C_2 * np.cosh(C_2 * (T - u_grid)) / np.sinh(C_2 * T) * dt  # (M,)
        wgts /= wgts.sum()

        kernels = target_kernels(
            self.params["kappa_phi"],
            self.params["kappa_delta"],
            self.params["nu_delta"],
            dt,
            M - 1,
        )

        for n in range(N):
            phi_n = phi.iloc[n]
            log_delta_n = np.log(dispersion.iloc[n])

            target = get_target(
                params=self.params,
                kernels=kernels,
                gamma=gamma,
                phi_t=phi_n,
                log_delta_t=log_delta_n,
                P=M,
            )

            aim_lbda[n] = C_1 * np.dot(wgts, target)

        aim_lbda = pd.Series(aim_lbda, index=phi.index)

        alpha = 1 - np.cosh(C_2 * (T - dt)) / np.cosh(C_2 * T)

        lbda = aim_lbda.ewm(alpha=alpha, adjust=False).mean()
        return lbda


class LbdaTrend(LbdaSignal):
    """Lambda signal for trending SDD model, using AR(1)-GARCH(1,1) model of diversity with EWMA mean."""

    def __init__(
        self,
        diversity: pd.Series,
        dispersion: pd.Series,
        dt: float,
        warmup_dates: tuple[str, str],
        h_phi: int,
        horizon: int,
    ) -> None:
        # daily diversity and dispersion
        super().__init__(
            diversity, dispersion, dt=dt, warmup_dates=warmup_dates, horizon=horizon
        )
        self.h_phi = h_phi

    def _fit_garch_volatility(
        self, b_phi: pd.Series
    ) -> tuple[pd.Series, dict[str, float]]:
        """Fit a zero-mean GARCH(1,1) to the detrended daily diversity increments over the warmup window and aggregate its one-step forecasts into the variance of monthly change in diversity."""
        ws, we = self.warmup_dates
        phi_diff = (self.diversity.diff() - b_phi.shift(1)).dropna().loc[ws:]
        C = phi_diff.loc[ws:we].std()
        phi_diff = phi_diff / C  # rescale for garch fitting

        gm = arch_model(
            phi_diff, vol="GARCH", p=1, o=0, q=1, dist="normal", mean="Zero"
        )
        res = gm.fit(last_obs=we, disp="off")
        alpha, beta, omega = (
            res.params["alpha[1]"],
            res.params["beta[1]"],
            res.params["omega"] * C**2,
        )
        apb = alpha + beta
        v_bar = omega / (1 - apb)

        fcst = res.forecast(horizon=1, start=phi_diff.index[0], reindex=True)
        v = fcst.variance["h.1"].resample("ME").last() * C**2
        v_arr = v.to_numpy()

        alpha_phi = 1 - 2 ** (-1 / self.h_phi)

        i_arr = np.arange(1, DAYS_PER_MONTH + 1)
        wgts = (1 + alpha_phi * (DAYS_PER_MONTH - i_arr)) ** 2
        v_fwd = v_bar + apb ** (i_arr[None, :] - 1) * (v_arr[:, None] - v_bar)
        var_phi = v_fwd @ wgts / self.dt

        var_phi = pd.Series(var_phi, name="var_phi", index=v.index)

        garch_params = {
            "omega": omega,
            "alpha_v": alpha,
            "beta_v": beta,
            "v_bar": v_bar,
        }

        return var_phi, garch_params

    def _get_features(self) -> tuple[dict, dict]:
        """Get EWMA mean of changes in diversity and fit GARCH(1,1) volatility, returning the signal features and the fitted parameters."""
        features = {}

        features["dispersion"] = self.dispersion

        b_phi_daily = (
            self.diversity.diff()
            .dropna()
            .ewm(halflife=self.h_phi, min_periods=self.h_phi, adjust=True)
            .mean()
        )
        features["b_phi"] = b_phi_daily.resample("ME").last() * 252

        features["var_phi"], garch_params = self._fit_garch_volatility(b_phi_daily)

        return features, garch_params

    def get_lbda(self, gamma: float, Lbda_1: float, Lbda_2: float) -> pd.Series:
        """Compute the optimal tilt lambda(t) as an exponential weighted average of lambda_0(t) and aim process."""
        C_2 = np.sqrt((1 + gamma + Lbda_1) / Lbda_2)

        M = self.horizon
        T = M * self.dt
        alpha = 1 - np.cosh(C_2 * (T - self.dt)) / np.cosh(C_2 * T)

        aim_lbda = self.signal_0 / (1 + Lbda_1 + gamma)

        lbda = aim_lbda.ewm(alpha=alpha, adjust=False).mean()
        return lbda


def get_ir_pct(
    wealth_series: pd.Series, benchmark_series: pd.Series, dt: float
) -> float:
    """Annualized information ratio of the portfolio against the benchmark, computed from simple returns."""
    active_ret = wealth_series.pct_change() - benchmark_series.pct_change()
    if np.all(active_ret.dropna() == 0):
        return 0.0
    active_ret_mean = active_ret.mean() * 1 / dt
    active_ret_vol = active_ret.std() * np.sqrt(1 / dt)
    active_ret_ir = active_ret_mean / active_ret_vol
    return active_ret_ir


def get_mdd(wealth_series: pd.Series) -> float:
    """Maximum drawdown of the wealth series, as a fraction of its running maximum."""
    rolling_max = wealth_series.cummax()
    drawdown = (
        rolling_max - wealth_series
    ) / rolling_max  # percentage drawdown relative to rolling max
    max_drawdown = drawdown.max()
    return max_drawdown


def get_rt(wealth_series: pd.Series) -> int:
    """Longest run of consecutive periods the wealth series spends below a previous running maximum."""
    rolling_max = wealth_series.cummax()
    drawdown = (
        rolling_max - wealth_series
    ) / rolling_max  # percentage drawdown relative to rolling max
    if drawdown.max() == 0:
        return 0
    else:
        recovery_time = (
            drawdown[drawdown > 0].groupby((drawdown == 0).cumsum()).size().max()
        )
        return recovery_time


def get_real_and_div_rets(
    caps: pd.DataFrame, rets: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Compute real and dividend returns from CRSP caps and rets."""
    c = caps.to_numpy()
    r = rets.fillna(0.0).to_numpy()  # includes dividends

    cap_ratio = np.ones_like(c)  # 1 + div returns
    cap_ratio[1:] = c[1:] / c[:-1]

    div_rets = 1 + r - cap_ratio
    div_rets[div_rets < 0] = 0
    div_rets = np.nan_to_num(div_rets, nan=0.0)

    real_rets = r - div_rets
    return real_rets, div_rets


def calc_psi_minus_and_D_minus(
    rr: np.ndarray, dr: np.ndarray, psi: np.ndarray
) -> tuple[np.ndarray, float]:
    """Update portfolio holdings and accumulated dividends for one period."""
    psi_ = (1 + rr) * psi
    D_ = np.sum(psi * dr)
    return psi_, D_


def calc_D_hat(
    D_: float, W_: float, pi_target: np.ndarray, psi_: np.ndarray, tc: float
) -> float:
    """Calculate normalized available dividends after liquidating exited positions."""
    psi_sold = (pi_target == 0) * psi_
    return (D_ + np.sum(psi_sold) - tc * np.sum(np.abs(psi_sold))) / W_


def get_c_vector(pi: np.ndarray, pi_target: np.ndarray) -> np.ndarray:
    """Get the ratio of current to target weights."""
    c = np.divide(pi, pi_target, out=np.zeros_like(pi), where=pi_target != 0)
    return c


def _c_objective(
    c: float,
    c_vec: np.ndarray,
    D_hat: float,
    pi_target: np.ndarray,
    tc: float,
) -> float:
    d = pi_target * (c - c_vec)
    return d.sum() + tc * np.abs(d).sum() - D_hat


def _check_alignment(
    rets: pd.DataFrame, caps: pd.DataFrame, tgt_port: pd.DataFrame
) -> bool:
    """Check if indices and columns of the given DataFrames are aligned."""
    return (
        rets.index.equals(caps.index)
        and rets.index.equals(tgt_port.index)
        and rets.columns.equals(caps.columns)
        and rets.columns.equals(tgt_port.columns)
    )


def run_portfolio(
    W0: float,
    tc: float,
    caps: pd.DataFrame,
    rets: pd.DataFrame,
    tgt_port: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.Series, float, float, pd.Series]:
    """Backtest target portfolio weights with proportional transaction costs following the methodology in https://doi.org/10.1137/19M1282313.

    Args:
        W0: Initial portfolio wealth. This value is also reported for dates before
            the first available target allocation.
        tc: Proportional transaction-cost rate per dollar traded. For example,
            ``0.001`` represents 10 basis points.
        caps: Asset market capitalizations. Rows are dates and columns are assets;
            changes in capitalization are used to infer ex-dividend returns.
        rets: Total asset returns, including dividends, with the same index and
            columns as ``caps``.
        tgt_port: Target portfolio weights with the same index and columns as
            ``caps``. Each nonempty row must sum to one. Leading all-NaN rows are
            treated as a warm-up period, and other NaNs are treated as zero weight.
        start_date: Optional inclusive lower bound used to slice all three input
            frames before running the backtest.
        end_date: Optional inclusive upper bound used to slice all three input
            frames before running the backtest.

    Returns:
        A tuple containing the portfolio wealth series, average pre-trade gross
        exposure, average L1 weight turnover, and the transaction costs paid at
        each date. Both series use the sliced input index; transaction costs are
        expressed in the same currency units as ``W0``.
    """
    if start_date:
        caps = caps.loc[start_date:]
        rets = rets.loc[start_date:]
        tgt_port = tgt_port.loc[start_date:]
    if end_date:
        caps = caps.loc[:end_date]
        rets = rets.loc[:end_date]
        tgt_port = tgt_port.loc[:end_date]

    # check indices and columns are aligned
    if not _check_alignment(rets, caps, tgt_port):
        raise ValueError(
            "Indices and columns of caps, rets, and tgt_port must be aligned."
        )

    # ensure the root finding procedure to rebalance portfolio has a unique solution
    if np.max(tc * tgt_port.abs().sum(axis=1)) >= 1.0:
        raise ValueError(
            "Transaction costs too high, rebalancing procedure cannot be performed."
        )

    # ensure that the target portfolio rows sum to 1 for nonempty rows
    if not np.allclose(tgt_port.dropna(how="all").sum(axis=1), 1.0):
        raise ValueError("Nonempty rows of target portfolio must sum to 1.")

    # find first idx where tgt_port_array row is not all NaNs (trend portfolio is NaN while window saturates)
    start_idx = np.argmax(np.any(~np.isnan(tgt_port), axis=1))

    tgt_port_array = tgt_port.fillna(0.0).to_numpy()

    real_rets, div_rets = get_real_and_div_rets(caps, rets)  # returns (real & dividend)

    (T, N) = tgt_port_array.shape

    gross_exposure = 0.0
    turnover = 0.0
    transaction_costs = np.zeros(T)  # to track transaction costs over time
    port_value = np.zeros(T)  # to track portfolio value over time
    port_value[: start_idx + 1] = W0

    # Initialize
    W = W0  # wealth at time t
    W_ = W0
    psi = tgt_port_array[start_idx] * W0  # dollar positions
    D_ = 0.0
    pi = tgt_port_array[start_idx].copy()

    for t in range(start_idx + 1, T):
        # Get returns on day t
        rr = real_rets[t]
        dr = div_rets[t]

        # Determine dollar position before trading on day t
        psi_, period_D = calc_psi_minus_and_D_minus(rr, dr, psi)
        D_ += period_D

        # Portfolio value before trading on day t
        W_ = np.sum(psi_)

        # Inherited portfolio weights on day t
        pi = psi_ / W_ if W_ != 0 else np.zeros(N)

        gross_exposure += np.sum(np.abs(pi)) / (T - start_idx - 1)

        # Target portfolio for day t
        pi_target = tgt_port_array[t]

        # Normalized dividends
        D_hat = calc_D_hat(D_, W_, pi_target, psi_, tc)

        # Ratio vector
        c_vec = get_c_vector(pi, pi_target)

        # Solve for c
        res = root_scalar(
            f=_c_objective,
            args=(c_vec, D_hat, pi_target, tc),
            bracket=(-10, 10),
            method="brentq",
        )
        c = res.root

        # Update portfolio post trade on day t
        W = c * W_
        psi = pi_target * W

        # transaction costs on day t
        transaction_costs[t] = np.sum(np.abs(psi - psi_)) * tc

        # incremental average turnover on day t
        turnover += np.sum(np.abs(pi_target - pi)) / (T - start_idx - 1)

        D_ = 0.0
        port_value[t] = W

    port_value = pd.Series(port_value, index=caps.index)
    transaction_costs = pd.Series(transaction_costs, index=caps.index)
    return port_value, gross_exposure, turnover, transaction_costs


def sensitivity_analysis(
    signal_dict: dict[str, LbdaSignal],
    Lbda_1_grid: np.ndarray,
    Lbda_2_grid: np.ndarray,
    gamma_dict: dict[str, float],
    rets: pd.DataFrame,
    mkt_caps: pd.DataFrame,
    mkt_wgts: pd.DataFrame,
    eql_wgts: pd.DataFrame,
    div_qv_rate: pd.Series,
    rf_ret: pd.Series,
    W0: float,
    tc: float,
    target_active_risk: float,
    warmup_dates: tuple[str, str],
    backtest_dates: tuple[str, str],
    dt: float,
) -> dict[str, np.ndarray]:
    """Perform a sensitivity analysis over grids of lambda_1 and lambda_2 values, returning the information ratios for each combination."""

    Lbda_ir_arrays = {
        label: np.zeros((len(Lbda_1_grid), Lbda_2_grid.shape[0]))
        for label in ["mr", "trend"]
    }

    tot = len(Lbda_1_grid) * len(Lbda_2_grid)
    count = 0
    for i, Lbda_1 in enumerate(Lbda_1_grid):
        for j, Lbda_2 in enumerate(Lbda_2_grid):
            clear_output(wait=True)
            Lbda_1_dict = {"mr": Lbda_1[0], "trend": Lbda_1[1]}
            Lbda_2_dict = {"mr": Lbda_2[0], "trend": Lbda_2[1]}
            print(f"Running sensitivity analysis ({count + 1} / {tot})")
            print("==============================================")
            print(
                f"Lbda_1_dict={dict(pd.Series(Lbda_1_dict).round(4))},\nLbda_2_dict={dict(pd.Series(Lbda_2_dict).round(4))}"
            )

            sens_ports = LbdaPortfolio(
                rets=rets,
                mkt_caps=mkt_caps,
                mkt_wgts=mkt_wgts,
                eql_wgts=eql_wgts,
                signal_dict=signal_dict,
                div_qv_rate=div_qv_rate,
                rf_ret=rf_ret,
                W0=W0,
                tc=tc,
                target_active_risk=target_active_risk,
                warmup_dates=warmup_dates,
                backtest_dates=backtest_dates,
                dt=dt,
                Lbda_1_dict=Lbda_1_dict,
                Lbda_2_dict=Lbda_2_dict,
                gamma_dict=gamma_dict,
            )

            sens_ports.run_backtest(save_results=False)
            Lbda_ir_arrays["mr"][i, j] = sens_ports.summary_stats.loc[
                "information_ratio", "mr"
            ]
            Lbda_ir_arrays["trend"][i, j] = sens_ports.summary_stats.loc[
                "information_ratio", "trend"
            ]
            count += 1
    return Lbda_ir_arrays


class LbdaPortfolio:
    """Class to run backtests of optimal lambda-tilt portfolios with transaction costs, given a dictionary of signals and their corresponding parameters."""

    def __init__(
        self,
        rets: pd.DataFrame,
        mkt_caps: pd.DataFrame,
        mkt_wgts: pd.DataFrame,
        eql_wgts: pd.DataFrame,
        rf_ret: pd.Series,
        div_qv_rate: pd.Series,
        signal_dict: dict[str, LbdaSignal],
        dt: float,
        warmup_dates: tuple[str, str],
        backtest_dates: tuple[str, str],
        tc: float,
        target_active_risk: float,
        gamma_dict: dict[str, float] | None = None,
        Lbda_1_dict: dict[str, float] | None = None,
        Lbda_2_dict: dict[str, float] | None = None,
        W0: float = 1000,
    ) -> None:
        self.rets = rets
        self.mkt_caps = mkt_caps
        self.mkt_wgts = mkt_wgts
        self.eql_wgts = eql_wgts
        self.rf_ret = rf_ret
        self.signal_dict = signal_dict
        self.div_qv_rate = div_qv_rate
        self.dt = dt
        self.warmup_dates = warmup_dates
        self.backtest_dates = backtest_dates
        self.W0 = W0
        self.tc = tc
        self.target_active_risk = target_active_risk

        # cannot set Lbda dicts if transaction costs are zero
        if self.tc == 0.0 and (Lbda_1_dict is not None or Lbda_2_dict is not None):
            raise ValueError("Cannot set Lbda dicts if transaction costs are zero.")

        # per-signal attributes, keyed by signal name
        self.gamma_dict = dict(gamma_dict) if gamma_dict is not None else {}
        self.Lbda_1_dict = dict(Lbda_1_dict) if Lbda_1_dict is not None else {}
        self.Lbda_2_dict = dict(Lbda_2_dict) if Lbda_2_dict is not None else {}
        self.lbda_0 = {}
        self.lbda = {}
        self.lbda_bar = {}
        self.lbda_0_port = {}
        self.lbda_port = {}
        self.lbda_bar_port = {}

        self.gross_wealth_df = None
        self.net_wealth_df = None
        self.gross_log_V_df = None
        self.net_log_V_df = None
        self.summary_stats = None

        self.active_costs_df = None
        self.turnover_df = None
        self.transaction_costs_df = None

        # caches for signal-independent computations
        self._mkt_wealth_cache = {}
        self._ew_active_costs = None
        self._ew_tc_penalty = None

    def _get_lbda_port(self, lbda: pd.Series) -> pd.DataFrame:
        """Construct lambda-tilt portfolio pi^lambda = lbda * ew + (1 - lbda) * mkt."""
        lbda_port = self.eql_wgts.multiply(lbda, axis=0) + self.mkt_wgts.multiply(
            1 - lbda, axis=0
        )
        return lbda_port

    def _set_gamma(self, name: str) -> float:
        """Calibrate gamma so the frictionless lbda_0 portfolio achieves the target active risk against the market over the warmup window."""
        ws, we = self.warmup_dates

        lbda_0_init = self.signal_dict[name].signal_0.loc[ws:we]
        gamma_init = lbda_0_init.std()

        def calibrate_gamma(gamma: float) -> float:
            lbda_0_warmup = self.signal_dict[name].signal_0.loc[ws:we] / (1 + gamma)
            lbda_port_warmup = self._get_lbda_port(lbda_0_warmup).loc[ws:we]
            lbda_log_ret_warmup = np.log1p(
                (self.rets * lbda_port_warmup.shift(1))
                .sum(axis=1, min_count=1)
                .loc[ws:we]
            )
            mkt_log_ret_warmup = np.log1p(
                (self.rets * self.mkt_wgts.shift(1)).sum(axis=1, min_count=1).loc[ws:we]
            )
            active_ret_warmup = lbda_log_ret_warmup - mkt_log_ret_warmup
            return active_ret_warmup.std() / np.sqrt(self.dt) - self.target_active_risk

        res = root_scalar(calibrate_gamma, x0=gamma_init, method="secant")

        gamma = res.root
        print(f"[{name}] Calibrated gamma: {gamma:.2f}")
        return gamma

    def _get_mkt_wealth(self, tc: float, start_date: str, end_date: str) -> pd.Series:
        """Run (and cache) the market portfolio backtest, which is signal-independent."""
        key = (tc, start_date, end_date)
        if key not in self._mkt_wealth_cache:
            wealth, _, _, _ = run_portfolio(
                W0=self.W0,
                tc=tc,
                caps=self.mkt_caps,
                rets=self.rets,
                tgt_port=self.mkt_wgts,
                start_date=start_date,
                end_date=end_date,
            )
            self._mkt_wealth_cache[key] = wealth
        return self._mkt_wealth_cache[key]

    def _get_active_costs(
        self, start_date: str, end_date: str, port_wgts: pd.DataFrame
    ) -> pd.Series:
        """Cumulative active trading costs of a portfolio, as the gap between its gross and net log relative value against the market."""
        if self.tc == 0:
            return pd.Series(0.0, index=port_wgts.index)[start_date:end_date]

        tc_list = [0, self.tc]
        log_V_dict = {}
        for tc in tc_list:
            mkt_wealth = self._get_mkt_wealth(tc, start_date, end_date)
            port_wealth, _, _, _ = run_portfolio(
                W0=self.W0,
                tc=tc,
                caps=self.mkt_caps,
                rets=self.rets,
                tgt_port=port_wgts,
                start_date=start_date,
                end_date=end_date,
            )
            log_V_dict[tc] = np.log(port_wealth / mkt_wealth)

        active_costs = log_V_dict[0] - log_V_dict[self.tc]
        return active_costs

    def _get_ew_active_costs(self) -> pd.Series:
        """Compute (and cache) the equal-weight active costs over [warmup start, backtest end], which are signal-independent."""
        ws, _ = self.warmup_dates
        _, be = self.backtest_dates
        if self._ew_active_costs is None:
            self._ew_active_costs = self._get_active_costs(ws, be, self.eql_wgts)
        return self._ew_active_costs

    def _get_ew_tc_penalty(self) -> pd.Series:
        """Compute (and cache) the equal-weight transaction cost penalty over [warmup start, backtest end], which is signal-independent."""
        ws, _ = self.warmup_dates
        _, be = self.backtest_dates
        if self._ew_tc_penalty is None:
            div_qv_rate = self.div_qv_rate[ws:be]
            lbda_ew = pd.Series(np.ones(len(div_qv_rate)), index=div_qv_rate.index)
            self._ew_tc_penalty = (
                get_TC_penalty_rate(
                    lbda=lbda_ew,
                    diversity_qv_rate=div_qv_rate,
                    dt=self.dt,
                    Lbda_1=self.Lbda_1_dict["ew"],
                    Lbda_2=0,
                ).cumsum()
                * self.dt
            )
        return self._ew_tc_penalty

    def _get_Lbda_1(self, name: str) -> np.float64:
        """Compute Lbda_1 for the given portfolio by fitting the terminal in-sample transaction cost penalty for the equal-weighted portfolio to the terminal active costs."""
        if self.tc == 0.0:
            return 0.0

        dt = self.dt
        ws, we = self.warmup_dates
        if name == "ew":
            div_qvr_IS = self.div_qv_rate[ws:we]
        else:
            div_qvr_IS = self.signal_dict[name].features["var_phi"][ws:we]
        terminal_div_qv = div_qvr_IS.multiply(dt).cumsum().iloc[-1]

        ew_active_tc_full = self._get_ew_active_costs()
        ew_terminal_ac = ew_active_tc_full.loc[:we].iloc[-1]

        Lbda_1 = 2 * ew_terminal_ac / terminal_div_qv
        print(f"Lbda_1: {Lbda_1:.3f}.")
        return Lbda_1

    def _calibrate_Lbda_2(self, name: str) -> tuple[float, pd.Series, pd.Series]:
        """Calibrate Lbda_2 by minimizing the difference between the terminal active costs and the terminal TC penalty of the lbda portfolio over the warmup period."""
        if self.tc == 0.0:
            self.Lbda_2_dict[name] = 0.0
            return (
                0.0,
                pd.Series(0.0, index=self.eql_wgts.index),
                pd.Series(0.0, index=self.eql_wgts.index),
            )
        if self.Lbda_1_dict.get(name) is None:
            raise ValueError("Lbda_1 must be calibrated before calibrating Lbda_2.")
        ws, we = self.warmup_dates
        _, be = self.backtest_dates
        div_qv_rate = self.signal_dict[name].features["var_phi"][ws:be]
        gamma = self.gamma_dict[name]
        Lbda_1 = self.Lbda_1_dict[name]

        def opt_func_Lbda_2(Lbda_2: np.ndarray) -> float:
            Lbda_2 = float(
                np.ravel(Lbda_2)[0]
            )  # minimize passes a shape-(1,) array, not a scalar
            lbda = self.signal_dict[name].get_lbda(
                gamma=gamma, Lbda_1=Lbda_1, Lbda_2=Lbda_2
            )
            lbda_port = self._get_lbda_port(lbda)

            lbda_active_costs = self._get_active_costs(ws, we, lbda_port)

            lbda_tc_penalty_full = (
                get_TC_penalty_rate(
                    lbda=lbda,
                    diversity_qv_rate=div_qv_rate,
                    dt=self.dt,
                    Lbda_1=self.Lbda_1_dict[name],
                    Lbda_2=Lbda_2,
                ).cumsum()
                * self.dt
            )
            lbda_tc_penalty = lbda_tc_penalty_full[ws:we]
            loss = (lbda_active_costs - lbda_tc_penalty).iloc[-1] ** 2
            return loss

        # x0 = max(1e-4, 0.1 * Lbda_2_ub)
        x0 = 0.1
        lb, ub = 1e-4, 5
        res = minimize(opt_func_Lbda_2, x0=x0, bounds=[(lb, ub)], method="nelder-mead")

        if res.success:
            Lbda_2_opt = res.x[0]
            print(f"[{name}] Calibrated Lbda_2: {Lbda_2_opt:.3f}")
            self.Lbda_2_dict[name] = Lbda_2_opt
            lbda = self.signal_dict[name].get_lbda(
                gamma=gamma, Lbda_1=Lbda_1, Lbda_2=Lbda_2_opt
            )
            lbda_port = self._get_lbda_port(lbda)
            lbda_active_costs_full = self._get_active_costs(ws, be, lbda_port)
            lbda_tc_penalty_full = (
                get_TC_penalty_rate(
                    lbda=lbda,
                    diversity_qv_rate=div_qv_rate,
                    dt=self.dt,
                    Lbda_1=self.Lbda_1_dict[name],
                    Lbda_2=Lbda_2_opt,
                ).cumsum()
                * self.dt
            )

            return Lbda_2_opt, lbda_active_costs_full, lbda_tc_penalty_full
        else:
            raise ValueError("Optimization failed for Lbda_2 calibration.")

    def _get_summary_stats(
        self, wealth_series: pd.Series, gross_exposure: float, turnover: float
    ) -> dict[str, float]:
        """Calculate summary statistics for given wealth series."""
        summary_stats = {}
        rf_ret_period = self.rf_ret[self.backtest_dates[0] : self.backtest_dates[1]]
        # port_log_ret = np.log(wealth_series).diff().dropna()
        port_ret = wealth_series.pct_change().dropna()
        summary_stats["total_ret"] = (1 + port_ret).prod() - 1
        summary_stats["mean_ret"] = port_ret.mean() / self.dt
        summary_stats["volatility"] = port_ret.std() / np.sqrt(self.dt)
        excess_ret = port_ret - rf_ret_period
        summary_stats["sharpe_ratio"] = (
            excess_ret.mean() / excess_ret.std() * np.sqrt(1 / self.dt)
        )
        summary_stats["gross_exposure"] = gross_exposure
        summary_stats["turnover"] = turnover
        summary_stats["max_drawdown"] = get_mdd(wealth_series)
        summary_stats["recovery_time"] = get_rt(wealth_series)
        return summary_stats

    def build(self) -> tuple[dict, dict]:
        """Build the lbda portfolios, one per signal in signal_dict."""
        print("Building lbda portfolios...")
        ws, _ = self.warmup_dates
        _, be = self.backtest_dates
        active_costs_dict = {}
        tc_penalty_dict = {}

        self.Lbda_1_dict["ew"] = self._get_Lbda_1(name="ew")

        active_costs_dict["ew"] = self._get_ew_active_costs()
        tc_penalty_dict["ew"] = self._get_ew_tc_penalty()

        for name in self.signal_dict:
            sig = self.signal_dict[name]
            print(f"--- Signal: {name} ---")
            gamma_n = self.gamma_dict.get(name)
            if gamma_n is None:
                gamma_n = self._set_gamma(name)
                self.gamma_dict[name] = gamma_n
            self.lbda_0[name] = sig.signal_0 / (1 + gamma_n)
            self.lbda_0_port[name] = self._get_lbda_port(self.lbda_0[name])

            # set Lbda_1 for portfolio
            Lbda_1_n = self.Lbda_1_dict.get(name)
            if Lbda_1_n is None:
                self.Lbda_1_dict[name] = self._get_Lbda_1(name)
            else:
                self.Lbda_1_dict[name] = Lbda_1_n

            # set Lbda_2 for portfolio and get active costs and transaction cost penalty
            Lbda_2_n = self.Lbda_2_dict.get(name)
            if Lbda_2_n is None:
                (
                    self.Lbda_2_dict[name],
                    active_costs_dict[name],
                    tc_penalty_dict[name],
                ) = self._calibrate_Lbda_2(name)
            else:
                self.Lbda_2_dict[name] = Lbda_2_n
                lbda = sig.get_lbda(
                    gamma=self.gamma_dict[name],
                    Lbda_1=self.Lbda_1_dict[name],
                    Lbda_2=Lbda_2_n,
                )
                lbda_port = self._get_lbda_port(lbda)
                active_costs_dict[name] = self._get_active_costs(ws, be, lbda_port)
                tc_penalty_dict[name] = (
                    get_TC_penalty_rate(
                        lbda=lbda,
                        diversity_qv_rate=sig.features["var_phi"][ws:be],
                        dt=self.dt,
                        Lbda_1=self.Lbda_1_dict[name],
                        Lbda_2=Lbda_2_n,
                    ).cumsum()
                    * self.dt
                )

            self.lbda[name] = sig.get_lbda(
                gamma=self.gamma_dict[name],
                Lbda_1=self.Lbda_1_dict[name],
                Lbda_2=self.Lbda_2_dict[name],
            )
            self.lbda_port[name] = self._get_lbda_port(self.lbda[name])

            self.lbda_bar[name] = self.lbda[name].clip(lower=0, upper=1)
            self.lbda_bar_port[name] = self._get_lbda_port(self.lbda_bar[name])
        print("Lbda portfolios built.")
        return active_costs_dict, tc_penalty_dict

    def run_backtest(self, save_results: bool = True) -> None:
        """Run backtest for the lbda portfolios (one per signal) and benchmark portfolios, and compute summary statistics."""
        print("Starting backtest...")

        bs, be = self.backtest_dates
        dt = self.dt
        tc_list = [0, self.tc]
        wealth_dict = {}
        summary_stats_dict = {}
        transaction_costs_dict = {}

        active_costs_dict, tc_penalty_dict = self.build()

        ports = {"mkt": self.mkt_wgts, "ew": self.eql_wgts}
        for name in self.signal_dict:
            ports[f"{name}_0"] = self.lbda_0_port[name]
            if self.tc != 0:
                ports[name] = self.lbda_port[name]
                ports[f"{name}_bar"] = self.lbda_bar_port[name]

        for tc in tc_list:
            wealth_dict[tc] = {}
            summary_stats_dict[tc] = {}
            for label, port in ports.items():
                wealth, gross_exposure, turnover, transaction_costs = run_portfolio(
                    W0=self.W0,
                    tc=tc,
                    caps=self.mkt_caps,
                    rets=self.rets,
                    tgt_port=port,
                    start_date=bs,
                    end_date=be,
                )
                wealth_dict[tc][label] = wealth
                summary_stats_dict[tc][label] = self._get_summary_stats(
                    wealth, gross_exposure, turnover
                )

                summary_stats_dict[tc][label]["gross_sharpe_ratio"] = (
                    summary_stats_dict[0][label]["sharpe_ratio"]
                )
                transaction_costs_dict[label] = transaction_costs

                if label != "mkt":
                    summary_stats_dict[tc][label]["information_ratio"] = get_ir_pct(
                        wealth_dict[tc][label], wealth_dict[tc]["mkt"], dt=dt
                    )

        gross_wealth_df = pd.DataFrame(wealth_dict[0])
        self.gross_wealth_df = gross_wealth_df
        net_wealth_df = pd.DataFrame(wealth_dict[self.tc])
        self.net_wealth_df = net_wealth_df

        gross_log_V_df = gross_wealth_df.apply(
            lambda x: np.log(x) - np.log(gross_wealth_df["mkt"])
        )
        self.gross_log_V_df = gross_log_V_df
        net_log_V_df = net_wealth_df.apply(
            lambda x: np.log(x) - np.log(net_wealth_df["mkt"])
        )
        self.net_log_V_df = net_log_V_df

        summary_stats_df = pd.DataFrame(summary_stats_dict[self.tc])
        self.summary_stats = summary_stats_df
        print("Backtest completed.")
        active_costs_df = pd.DataFrame(active_costs_dict)
        self.active_costs_df = active_costs_df
        tc_penalty_df = pd.DataFrame(tc_penalty_dict)
        self.tc_penalty_df = tc_penalty_df
        transaction_costs_df = pd.DataFrame(transaction_costs_dict)
        self.transaction_costs_df = transaction_costs_df

        if save_results:
            os.makedirs(RESULTS, exist_ok=True)
            save_path = RESULTS / "summary_stats.csv"
            summary_stats_df.to_csv(save_path)
            print(f"Saved summary results to {relative_path(save_path)}")
