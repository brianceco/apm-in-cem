"""Simulation utilities to implement the mean-reverting SDD model. Includes calibrating model to real data, simulating paths, and computing optimal lambda-tilt processes for simulated paths."""

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize

from config import DT_DAILY
from paths import RESULTS, relative_path
from utils import (
    get_ci,
    get_ou_params,
    get_target,
    get_TC_penalty_rate,
    log_likelihood_with_grad,
    target_kernels,
)


def autocorr_sim(
    stationary_var: float, kappa_delta: float, delta_t: float, max_lags: int
) -> np.ndarray:
    """Theoretical autocorrelation function (ACF) for the stationary mean-reverting SDD model."""
    autocorr = np.zeros(max_lags)
    denom = np.exp(stationary_var) - 1

    def num(lag: int) -> float:
        return np.exp(stationary_var * np.exp(-kappa_delta * lag * delta_t)) - 1

    for lag in range(max_lags):
        autocorr[lag] = num(lag) / denom
    return autocorr


def get_log_V_lbda(
    nu_phi: float,
    dt: float,
    phi: pd.DataFrame,
    delta: pd.DataFrame,
    lbda: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the log relative value of the tilt process lambda(t) for simulated diversity and dispersion paths."""

    log_V_lbda = np.zeros(phi.shape)

    phi_diff_vals = phi.diff().values
    lbda_vals = lbda.values
    delta_vals = delta.values
    active_var_vals = nu_phi**2 * delta_vals
    for n in range(1, log_V_lbda.shape[0]):
        log_V_lbda[n] = (
            log_V_lbda[n - 1]
            + lbda_vals[n - 1] * (phi_diff_vals[n] + 0.5 * delta_vals[n - 1] * dt)
            + 0.5
            * lbda_vals[n - 1]
            * (1 - lbda_vals[n - 1])
            * active_var_vals[n - 1]
            * dt
        )

    log_V_lbda = pd.DataFrame(log_V_lbda, index=phi.index, columns=phi.columns)
    return log_V_lbda


class MeanRevertingSDDModel:
    """Mean-reverting stochastic diversity dispersion model."""

    def __init__(
        self, dt: float, config: dict[str, float] | None = None, seed: int | None = None
    ) -> None:
        self.dt = dt
        self.seed = seed
        self.config = config if config else {}

    def save_model(self, name: str) -> None:
        """Write the model's dt, seed and fitted config to a JSON file."""
        os.makedirs(RESULTS, exist_ok=True)
        save_path = RESULTS / name
        with open(save_path, "w") as f:
            json.dump({"dt": self.dt, "seed": self.seed, "config": self.config}, f)
        print(f"Saved model parameters to {relative_path(save_path)}")

    @classmethod
    def load_model(cls, name: str) -> "MeanRevertingSDDModel":
        """Reconstruct a model from a JSON file previously written by save_model."""
        with open(RESULTS / name, "r") as f:
            data = json.load(f)
        model = cls(dt=data["dt"], seed=data["seed"], config=data["config"])
        return model

    def _get_init_params(
        self, diversity: pd.Series, dispersion: pd.Series, diversity_qv: pd.Series
    ) -> dict[str, float]:
        """Estimate starting values for the MLE by fitting OU parameters to diversity and to log dispersion separately."""
        dt = self.dt
        init_params = {}

        kappa_phi_0, phi_bar_0, _ = get_ou_params(diversity, dt)
        init_params["kappa_phi"] = kappa_phi_0
        init_params["phi_bar"] = phi_bar_0
        init_params["nu_phi"] = np.sqrt(diversity_qv.mean() / dispersion.mean())

        kappa_delta_0, log_delta_bar_0, nu_delta_0 = get_ou_params(
            np.log(dispersion), dt
        )
        init_params["kappa_delta"] = kappa_delta_0
        init_params["delta_bar"] = np.exp(log_delta_bar_0)
        init_params["nu_delta"] = nu_delta_0

        init_params["rho"] = diversity.diff().corr(np.log(dispersion).diff())

        print(
            f"Initial parameter estimates:\n{'\n'.join([f'{k}:  {val:.3f}' for k, val in init_params.items()])}"
        )

        return init_params

    def _get_mle(
        self,
        div_array: np.ndarray,
        disp_array: np.ndarray,
        init_params: dict[str, float],
    ) -> dict[str, float]:
        """Maximize the joint log likelihood by SLSQP, with bounds set as multiples of the initial parameter estimates."""
        init_params_list = list(init_params.values())
        mult_bounds = [0.1, 10]
        bounds = [
            (lb * param, ub * param)
            for param, (lb, ub) in zip(
                init_params.values(), [mult_bounds] * (len(init_params) - 1)
            )
        ]
        bounds.append((-1, 1))  # bound for rho
        bounds[1] = (
            bounds[1][1],
            bounds[1][0],
        )  # reorder bounds for phi_bar because higher value occurs first

        fun = lambda params: log_likelihood_with_grad(
            params=params, diversity=div_array, dispersion=disp_array, dt=self.dt
        )

        mle_results = minimize(
            fun=fun, x0=init_params_list, jac=True, bounds=bounds, method="SLSQP"
        )
        print("-----------------------")
        if mle_results.success:
            print("MLE successfully converged.")
            mle_parameters = dict(zip(init_params.keys(), mle_results.x))
        else:
            raise RuntimeError("MLE did not converge.")

        print("-----------------------")
        print(
            f"MLE parameters:\n{'\n'.join([f'{k}: {val:.3f}' for k, val in mle_parameters.items()])}"
        )
        return mle_parameters

    def fit(self, IS_monthly_data: pd.DataFrame, alpha: float | None = 0.05) -> dict:
        """Fit the model by MLE and store the estimates in config, returning them with confidence intervals at level alpha."""
        diversity = IS_monthly_data["diversity"]
        dispersion = IS_monthly_data["dispersion"]
        diversity_qv = IS_monthly_data["diversity_qv"]

        init_params = self._get_init_params(diversity, dispersion, diversity_qv)

        div_array = diversity.values
        disp_array = dispersion.values

        mle_parameters = self._get_mle(div_array, disp_array, init_params)

        self.config.update(mle_parameters)

        if alpha:
            ci = get_ci(mle_parameters, div_array, disp_array, alpha=alpha, dt=self.dt)
            print("-----------------------")
            print(
                f"Confidence intervals:\n{'\n'.join([f'{k}:  ({val[0]:.3f}, {val[1]:.3f})' for k, val in ci.items()])}"
            )
            return {"mle_parameters": mle_parameters, "ci": ci}
        else:
            return mle_parameters

    def _log_delta_drift(self, log_delta: np.ndarray) -> np.ndarray:
        """Drift of log dispersion, mean-reverting to log(delta_bar) at rate kappa_delta."""
        kappa = self.config["kappa_delta"]
        mu = self.config["delta_bar"]
        return kappa * (np.log(mu) - log_delta)

    def _log_delta_diffusion(self) -> float:
        """Diffusion coefficient of log dispersion, constant at nu_delta."""
        nu_delta = self.config["nu_delta"]
        return nu_delta

    def _phi_drift(self, phi: np.ndarray) -> np.ndarray:
        """Drift of diversity, mean-reverting to phi_bar at rate kappa_phi."""
        kappa = self.config["kappa_phi"]
        mu = self.config["phi_bar"]
        return kappa * (mu - phi)

    def _phi_diffusion(self, log_delta: np.ndarray) -> np.ndarray:
        """Diffusion coefficient of diversity, proportional to the square root of dispersion."""
        nu_delta = self.config["nu_phi"]
        return nu_delta * np.exp(log_delta) ** 0.5

    def sample(
        self,
        N: int,
        n_samples: int = 1,
        phi_0: float | None = None,
        delta_0: float | None = None,
        delta_min: float = 0,
        seed: int | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate joint sample paths for diversity and dispersion processes. Hard coded to simulate paths at daily frequency and then resample according to dt (default = 1/12, i.e., monthly)."""

        # N represents number of increments of size dt
        days_per_dt = int(self.dt / DT_DAILY)

        N_daily = N * days_per_dt

        if not self.config:
            raise ValueError(
                "Model parameters not set. Please fit the model or set a config before sampling."
            )

        if seed is None:
            seed = self.seed
        rng = np.random.default_rng(seed)

        div_noise = rng.normal(0, 1, (N_daily, n_samples))
        disp_noise = rng.normal(0, 1, (N_daily, n_samples))

        W1 = div_noise
        W2 = self.config["rho"] * W1 + np.sqrt(1 - self.config["rho"] ** 2) * disp_noise

        # Simulate diversity and dispersion processes
        log_delta = np.zeros((N_daily, n_samples))
        phi = np.zeros((N_daily, n_samples))

        if phi_0 is None:
            phi_0 = self.config["phi_bar"]
        if delta_0 is None:
            delta_0 = self.config["delta_bar"]
        phi[0] = phi_0
        log_delta[0] = np.log(delta_0)

        log_delta_min = np.log(delta_min) if delta_min > 0 else -np.inf

        for t in range(1, N_daily):
            dW1 = np.sqrt(DT_DAILY) * W1[t]
            b_phi_t = self._phi_drift(phi[t - 1])
            sigma_phi_t = self._phi_diffusion(log_delta[t - 1])
            phi[t] = phi[t - 1] + b_phi_t * DT_DAILY + sigma_phi_t * dW1

            dW2 = np.sqrt(DT_DAILY) * W2[t]
            log_delta_drift = self._log_delta_drift(log_delta[t - 1])
            log_delta_diffusion = self._log_delta_diffusion()
            log_delta[t] = (
                log_delta[t - 1]
                + log_delta_drift * DT_DAILY
                + log_delta_diffusion * dW2
            )

        log_delta = np.maximum(log_delta, log_delta_min)

        # dt resampling
        phi = phi[days_per_dt - 1 :: days_per_dt]
        delta = np.exp(log_delta).reshape(-1, days_per_dt, n_samples)
        delta = delta.sum(axis=1) / days_per_dt

        phi = pd.DataFrame(phi)
        delta = pd.DataFrame(delta)

        return phi, delta

    def get_lbda_0(
        self,
        phi: pd.Series | pd.DataFrame,
        delta: pd.Series | pd.DataFrame,
        gamma: float = 0,
    ) -> pd.Series | pd.DataFrame:
        """Compute the frictionless optimal tilt process lambda_0(t) for simulated diversity and dispersion paths under the mean-reverting model."""
        kappa = self.config["kappa_phi"]
        mu = self.config["phi_bar"]
        nu_phi = self.config["nu_phi"]
        active_drift = kappa * (mu - phi) + 0.5 * (1 + nu_phi**2) * delta
        active_var = nu_phi**2 * delta
        lbda_0 = active_drift / active_var
        lbda_0 = lbda_0 / (1 + gamma)
        return lbda_0

    def _get_lbda(
        self,
        gamma: float,
        lbda_0_init: np.ndarray | float,
        Lbda_1: float,
        Lbda_2: float,
        phi_values: np.ndarray,
        delta_values: np.ndarray,
    ) -> np.ndarray:
        """Get the smoothed optimal switching process lambda(t) for simulated diversity and dispersion paths under the mean-reverting model."""
        kappa_phi = self.config["kappa_phi"]
        kappa_delta = self.config["kappa_delta"]
        nu_delta = self.config["nu_delta"]
        dt = self.dt

        # make 2-dimensional
        if phi_values.ndim == 1:
            phi_values = phi_values[:, None]
            delta_values = delta_values[:, None]

        N, M = phi_values.shape

        C = np.sqrt((1 + gamma + Lbda_1) / Lbda_2)
        T_minus_t = dt * (N - np.arange(N + 1))  # T - t on the grid, (N+1,)
        G = np.cosh(C * T_minus_t)  # G(t), (N+1,)

        if np.any(G == np.inf):
            raise ValueError(
                "G(t) diverges to infinity. You must adjust the parameters gamma, Lbda_1, Lbda_2, or T."
            )

        kernels = target_kernels(kappa_phi, kappa_delta, nu_delta, dt, N)

        # F(s) = int_s^T G(u) E_s[lbda_0(u)] du, by trapezoidal integration in u
        F = np.zeros((N, M))
        for t in range(N):
            P = N + 1 - t  # grid points u remaining in [s, T]
            phi_t = phi_values[t]
            log_delta_t = np.log(delta_values[t])

            target = get_target(
                params=self.config,
                kernels=kernels,
                gamma=gamma,
                phi_t=phi_t,
                log_delta_t=log_delta_t,
                P=P,
            )

            weights = np.full(P, dt)
            weights[[0, -1]] = 0.5 * dt
            weights *= G[t:]

            F[t] = weights @ target

        # lbda(t) = G(t) [lbda_0 / G(0) + int_0^t (1 + gamma) F(s) / (Lbda_2 G(s)^2) ds],
        G_s = G[:N, None]  # G(s) on the grid, (N, 1)
        integrand = (1 + gamma) * F / G_s / (Lbda_2 * G_s)  # (N, M) prevent overflow
        s_integral = np.zeros((N, M))
        s_integral[1:] = np.cumsum(0.5 * dt * (integrand[:-1] + integrand[1:]), axis=0)

        init = np.broadcast_to(np.asarray(lbda_0_init, dtype=float).ravel(), (M,))
        lbda = G_s * (init / G[0] + s_integral)
        return lbda

    def _set_gamma(
        self,
        target_active_risk: float,
        phi_samples: pd.Series | pd.DataFrame,
        delta_samples: pd.Series | pd.DataFrame,
    ) -> float:
        """Calibrate gamma so log relative value of optimal frictionless tilt process achieves the target active risk for simulated diversity and dispersion paths under the mean-reverting model."""
        nu_phi = self.config["nu_phi"]
        dt = self.dt

        def calibrate_gamma(gamma: float) -> float:
            lbda_0 = self.get_lbda_0(phi_samples, delta_samples, gamma)
            log_V_lbda = get_log_V_lbda(nu_phi, dt, phi_samples, delta_samples, lbda_0)
            mean_active_risk = (
                log_V_lbda.diff().std() / np.sqrt(dt)
            ).mean()  # compute the average annualized active risk across time and sample paths
            return target_active_risk - mean_active_risk

        res = brentq(calibrate_gamma, a=1, b=10e3)
        return res

    def _get_performance(
        self,
        lbda: pd.Series | pd.DataFrame,
        diversity_qv_rate: pd.Series | pd.DataFrame,
        phi_samples: pd.Series | pd.DataFrame,
        delta_samples: pd.Series | pd.DataFrame,
        Lbda_1: float,
        Lbda_2: float,
    ) -> tuple[
        pd.Series | pd.DataFrame, pd.Series | pd.DataFrame, pd.Series | pd.DataFrame
    ]:
        dt = self.dt
        nu_phi = self.config["nu_phi"]
        sim_log_V = get_log_V_lbda(
            nu_phi=nu_phi, dt=dt, phi=phi_samples, delta=delta_samples, lbda=lbda
        )

        sim_tc_rate = get_TC_penalty_rate(lbda, diversity_qv_rate, dt, Lbda_1, Lbda_2)

        sim_net_log_V = sim_log_V - sim_tc_rate.multiply(dt).cumsum()

        return sim_log_V, sim_tc_rate, sim_net_log_V

    def backtest(
        self,
        target_active_risk: float,
        Lbda_1: float,
        Lbda_2: float,
        samples: list[pd.DataFrame] | list[pd.Series] | None = None,
        **kwargs,
    ) -> tuple[dict, dict, dict, dict, float]:
        """Backtest the frictionless and optimal tilt processes on simulated paths, returning their lambda, gross and net log relative value, and trading cost rate."""
        if not Lbda_2:
            raise ValueError("Lbda_2 must be provided.")

        if not self.config:
            raise ValueError(
                "Model parameters not set. Please fit the model or set a config before backtesting."
            )

        lbda_dict = {}
        gross_log_V_dict = {}
        net_log_V_dict = {}
        tc_rate_dict = {}

        nu_phi = self.config["nu_phi"]

        if samples:
            phi_samples, delta_samples = samples
        else:
            phi_samples, delta_samples = self.sample(**kwargs)

        sim_lbda_ew = pd.DataFrame(
            np.ones_like(phi_samples),
            index=phi_samples.index,
            columns=phi_samples.columns,
        )

        lbda_dict["ew"] = sim_lbda_ew

        diversity_qv_rate = nu_phi**2 * delta_samples

        gamma = self._set_gamma(target_active_risk, phi_samples, delta_samples)

        print(f"Calibrated gamma: {gamma:.3f}")

        lbda_0 = self.get_lbda_0(phi_samples, delta_samples, gamma)
        lbda_dict["lbda_0"] = lbda_0

        lbda = self._get_lbda(
            gamma=gamma,
            lbda_0_init=lbda_0.iloc[0],
            Lbda_1=Lbda_1,
            Lbda_2=Lbda_2,
            phi_values=phi_samples.values,
            delta_values=delta_samples.values,
        )

        lbda = pd.DataFrame(lbda, index=phi_samples.index, columns=phi_samples.columns)

        lbda_dict["lbda"] = lbda

        for key, l in lbda_dict.items():
            gross_log_V_dict[key], tc_rate_dict[key], net_log_V_dict[key] = (
                self._get_performance(
                    lbda=l,
                    diversity_qv_rate=diversity_qv_rate,
                    phi_samples=phi_samples,
                    delta_samples=delta_samples,
                    Lbda_1=Lbda_1,
                    Lbda_2=Lbda_2,
                )
            )

        return lbda_dict, gross_log_V_dict, net_log_V_dict, tc_rate_dict, gamma
