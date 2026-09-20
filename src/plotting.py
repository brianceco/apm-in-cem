"""Plotting utilities, mainly to set matplotlib rcParams which produce nice-looking plots designed for LaTeX documents"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, Normalize
from scipy import stats

import utils
from backtest import LbdaPortfolio
from config import DT_DAILY, DT_MONTHLY
from paths import FIGURES, STYLE, relative_path
from simulate import MeanRevertingSDDModel as SDDModel
from simulate import autocorr_sim

DIVERSITY_COLOR = "green"
DISPERSION_COLOR = "indigo"


def adjust_xaxis(axs: Axes | np.ndarray) -> None:
    """
    Adjust x-axis to start and end at boundaries of the data, spanning every line on the axes
    """
    for ax in np.atleast_1d(axs).ravel():
        # skip axhline/axvline, whose xdata is in axes rather than data coordinates
        lines = [line for line in ax.lines if line.get_transform() is ax.transData]
        if not lines:
            continue
        data = [np.asarray(line.get_xdata()) for line in lines]
        ax.set_xlim(min(d.min() for d in data), max(d.max() for d in data))


def use_paper_style(usetex: bool = True) -> list[float]:
    """Apply the paper's matplotlib style and return the default figure size, disabling LaTeX rendering when usetex is False."""
    plt.style.use(STYLE)
    if not usetex:
        plt.rcParams["text.usetex"] = False
    return plt.rcParams["figure.figsize"]


def compress_cmap(
    cmap: Colormap, exponent: float = 0.5, n: int = 256
) -> ListedColormap:
    """
    Compress colormap colors.
    """
    t = np.linspace(0, 1, n)
    s = 2 * t - 1
    return ListedColormap(cmap(0.5 * (1 + np.sign(s) * np.abs(s) ** exponent)))


def plot_dispersion(signals: pd.DataFrame, dspx: pd.Series, save_fig: bool = True, **kwargs) -> None:
    """Plot monthly equal-weighted realized dispersion, with a figure-in-figure comparing lagged implied dispersion (DSPX) against rolling cap-weighted realized dispersion."""
    _, ax = plt.subplots(**kwargs)

    rd_eql_monthly = signals["rd_eql"].resample("ME").sum() * (DT_DAILY / DT_MONTHLY)

    rd_eql_monthly.plot(ax=ax, color="b", x_compat=True)
    ax.set_xlabel("Year")
    ax.set_ylabel("Realized Dispersion $(\\mathrm{RD}_{\\mathrm{ew}})$")
    adjust_xaxis(ax)

    rolling_mkt_rd_monthly = signals["rd_mkt"].multiply(DT_DAILY).rolling(21).sum()
    adjusted_rolling_mkt_rd_monthly = rolling_mkt_rd_monthly / DT_MONTHLY

    dspx_lag = dspx.divide(100).pow(2).shift(21).dropna()

    start_date = max(dspx_lag.index[0], adjusted_rolling_mkt_rd_monthly.index[0])
    end_date = min(dspx_lag.index[-1], adjusted_rolling_mkt_rd_monthly.index[-1])

    axins = ax.inset_axes([0.1, 0.5, 0.47, 0.46])
    axins.plot(
        dspx_lag.loc[start_date:end_date],
        color="grey",
        label="$\\mathrm{ID}_{\\mu}$",
        linewidth=1,
    )
    axins.plot(
        adjusted_rolling_mkt_rd_monthly.loc[start_date:end_date],
        color="r",
        label="$\\mathrm{RD}_{\\mu}$",
        linewidth=1,
    )
    adjust_xaxis(axins)
    axins.set_xticklabels([])
    axins.tick_params(labelsize=8)
    axins.legend(fontsize=7, loc="upper right")

    ax.indicate_inset_zoom(axins, edgecolor="0.2")

    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "dispersion.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_portfolio_costs(
    lbda_ports: LbdaPortfolio, save_fig: bool = True, **kwargs
) -> None:
    """Plot the actual costs of the backtested portfolios vs calibrated TC penalties."""

    fig, axs = plt.subplots(3, 2, **kwargs)

    labels = ["ew", "mr", "trend"]
    title_labels = [
        f"Equal-weight ($\\Lambda_1={lbda_ports.Lbda_1_dict['ew']:.2f}, \\Lambda_2=--$)",
        f"Mean reversion ($\\Lambda_1={lbda_ports.Lbda_1_dict['mr']:.2f}, \\Lambda_2={lbda_ports.Lbda_2_dict['mr']:.2f}$)",
        f"Trend ($\\Lambda_1={lbda_ports.Lbda_1_dict['trend']:.2f}, \\Lambda_2={lbda_ports.Lbda_2_dict['trend']:.2f}$)",
    ]
    legend_labels = {
        "left": [
            "$\\log V^{\\lambda,\\, \\mathrm{gross}}- \\log V^{\\lambda,\\, \\mathrm{net}}$",
            "$\\mathit{TC}^{\\lambda}$",
        ],
        "right": [
            "$\\log V^{\\lambda,\\, \\mathrm{net}}$",
            "$\\log V^{\\lambda,\\, \\mathrm{gross}}-\\mathit{TC}^{\\lambda}$",
        ],
    }

    ws, we = lbda_ports.warmup_dates

    for i, ax_row in enumerate(axs):
        label = labels[i]
        active_tc_IS = lbda_ports.active_costs_df[label].loc[ws:we].diff().cumsum()
        active_tc_OOS = lbda_ports.active_costs_df[label].loc[we:].diff().cumsum()
        tc_penalty_IS = lbda_ports.tc_penalty_df[label].loc[ws:we].diff().cumsum()
        tc_penalty_OOS = lbda_ports.tc_penalty_df[label].loc[we:].diff().cumsum()
        log_V_gross = lbda_ports.gross_log_V_df[label]
        log_V_net = lbda_ports.net_log_V_df[label]
        ax_row[0].plot(active_tc_IS, color="grey")
        ax_row[0].plot(tc_penalty_IS, color="r")
        ax_row[0].plot(active_tc_OOS, color="grey")
        ax_row[0].plot(tc_penalty_OOS, color="r")
        ax_row[0].axvline(pd.Timestamp(we), color="black", linestyle="--")
        ax_row[0].set_title(title_labels[i])

        ax_row[1].plot(log_V_net, color="grey")
        ax_row[1].plot(log_V_gross.subtract(tc_penalty_OOS), color="b")
        ax_row[1].axhline(0, color="black", linestyle="--")
        ax_row[1].set_title(title_labels[i])
        ax_row[0].set_ylabel("Cumulative Cost")
        ax_row[1].set_ylabel("Log Relative Value")

    ax_row[0].set_xlabel("Year")
    ax_row[1].set_xlabel("Year")
    ax_row[1].set_yticks(np.arange(-0.25, 1.0, 0.25))

    adjust_xaxis(axs.flatten())

    left_handles = axs[0, 0].get_lines()[:2]
    right_handles = axs[0, 1].get_lines()[:2]
    fig.legend(
        handles=left_handles,
        labels=legend_labels["left"],
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.27, 0),
    )
    fig.legend(
        handles=right_handles,
        labels=legend_labels["right"],
        loc="upper center",
        ncols=2,
        bbox_to_anchor=(0.77, 0),
        frameon=True,
    )
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "portfolio_costs.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_portfolio_performance(
    lbda_ports: LbdaPortfolio,
    diversity_monthly: pd.Series,
    dispersion_monthly: pd.Series,
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Plot portfolio performance of optimal lambda-tilt processes."""
    if lbda_ports.net_wealth_df is None:
        raise ValueError(
            "Backtest has not been run yet. Please call lbda_ports.run_backtest() before plotting portfolio performance."
        )

    fig, axs = plt.subplots(2, 2, **kwargs)

    colors = ["black", "black", "blue", "red"]
    styles = ["--", "-", "-", "-"]
    labels = {
        "mkt": "$\\lambda\\equiv 0$ (mkt)",
        "ew": "$\\lambda\\equiv 1$ (ew)",
        "mr": "$\\lambda^{*,\\, \\mathrm{{mr}}}$",
        "trend": "$\\lambda^{*,\\, \\mathrm{{trend}}}$",
    }

    lambda_signal_dict = {
        "mkt": np.zeros(len(lbda_ports.lbda["mr"])),
        "ew": np.ones(len(lbda_ports.lbda["mr"])),
        "mr": lbda_ports.lbda["mr"],
        "trend": lbda_ports.lbda["trend"],
    }
    lbda_signal_df = pd.DataFrame(lambda_signal_dict, index=lbda_ports.lbda["mr"].index)

    _, we = lbda_ports.warmup_dates

    for i, (key, val) in enumerate(labels.items()):
        lbda_ports.net_wealth_df[key].plot(
            ax=axs[1, 0],
            color=colors[i],
            linestyle=styles[i],
            label=val,
            zorder=len(lbda_ports.net_wealth_df.columns) - i,
        )

        lbda_ports.net_log_V_df[key].plot(
            ax=axs[1, 1],
            color=colors[i],
            linestyle=styles[i],
            label=val,
            zorder=len(lbda_ports.net_log_V_df.columns) - i,
        )

        lbda_signal_df[key].loc[we:].plot(
            ax=axs[0, 1],
            color=colors[i],
            linestyle=styles[i],
            label=val,
            zorder=len(lbda_signal_df.columns) - i,
        )

    axs[1, 0].set_ylabel("Wealth")
    axs[1, 0].set_xlabel("Year")
    axs[1, 1].set_ylabel("Log Relative Value")
    axs[0, 1].set_ylabel("$\\lambda$")
    axs[1, 1].set_xlabel("Year")

    diversity_monthly.loc[we:].plot(ax=axs[0, 0], color=DIVERSITY_COLOR, label="$\\varphi$")

    ax2 = axs[0, 0].twinx()
    dispersion_monthly.loc[we:].plot(
        ax=ax2, color=DISPERSION_COLOR, label="$\\widehat{\\delta}$"
    )

    axs[0, 0].axhline(
        diversity_monthly.loc[:we].mean(),
        linestyle="-.",
        color="black",
    )

    axs[0, 0].set_zorder(2)
    axs[0, 0].patch.set_visible(False)

    axs[0, 0].set_ylabel("Diversity")
    axs[0, 0].set_xlabel("Year")
    axs[0, 1].set_xlabel("Year")
    ax2.set_ylabel("Dispersion")

    axs[0, 0].legend().remove()
    axs[0, 1].legend().remove()
    axs[1, 0].legend().remove()
    axs[1, 1].legend().remove()
    handles = axs[0, 0].get_lines()[:1] + ax2.get_lines()[:1] + axs[1, 0].get_lines()
    labels = [line.get_label() for line in handles]

    fig.legend(
        handles=handles,
        labels=labels,
        ncols=8,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
    )
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "portfolio_performance.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_ir_surf(
    Lbda_1_idxs: np.ndarray,
    Lbda_2_idxs: np.ndarray,
    ir_arrays: dict[str, np.ndarray],
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Draw an information ratio surface over the (Lbda_1, Lbda_2) grid for each portfolio, on a shared colour scale and z-axis."""

    _, axs = plt.subplots(1, 2, **kwargs)

    cmap = compress_cmap(plt.cm.jet.reversed(), exponent=0.6)

    X, Y = np.meshgrid(Lbda_1_idxs, Lbda_2_idxs)

    vmin = np.min(np.concatenate(list(ir_arrays.values()))) - 0.1
    vmax = np.max(np.concatenate(list(ir_arrays.values()))) + 0.1

    norm = Normalize(vmin=vmin, vmax=vmax)

    for i, port in enumerate(ir_arrays.keys()):
        ax = axs[i]
        Z = ir_arrays[port].T
        ax.plot_surface(
            X,
            Y,
            Z,
            cmap=cmap,
            norm=norm,
            rcount=Z.shape[0],
            ccount=Z.shape[1],
            edgecolor="k",
            linewidth=0.15,
            antialiased=True,
        )

        ax.set_xlabel("$N_1$")
        ax.set_ylabel("$N_2$")
        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel("Information Ratio", rotation=90)
        ax.set_zlim(vmin, vmax)  # shared z-limits across both subplots
        # ax.set_box_aspect((1, 1, 0.5))   # flatten z so the peak isn't too tall
        ax.set_xticks(Lbda_1_idxs[::2])
        ax.set_yticks(Lbda_2_idxs[::2])
        ax.view_init(elev=30, azim=-127)  # MATLAB-ish default viewing angle
        ax.set_box_aspect((1, 1, 0.6), zoom=1.1)
        ax.minorticks_off()
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((1.0, 1.0, 1.0, 1.0))  # white panes

    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "ir_surface.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_sim_portfolio_performance(
    sdd_model: SDDModel,
    sim_diversity: pd.DataFrame,
    sim_dispersion: pd.DataFrame,
    Lbda_1_sim: float,
    Lbda_2_sim: float,
    target_active_risk: float,
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Backtest the tilt processes on one simulated path and plot the resulting portfolio performance."""
    lbda_dict, gross_log_V_dict, net_log_V_dict, _, _ = sdd_model.backtest(
        target_active_risk=target_active_risk,
        Lbda_1=Lbda_1_sim,
        Lbda_2=Lbda_2_sim,
        samples=[sim_diversity, sim_dispersion],
    )

    sim_diversity = sim_diversity.squeeze()
    sim_dispersion = sim_dispersion.squeeze()

    sim_lbda_df = pd.DataFrame({k: v.iloc[:, 0] for k, v in lbda_dict.items()})
    sim_log_V_gross_df = pd.DataFrame(
        {k: v.iloc[:, 0] for k, v in gross_log_V_dict.items()}
    )
    sim_log_V_net_df = pd.DataFrame(
        {k: v.iloc[:, 0] for k, v in net_log_V_dict.items()}
    )

    fig, axs = plt.subplots(2, 2, **kwargs)
    colors = ["black", "blue", "red"]

    axs[0, 1].axhline(0, color="black", label="$\\mu$", linestyle="--", zorder=-1)
    axs[1, 0].axhline(0, color="black", label="$\\mu$", linestyle="--", zorder=-1)
    axs[1, 1].axhline(0, color="black", label="$\\mu$", linestyle="--", zorder=-1)

    sim_diversity.plot(ax=axs[0, 0], label="$\\varphi$", color=DIVERSITY_COLOR)
    axs[0, 0].axhline(
        sdd_model.config["phi_bar"],
        color="black",
        label="$\\bar{\\varphi}$",
        linestyle="-.",
    )
    axs[0, 0].set_ylabel("Diversity")

    ax2 = axs[0, 0].twinx()
    sim_dispersion.plot(ax=ax2, label="$\\widehat{\\delta}$", color=DISPERSION_COLOR)
    ax2.set_ylabel("Dispersion")

    axs[0, 0].set_zorder(2)
    axs[0, 0].patch.set_visible(False)
    ax2.set_zorder(1)

    sim_lbda_df.plot(ax=axs[0, 1], color=colors)
    axs[0, 1].set_ylabel("Lambda")

    sim_log_V_gross_df.plot(ax=axs[1, 0], color=colors)
    sim_log_V_net_df.plot(ax=axs[1, 1], color=colors)

    axs[1, 0].set_ylabel("Log relative value (unpenalized)")
    axs[1, 1].set_ylabel("Log relative value (penalized)")
    axs[1, 1].sharey(axs[1, 0])

    axs[0, 1].legend().remove()
    axs[1, 0].legend().remove()
    axs[1, 1].legend().remove()
    axs[1, 0].set_xlabel("Time (Years)")
    axs[1, 1].set_xlabel("Time (Years)")

    handles, _ = axs[0, 1].get_legend_handles_labels()

    labels = (
        "$\\lambda\\equiv 0$ (mkt)",
        "$\\lambda\\equiv 1$ (ew)",
        "$\\lambda^{*}_0$",
        "$\\lambda^{*}$",
    )

    fig.legend(handles, labels, ncols=4, loc="upper center", bbox_to_anchor=(0.5, 0))

    adjust_xaxis(axs.flatten())

    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "sim_portfolio_performance.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plotter(
    data: pd.Series,
    IS_end_date: str,
    title: str,
    ax: Axes,
    include_vline: bool = False,
    xlabel: str | None = None,
    ylabel: str | None = None,
    **kwargs,
) -> None:
    """Draw a series on the given axis with optional title, axis labels and an in-sample/out-of-sample split line."""
    ax.plot(data, **kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    adjust_xaxis(ax)
    if include_vline:
        ax.axvline(
            x=pd.to_datetime(IS_end_date),
            color="k",
            linestyle="--",
            label="in-sample / out-of-sample split",
        )


def simulate_and_plot(
    sdd_model: SDDModel,
    diversity_monthly: pd.Series,
    dispersion_monthly: pd.Series,
    IS_end_date: str,
    seed: int | None,
    save_fig: bool = True,
    **kwargs,
) -> tuple[pd.Series, pd.Series]:
    """Sample paths from SDD model and plot against realized diversity and dispersion used to calibrate model."""
    _, axs = plt.subplots(2, 2, **kwargs)

    sim_phi, sim_delta = sdd_model.sample(
        N=diversity_monthly.size,
        n_samples=1,
        phi_0=diversity_monthly.iloc[0],
        delta_0=dispersion_monthly.iloc[0],
        seed=seed,
    )

    sim_phi = sim_phi.squeeze()
    sim_delta = sim_delta.squeeze()

    sim_phi.index = sim_phi.index / 12.0
    sim_delta.index = sim_delta.index / 12.0

    sim_phi_bulk, sim_delta_bulk = sdd_model.sample(
        N=diversity_monthly.size,
        n_samples=3,
        phi_0=diversity_monthly.iloc[0],
        delta_0=dispersion_monthly.iloc[0],
        seed=seed,
    )

    sim_phi_bulk.index = sim_phi_bulk.index / 12.0
    sim_delta_bulk.index = sim_delta_bulk.index / 12.0

    # real-data

    plotter(
        diversity_monthly,
        IS_end_date,
        "Market Diversity",
        axs[0, 0],
        include_vline=True,
        color="b",
        xlabel="Year",
        ylabel="$\\varphi$",
    )

    plotter(
        dispersion_monthly,
        IS_end_date,
        "Dispersion",
        axs[0, 1],
        include_vline=True,
        color="b",
        xlabel="Year",
        ylabel="$\\widehat{\\delta}$",
    )

    # simulation
    plotter(
        sim_phi,
        IS_end_date,
        "Simulated Diversity",
        axs[1, 0],
        color="b",
        xlabel="Time (Years)",
        ylabel="$\\varphi^\\mathrm{SDD}$",
    )

    axs[1, 0].plot(sim_phi_bulk, color="grey", alpha=0.3, zorder=0)
    plotter(
        sim_delta,
        IS_end_date,
        "Simulated Dispersion",
        axs[1, 1],
        color="b",
        xlabel="Time (Years)",
        ylabel="$\\delta^\\mathrm{SDD}$",
    )

    axs[1, 1].plot(sim_delta_bulk, color="grey", alpha=0.3, zorder=0)
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "sim_paths.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")

    return sim_phi, sim_delta


def plot_sim_dists(
    sdd_model: SDDModel,
    sim_phi: pd.Series,
    IS_data: pd.DataFrame,
    OOS_data: pd.DataFrame,
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Compare empirical distributions of diversity and realized dispersion against those of simulated SDD model."""
    stationary_var = (sdd_model.config["nu_delta"] ** 2) / (
        2 * sdd_model.config["kappa_delta"]
    )
    _, axs = plt.subplots(1, 2, **kwargs)

    diversity_diff_IS = IS_data["diversity"].diff().dropna()
    diversity_diff_OOS = OOS_data["diversity"].diff().dropna()
    sim_phi_diff = sim_phi.diff().dropna()

    sns.kdeplot(
        diversity_diff_IS,
        bw_adjust=1,
        color="b",
        fill=False,
        ax=axs[0],
        label="$\\Delta \\varphi$ (IS)",
    )
    sns.kdeplot(
        diversity_diff_OOS,
        bw_adjust=1,
        color="r",
        fill=False,
        ax=axs[0],
        label="$\\Delta \\varphi$ (OOS)",
    )
    sns.kdeplot(
        sim_phi_diff,
        bw_adjust=1,
        color="orange",
        fill=False,
        ax=axs[0],
        label="$\\Delta \\varphi^\\mathrm{SDD}$",
    )

    axs[0].set_ylabel("Density")
    axs[0].set_xlabel("$\\Delta \\varphi$")
    axs[0].legend()

    sns.kdeplot(
        IS_data["dispersion"],
        color="b",
        fill=False,
        ax=axs[1],
        label="$\\widehat{\\delta}$ (IS)",
        bw_adjust=0.5,
    )
    sns.kdeplot(
        OOS_data["dispersion"],
        color="r",
        fill=False,
        ax=axs[1],
        label="$\\widehat{\\delta}$ (OOS)",
        bw_adjust=0.5,
    )
    x_min = 0
    x_max = max(IS_data["dispersion"].max(), OOS_data["dispersion"].max())

    stationary_simul_pdf = stats.lognorm.pdf(
        np.linspace(x_min, x_max, 1000),
        s=np.sqrt(stationary_var),
        loc=0,
        scale=sdd_model.config["delta_bar"],
    )

    axs[1].plot(
        np.linspace(x_min, x_max, 1000),
        stationary_simul_pdf,
        color="orange",
        label="$\\pi_\\delta$",
    )
    axs[1].set_xlabel("$\\delta$")
    axs[1].legend()
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "sim_dists.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_sim_moments(
    sdd_model: SDDModel,
    IS_monthly_data: pd.DataFrame,
    OOS_monthly_data: pd.DataFrame,
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Compare in-sample and out-of-sample dispersion moments against the SDD model's stationary law."""
    stationary_var = (sdd_model.config["nu_delta"] ** 2) / (
        2 * sdd_model.config["kappa_delta"]
    )

    _, axs = plt.subplots(1, 2, **kwargs)

    print("In-sample fit:")
    right_tail_IS_monthly_data = utils.plot_cdf_tail(
        np.log(IS_monthly_data["dispersion"])
    )
    axs[0].plot(
        right_tail_IS_monthly_data["right_tail_x"],
        right_tail_IS_monthly_data["right_tail_fit"],
        color="black",
    )
    axs[0].scatter(
        right_tail_IS_monthly_data["right_tail_x"],
        right_tail_IS_monthly_data["right_tail_y"],
        marker="o",
        linestyle="None",
        facecolor="none",
        edgecolor="b",
        label="$\\widehat{\\delta}$ (IS)",
    )

    print("Out-of-sample fit:")
    right_tail_OOS_monthly_data = utils.plot_cdf_tail(
        np.log(OOS_monthly_data["dispersion"])
    )
    axs[0].plot(
        right_tail_OOS_monthly_data["right_tail_x"],
        right_tail_OOS_monthly_data["right_tail_fit"],
        color="black",
    )
    axs[0].scatter(
        right_tail_OOS_monthly_data["right_tail_x"],
        right_tail_OOS_monthly_data["right_tail_y"],
        marker="o",
        linestyle="None",
        facecolor="none",
        edgecolor="r",
        label="$\\widehat{\\delta}$ (OOS)",
    )

    x_min, x_max = axs[0].get_xlim()
    simul_tail = 1 - stats.norm.cdf(
        np.linspace(x_min, x_max, 1000),
        loc=np.log(sdd_model.config["delta_bar"]),
        scale=np.sqrt(stationary_var),
    )
    axs[0].plot(
        np.linspace(x_min, x_max, 1000),
        simul_tail,
        color="orange",
        label="$\\pi_\\delta$",
    )

    axs[0].set_xlabel("$x$")
    axs[0].set_ylabel("$\\mathbf{P}(\\log X > x)$")
    axs[0].set_yscale("log")
    axs[0].legend(loc="lower left")

    # in-sample
    # acf_IS = utils.acf(IS_monthly_data["dispersion"], lags=20)
    acf_IS = sm.tsa.acf(IS_monthly_data["dispersion"], nlags=20)[1:]

    axs[1].scatter(
        np.arange(1, 21), acf_IS, color="blue", s=6, label="$\\widehat{\\delta}$ (IS)"
    )

    acf_OOS = sm.tsa.acf(OOS_monthly_data["dispersion"], nlags=20)[1:]
    axs[1].scatter(
        np.arange(1, 21), acf_OOS, color="red", s=6, label="$\\widehat{\\delta}$ (OOS)"
    )

    axs[1].plot(
        np.arange(1, 21),
        autocorr_sim(
            stationary_var,
            kappa_delta=sdd_model.config["kappa_delta"],
            delta_t=1 / 12,
            max_lags=21,
        )[1:],
        color="orange",
        label="$\\pi_\\delta$",
    )

    x_ci = stats.norm.ppf(0.975) / np.sqrt(len(OOS_monthly_data))
    axs[1].axhline(y=x_ci, color="black", linestyle="--")

    axs[1].set_xlabel("Lag $\\ell$")
    axs[1].set_ylabel("Autocorrelation")

    axs[1].legend()
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "sim_moments.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")


def plot_sim_lowess(
    sim_phi: pd.Series,
    sim_delta: pd.Series,
    IS_monthly_data: pd.DataFrame,
    OOS_monthly_data: pd.DataFrame,
    frac: float,
    save_fig: bool = True,
    **kwargs,
) -> None:
    """Draw LOWESS fits of dispersion against diversity increments for the in-sample, out-of-sample and simulated series."""
    _, axs = plt.subplots(1, 3, **kwargs)

    # in-sample
    diversity_diff_IS = IS_monthly_data["diversity"].diff().dropna()
    dispersion_diff_IS = IS_monthly_data["dispersion"].iloc[1:]

    lowess = sm.nonparametric.lowess(dispersion_diff_IS, diversity_diff_IS, frac=frac)

    axs[0].scatter(
        diversity_diff_IS, dispersion_diff_IS, edgecolor="b", facecolor="none", s=4
    )

    axs[0].plot(lowess[:, 0], lowess[:, 1], color="red", label="LOWESS Fit")
    axs[0].set_ylabel("$\\widehat{\\delta}$ (IS)")
    axs[0].set_xlabel("$\\Delta \\varphi$ (IS)")

    # out-of-sample
    diversity_diff_OOS = OOS_monthly_data["diversity"].diff().dropna()
    dispersion_diff_OOS = OOS_monthly_data["dispersion"].iloc[1:]

    lowess = sm.nonparametric.lowess(dispersion_diff_OOS, diversity_diff_OOS, frac=frac)

    axs[1].scatter(
        diversity_diff_OOS, dispersion_diff_OOS, edgecolor="b", facecolor="none", s=4
    )
    axs[1].plot(lowess[:, 0], lowess[:, 1], color="red", label="LOWESS Fit")
    axs[1].set_ylabel("$\\widehat{\\delta}$ (OOS)")
    axs[1].set_xlabel("$\\Delta \\varphi$ (OOS)")

    # simulation
    diversity_diff_sim = sim_phi.diff().dropna()

    lowess = sm.nonparametric.lowess(sim_delta.iloc[1:], diversity_diff_sim, frac=frac)

    axs[2].scatter(
        diversity_diff_sim, sim_delta.iloc[1:], edgecolor="b", facecolor="none", s=4
    )
    axs[2].plot(lowess[:, 0], lowess[:, 1], color="red", label="LOWESS Fit")

    axs[2].set_xlabel("$\\Delta \\varphi^\\mathrm{SDD}$")
    axs[2].set_ylabel("$\\delta^\\mathrm{SDD}$")
    if save_fig:
        os.makedirs(FIGURES, exist_ok=True)
        save_path = FIGURES / "sim_lowess.pdf"
        plt.savefig(save_path, format="pdf", dpi=600)
        print(f"Saved figure to {relative_path(save_path)}")
