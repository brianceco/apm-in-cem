import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest as bt
from paths import DATA, relative_path

START_YEAR = 1977

PROCESSED = DATA / "processed"

# check that the monthly panels and signals have been written by src/data.py
required = [
    PROCESSED / f"rets_monthly_{START_YEAR}.csv",
    PROCESSED / f"caps_monthly_{START_YEAR}.csv",
    PROCESSED / f"mkt_wgts_monthly_{START_YEAR}.csv",
    PROCESSED / f"eql_wgts_monthly_{START_YEAR}.csv",
    PROCESSED / f"signals_{START_YEAR}.csv",
    PROCESSED / f"signals_clean_{START_YEAR}.csv",
]
missing = [relative_path(path) for path in required if not path.exists()]
if missing:
    raise ValueError(
        "Run `python src/data.py` first.\n"
        f"Missing {len(missing)} data file(s):\n" + "\n".join(missing)
    )

monthly_rets = pd.read_csv(
    PROCESSED / f"rets_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)
monthly_caps = pd.read_csv(
    PROCESSED / f"caps_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)
monthly_mkt_wgts = pd.read_csv(
    PROCESSED / f"mkt_wgts_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)
monthly_eql_wgts = pd.read_csv(
    PROCESSED / f"eql_wgts_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

signals = pd.read_csv(
    PROCESSED / f"signals_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

signals_clean = pd.read_csv(
    PROCESSED / f"signals_clean_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

#######################
# Helper functions
#######################


def create_backtest_data():
    """Create synthetic backtest data (rets, caps) satisfying the following:
    1. Column 0 represents a stock which enters the S&P500 after start date (nan rets, nan caps).
    2. Column 1 represents a stock which exits the S&P500 before end date (nan rets, nan caps).
    5. Column 2 represents a stock which is always in the S&P500 (rets, caps).
    """
    T = 20
    N = 3
    nan_out = 5

    rng = np.random.default_rng(seed=1)

    real_rets = rng.uniform(low=-0.9, high=1.0, size=(T, N))
    div_rets = rng.uniform(low=0, high=0.1, size=(T, N))
    rets = real_rets + div_rets
    caps = (1 + real_rets).cumprod(axis=0)

    caps[:nan_out, 0] = np.nan
    rets[:nan_out, 0] = np.nan
    div_rets[: nan_out + 1, 0] = 0.0
    real_rets[: nan_out + 1, 0] = 0.0
    real_rets[nan_out, 0] = rets[nan_out, 0]

    caps[-nan_out:, 1] = np.nan
    rets[-nan_out:, 1] = np.nan
    div_rets[-nan_out:, 1] = 0.0
    real_rets[-nan_out:, 1] = 0.0

    rets = pd.DataFrame(rets)
    caps = pd.DataFrame(caps)

    return rets, caps, real_rets, div_rets


def get_backtest_data(T, t_switch):
    caps = pd.DataFrame(np.ones([T, 2]))
    rets = pd.DataFrame(np.zeros([T, 2]))

    port = np.zeros([T, 2])
    port[:t_switch, 0] = 2
    port[t_switch:, 0] = 1
    port[:t_switch, 1] = -1
    port[t_switch:, 1] = 0

    port = pd.DataFrame(port)
    return caps, rets, port


####################
# Tests
####################


def test_ir_pct_hand_computed():
    wealth = pd.Series([100.0, 110.0, 99.0, 108.9])
    benchmark = pd.Series([1.0, 1.0, 1.0, 1.0])
    assert np.isclose(bt.get_ir_pct(wealth, benchmark, dt=1 / 12), 1.0)


def test_mdd_rt_hand_computed():
    wealth = pd.Series([100.0, 120.0, 60.0, 80.0, 200.0])
    assert np.isclose(bt.get_mdd(wealth), 0.5)
    assert bt.get_rt(wealth) == 2


def test_backtest_no_tc():
    """Test that running the portfolio with no transaction costs produces the expected wealth and trading cost outcomes."""
    naive_mkt_ret = monthly_mkt_wgts.shift(1).multiply(monthly_rets).sum(axis=1)

    wealth, ge, _, tc = bt.run_portfolio(
        W0=1.0,
        tc=0.0,
        caps=monthly_caps,
        rets=monthly_rets,
        tgt_port=monthly_mkt_wgts,
    )
    mkt_ret = wealth.pct_change().fillna(0.0)

    assert (
        np.allclose(naive_mkt_ret, mkt_ret, atol=1e-6)
        & np.allclose(ge, 1.0, atol=1e-6)
        & np.allclose(tc, 0.0, atol=1e-6)
    )


def test_real_and_div_rets():
    """Test that the real and diversified returns are correctly computed."""
    rets, caps, true_rr, true_dr = create_backtest_data()

    real_rets, div_rets = bt.get_real_and_div_rets(caps, rets)

    assert np.allclose(real_rets[1:], true_rr[1:])
    assert np.allclose(div_rets[1:], true_dr[1:])


def test_run_portfolio_costs():
    """Test that transaction costs are correctly applied when switching portfolio allocations."""
    T = 20
    t_switch = 10
    W0 = 1000
    tc = 0.1
    c = 7 / 9
    W = W0 * c

    caps, rets, port = get_backtest_data(T=T, t_switch=t_switch)

    true_tc = np.zeros(T)
    true_tc[t_switch] = W0 - W

    true_wealth = np.array([W0] * t_switch + [W] * (T - t_switch))

    wealth, _, turnover, transaction_costs = bt.run_portfolio(
        W0=W0, tc=tc, caps=caps, rets=rets, tgt_port=port
    )

    assert np.allclose(wealth, true_wealth)
    assert np.isclose(turnover, 2 / (T - 1))
    assert np.allclose(transaction_costs, true_tc)
