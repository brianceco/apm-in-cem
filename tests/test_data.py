import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from paths import DATA, relative_path

START_YEAR = 1977  

# check that the raw and clean data files have been written by src/data.py
required = [
    DATA / "raw" / f"sp500_raw_{START_YEAR}.parquet",
    DATA / "raw" / f"sp500_membership_{START_YEAR}.csv",
    DATA / "processed" / f"sp500_clean_{START_YEAR}.parquet",
    DATA / "processed" / f"rets_monthly_{START_YEAR}.csv",
    DATA / "processed" / f"caps_monthly_{START_YEAR}.csv",
    DATA / "processed" / f"mkt_wgts_monthly_{START_YEAR}.csv",
    DATA / "processed" / f"eql_wgts_monthly_{START_YEAR}.csv",
]
missing = [relative_path(path) for path in required if not path.exists()]
if missing:
    raise ValueError(
        "Run `python src/data.py` first.\n"
        f"Missing {len(missing)} data file(s):\n" + "\n".join(missing)
    )

sp500_raw = pd.read_parquet(f"{DATA}/raw/sp500_raw_{START_YEAR}.parquet")

sp500_raw["date"] = pd.to_datetime(sp500_raw["date"])

sp500_clean = pd.read_parquet(f"{DATA}/processed/sp500_clean_{START_YEAR}.parquet")
sp500_clean["date"] = pd.to_datetime(sp500_clean["date"])

sp500_members = pd.read_csv(
    f"{DATA}/raw/sp500_membership_{START_YEAR}.csv",
    parse_dates=["start", "ending"],
)

monthly_rets = pd.read_csv(
    f"{DATA}/processed/rets_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

monthly_caps = pd.read_csv(
    f"{DATA}/processed/caps_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

monthly_mkt_wgts = pd.read_csv(
    f"{DATA}/processed/mkt_wgts_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)

monthly_eql_wgts = pd.read_csv(
    f"{DATA}/processed/eql_wgts_monthly_{START_YEAR}.csv",
    index_col=0,
    parse_dates=True,
)


def test_data_cleaning_no_change():
    permno = 45356

    raw_df = sp500_raw[sp500_raw["permno"] == permno].copy()
    raw_df = (
        raw_df.sort_values(by="date").reset_index(drop=True).drop(columns=["prc_flg"])
    )

    clean_df = (
        sp500_clean[sp500_clean["permno"] == permno]
        .reset_index(drop=True)
        .drop(columns=["bcktst_flg"])
    )

    assert np.all(clean_df == raw_df)


def test_wgts_close():
    assert np.allclose(monthly_eql_wgts.sum(axis=1), 1)
    assert np.allclose(monthly_mkt_wgts.sum(axis=1), 1)


def test_sp500_counts():
    lb, ub = (
        480,
        520,
    )  # entrants/exists between rebalancing dates implies index not constant at 500
    assert monthly_caps.notna().sum(axis=1).between(lb, ub).all()
    assert monthly_mkt_wgts.notna().sum(axis=1).between(lb, ub).all()
    assert monthly_eql_wgts.notna().sum(axis=1).between(lb, ub).all()
