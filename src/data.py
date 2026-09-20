"""Dataset utilities for downloading and processing S&P 500 monthly return + market cap data from WRDS, as well as S&P 500 implied dispersion index from CBOE and Fama-French 5-factor data from Dartmouth. The processed data is saved to disk for use in backtesting and analysis."""

import io
import os
import sys
import urllib.error
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import wrds

import utils
from config import DT_DAILY
from paths import DATA, relative_path

# -----------------------------
# Raw Data Loading
# -----------------------------

DSPX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/DSPX_History.csv"

FAMA_FRENCH_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip"

START_DATE = "1977-01-01"  # S&P500 in CRSP database first reaches 99% coverage in 1957-03-01. We begin at 1977-01-01 as financials are not included in the S&P500 until 1976-12-31 which introduces artificial spike in diversity
END_DATE = "2024-12-31"

def download_sp500_raw(
    start_date: str, end_date: str, save: bool = True
) -> pd.DataFrame:
    """Download daily CRSP returns, lagged market caps and price flags for S&P 500 constituents from WRDS, from start_date onwards."""

    tmp_end_date = str((pd.to_datetime(end_date) + pd.offsets.MonthEnd(1)).date())
    # used for shifting lagged market caps on final month of end date

    print(f"Downloading S&P500 data from WRDS starting from {start_date}...")
    db = wrds.Connection()

    query_data = f"""
    SELECT d.dlycaldt, d.permno, d.dlyprevcap, d.dlyret, d.dlyprcflg 
    FROM crsp.StkDlySecurityData AS d
    JOIN (SELECT DISTINCT permno 
    FROM crsp.msp500list
    WHERE ending >= '{start_date}') as m
        ON d.permno = m.permno
    WHERE d.dlycaldt BETWEEN '{start_date}' and '{tmp_end_date}'
    """  # Includes delisting returns unlike Ruf's query for common shares

    df = db.raw_sql(query_data, date_cols="dlycaldt")

    print(
        f"Downloaded {len(df)} rows of S&P500 data from WRDS starting from {start_date}."
    )

    db.close()

    df = df.rename(
        columns={
            "dlycaldt": "date",
            "dlyprevcap": "cap_lag1",
            "dlyret": "ret",
            "dlyprcflg": "prc_flg",
        }
    )

    df = df.sort_values(["permno", "date"], ignore_index=True)
    df["cap"] = df.groupby(["permno"])["cap_lag1"].shift(-1)

    df = df[df["date"] <= pd.to_datetime(end_date)]

    sy = start_date[:4]

    if save:
        os.makedirs(DATA / "raw", exist_ok=True)
        save_path = DATA / "raw" / f"sp500_raw_{sy}.parquet"
        df.to_parquet(save_path, index=False)
        print(f"Saved raw S&P 500 data to {relative_path(save_path)}")
    return df


def download_sp500_membership(start_date: str, save: bool = True) -> pd.DataFrame:
    """Download S&P 500 membership data from WRDS starting from start_date."""

    db = wrds.Connection()
    query = f"""
    SELECT permno, start, ending
    FROM crsp.msp500list
    WHERE ending >= '{start_date}'
    """
    df = db.raw_sql(query, date_cols=["start", "ending"])
    db.close()

    sy = start_date[:4]

    if save:
        os.makedirs(DATA / "raw", exist_ok=True)
        save_path = DATA / "raw" / f"sp500_membership_{sy}.csv"
        df.to_csv(save_path, index=False)
        print(f"Saved S&P 500 membership data to {relative_path(save_path)}")
    return df


# -----------------------------
# Preprocessing
# -----------------------------

PRC_FLAGS = ["TR", "BA", "MP", "DA", "NT", "DP", "HA", "DM", "SU"]

# TR = total return
# BA = bid-ask average
# MP = missing price
# DA = daily average
# NT = not traded
# DP = delisting price
# HA = half-adjusted
# DM = dividend missing
# SU = suspended

PROBLEMATIC_PRC_FLAGS = ["NT", "MP", "HA", "SU", "DM"]

# Return flags
FLAG_PROBLEMATIC_INTERMEDIATE_RETURN = 2
# If a return or previous market cap is missing or price flag is in problematic price flags.

FLAG_TEMPORARY_DELISTING = 3

FLAG_DELRET_MISSING = 4

FLAG_MISSING_RETURN_IMPUTED = 5
# If return was missing but the trading days before and after have 'good' returns.
# The missing returns are replaced by 0.0 on those days.

FLAG_RETURN_BASED_ON_BA = 1
# if return is based on a bid-ask average and not corresponding to a missing delisting or a problematic intermediate return

# Placehorder returns
CUTOFF_LARGE_RETURN = 1.0
CUTOFF_SMALL_RETURN = -0.5

TEMPORARY_DELISTING_RETURN = -0.1
MISSING_DELIST_RETURN = -0.3
# We define a security as being "temoprarily delisted" if the security's previous market cap is available but return is not


def preprocess_prc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create a new column 'bcktst_flg' based on 'prc_flg', 'ret', and 'cap_lag1'. Set default to 0 for good returns, 1 for returns based on bid-ask average, and 2 for problematic returns."""
    df["bcktst_flg"] = 0

    ba_mask = df["prc_flg"] == "BA"
    print(
        f"There are {ba_mask.sum() / len(ba_mask) * 100:.2f}% of prices based on bid-ask spread rather than market close."
    )
    df.loc[ba_mask, "bcktst_flg"] = FLAG_RETURN_BASED_ON_BA

    problematic_prc_mask = (
        df["ret"].isna()
        | df["cap_lag1"].isna()
        | df["prc_flg"].isin(PROBLEMATIC_PRC_FLAGS)
    )
    df = df.drop(columns=["prc_flg"])
    print(
        f"There are {problematic_prc_mask.sum() / len(problematic_prc_mask) * 100:.2f}% of problematic prices."
    )
    df.loc[problematic_prc_mask, "bcktst_flg"] = FLAG_PROBLEMATIC_INTERMEDIATE_RETURN

    df = df.pivot(index="date", columns="permno")
    assert df.index.is_monotonic_increasing

    return df


def get_prob_rets_at_end_mask(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["ret"].isnull() | df["cap_lag1"].isnull() | (df["bcktst_flg"] != 0)
    )  # ret or mkt cap missing or nonzero backtest flag

    mask = mask[
        ::-1
    ].cummin()[
        ::-1
    ]  # for each permno, identify problematic return which occur after the last valid return

    mask = mask.mask(
        df["bcktst_flg"].isnull()[::-1].cummin()[::-1], other=False
    )  # Set mask to False following last nonempty backtest flag

    print(
        f"There are {mask.any().sum()} ({mask.any().mean() * 100:.2f}%) PERMNOs which have problematic returns at the end of their time series."
    )

    print(
        f"There are {mask.sum().sum()} problematic returns at the end of time series to be modified."
    )

    mask = (
        mask
        | (mask.shift(-1, fill_value=False))
        & ~mask
        & df["cap_lag1"].isnull().shift(-1, fill_value=False)
    )  # in the second intersection mask, the first two factors pick up the last valid return, the third factor picks up cases where the first problematic return has a missing dlyprevcap
    return mask


def clean_returns_and_delistings(df: pd.DataFrame) -> pd.DataFrame:
    """Mask problematic returns at the start of each PERMNO's series, drop PERMNOs that never have a valid return, and impute missing delisting returns at the end."""
    # cleaning beginning of time series
    mask = df["ret"].isnull() | df["cap_lag1"].isnull() | (df["bcktst_flg"] != 0)
    mask = mask.cummin()  # this ensures that for each permno, represented by a column, we only care about problematic returns which occur before the first valid return

    # zero out all problematic returns at the beginning of each time series
    df = df.mask(mask)

    bl = mask.iloc[-1]
    print(f"There are {bl.sum()} PERMNOs which never have a single valid return.")
    df = df.loc[:, df.columns.get_level_values("permno").isin(bl.index[~bl])]

    # cleaning end of time series and set missing delisting returns to MISSING_DELRET_RETURN
    mask = get_prob_rets_at_end_mask(df)

    mask_first_return = mask & ~mask.shift(
        1, fill_value=False
    )  # the first problematic return in each time series

    mask_others = (
        mask & ~mask_first_return
    )  # all other problematic returns in each time series

    df["ret"] = df["ret"].mask(
        mask_first_return,
        other=df["ret"]
        .fillna(0)
        .add(1)
        .multiply(1 + MISSING_DELIST_RETURN)
        .subtract(1),
    )

    df["bcktst_flg"] = df["bcktst_flg"].mask(
        mask_first_return, other=FLAG_DELRET_MISSING
    )

    # we NaN out all other (i.e. non-first) problematic returns at the end of each time series
    df = df.mask(mask_others)

    # check that each return time series has at least one value
    assert df["ret"].notnull().any().all()

    # check that if a return is provided then so is the dlyprevcap
    assert ~(df["ret"].notnull() & df["cap_lag1"].isnull()).any().any()

    return df


def clean_temporary_delistings_and_missing_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle the three cases of missing return by dropping those that follow another problematic return, imputing temporary delistings, and filling the rest with zero."""
    # There are three cases of missing returns to handle:
    # (a) previous (t-1) and following (t+1) returns are valid (treat as non-trade day)
    # (b) previous (t-1) exists but following (t+1) does not exist (temporary delisting)
    # (c) previous (t-1) is problematic or missing

    mask = (
        df["ret"].isnull()
        & df["bcktst_flg"].notnull()
        & (
            df["ret"].isnull()
            | df["bcktst_flg"].eq(FLAG_PROBLEMATIC_INTERMEDIATE_RETURN)
        ).shift(1, fill_value=False)
    )  # missing returns that follow a problematic (i.e. flagged) or missing return (case (c))

    print(
        f"There are {mask.any().sum()} ({mask.any().mean() * 100:.2f}%) PERMNOs which have missing returns following a problematic or missing return and which we remove from the investment universe."
    )
    print(f"In total, we remove {mask.sum().sum()} returns.")

    df = df.mask(mask)

    # check that each return time series has at least one value
    assert df["ret"].notnull().any().all()

    mask = df["ret"].isnull() & df["bcktst_flg"].notnull()  # case (a) or (b)
    mask_tmp_delist = mask & df["ret"].isnull().shift(-1, fill_value=False)  # case (b)
    mask_fillna = mask & ~mask_tmp_delist  # case (a)

    print(
        f"There are {mask_tmp_delist.any().sum()} ({mask_tmp_delist.any().mean() * 100:.2f}%) permnos with temporary delistings (case (b)) to be filled with {TEMPORARY_DELISTING_RETURN}."
    )

    print(
        f"There are {mask_tmp_delist.sum().sum()} temporary delistings (case (b)) in total."
    )

    print(
        f"There are {mask_fillna.any().sum()} ({mask_fillna.any().mean() * 100:.2f}%) permnos with missing returns that we fill with 0.0 (case (a))"
    )

    df["cap_lag1"] = df["cap_lag1"].mask(
        mask, other=df["cap_lag1"].bfill()
    )  # fill missing previous market cap with next valid value for cases (a) or (b)

    df["ret"] = df["ret"].mask(mask_tmp_delist, other=TEMPORARY_DELISTING_RETURN)
    df["bcktst_flg"] = df["bcktst_flg"].mask(
        mask_tmp_delist, other=FLAG_TEMPORARY_DELISTING
    )

    df["ret"] = df["ret"].mask(mask_fillna, other=0.0)
    df["bcktst_flg"] = df["bcktst_flg"].mask(
        mask_fillna, other=FLAG_MISSING_RETURN_IMPUTED
    )

    return df


def clean_extreme_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Flag returns outside [CUTOFF_SMALL_RETURN, CUTOFF_LARGE_RETURN] by adding 10 to bcktst_flg."""
    mask = df["ret"].gt(CUTOFF_LARGE_RETURN) | df["ret"].lt(CUTOFF_SMALL_RETURN)

    print(
        f"There are {mask.any().sum()} ({mask.any().mean() * 100:.2f}%) permnos with very large/small returns to be flagged."
    )

    print(
        f"There are {mask.sum().sum()} very large/small returns in total to be flagged."
    )

    df["bcktst_flg"] = df["bcktst_flg"].mask(mask, other=df["bcktst_flg"].add(10))

    return df


# -----------------------------
# Complete Pipeline
# -----------------------------
def clean_raw_data(
    data_raw: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    Process raw CRSP daily data:
    1) Add custom price flags (one of {1,10}*{0,1,2,3,4,5}).
    2) Clean data (missing/flagged returns/caps).
    3) Clean "temporary delistings" and missing returns.
    4) Flag extreme returns

    Optionally save cleaned data and return it.
    """

    data = preprocess_prc_flags(data_raw)

    print("Preprocessed price flags.")

    data = clean_returns_and_delistings(data)

    print("Cleaned returns and delistings.")

    data = clean_temporary_delistings_and_missing_returns(data)

    print("Cleaned temporary delistings and missing returns.")

    data = clean_extreme_returns(data)

    print("Flagged extreme returns.")

    # impute missing caps using previous cap and return
    mask = data["cap"].isnull() & data["cap_lag1"].notnull() & data["ret"].notnull()
    print(
        f"There are {mask.any().sum()} permnos ({mask.any().mean() * 100:.2f}%) where missing cap(s) are imputed by compounded previous cap. This represents {mask.sum().sum()} imputed caps."
    )

    data["cap"] = data["cap"].fillna(
        data["cap_lag1"] * (1 + data["ret"])
    )  # if prev_cap and return are available, estimate current cap (includes dividends)

    # convert back to long format
    data = (
        data.stack(level="permno", future_stack=True)
        .dropna(how="all")
        .sort_index()
        .reset_index()
    )

    sy = data["date"].iloc[0].year
    if save:
        os.makedirs(DATA / "processed", exist_ok=True)
        save_path = DATA / "processed" / f"sp500_clean_{sy}.parquet"
        data.to_parquet(save_path, index=False)
        print(f"Saved cleaned S&P 500 data to {relative_path(save_path)}")
    return data


def sp500_membership_mask(
    index: pd.DatetimeIndex, columns: pd.Index, sp500_members: pd.DataFrame
) -> pd.DataFrame:
    """Boolean (date x permno) frame: True while the permno is an S&P 500 constituent."""
    mask = pd.DataFrame(False, index=index, columns=columns)
    for permno, start, ending in sp500_members[
        ["permno", "start", "ending"]
    ].itertuples(index=False):
        if permno in mask.columns:
            mask.loc[start:ending, permno] = True
    return mask


def get_dataframes(
    df: pd.DataFrame, sp500_members: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pivot the cleaned long-format data into monthly returns, caps and weights, together with the daily diversity and realized dispersion signals."""
    rets = (
        df.pivot(index="date", columns="permno", values="ret")
        .sort_index()
        .astype(np.float64)
    )

    caps_lag1 = (
        df.pivot(index="date", columns="permno", values="cap_lag1")
        .sort_index()
        .astype(np.float64)
    )

    caps = (
        df.pivot(index="date", columns="permno", values="cap")
        .sort_index()
        .astype(np.float64)
    )

    sp500_mask = sp500_membership_mask(rets.index, rets.columns, sp500_members)

    # only keep S&P 500 constituents in the cap data
    caps = caps.where(sp500_mask)
    caps_lag1 = caps_lag1.where(sp500_mask)

    mkt_wgts_lag1 = caps_lag1.div(caps_lag1.sum(axis=1), axis=0)
    mkt_wgts = caps.div(caps.sum(axis=1), axis=0)
    mask = mkt_wgts_lag1 > 0.0
    eql_wgts_lag1 = mask.astype(float).apply(lambda x: x / x.sum(), axis=1)
    eql_wgts_lag1 = eql_wgts_lag1.where(mask)

    diversity = utils.get_ew_diversity(mkt_wgts)
    rd_eql = utils.get_dispersion(rets, eql_wgts_lag1) / DT_DAILY
    rd_mkt = utils.get_dispersion(rets, mkt_wgts_lag1) / DT_DAILY

    signals = pd.concat(
        [diversity, rd_eql, rd_mkt], axis=1, keys=["diversity", "rd_eql", "rd_mkt"]
    )

    # store rets, caps, wgts as monthly
    rets_monthly = (1 + rets).resample("ME").prod(min_count=1) - 1
    caps_monthly = caps.resample("ME").last(skipna=False)
    mkt_wgts_monthly = caps_monthly.div(caps_monthly.sum(axis=1), axis=0)

    mask = mkt_wgts_monthly > 0.0
    eql_wgts_monthly = mask.astype(float).apply(lambda x: x / x.sum(), axis=1)
    eql_wgts_monthly = eql_wgts_monthly.where(mask)

    return rets_monthly, caps_monthly, mkt_wgts_monthly, eql_wgts_monthly, signals


def format_data(
    df_clean_full: pd.DataFrame, sp500_members: pd.DataFrame, start_date: str
) -> None:
    """Build the monthly panels and signals from the cleaned data and write them to the processed data directory, with signals computed both with and without extreme returns."""
    print(f"Formatting data for {start_date} and saving to disk.")

    df_clean = df_clean_full[
        df_clean_full["bcktst_flg"] < 10.0
    ].copy()  # remove extreme returns

    rets_monthly, caps_monthly, mkt_wgts_monthly, eql_wgts_monthly, signals = (
        get_dataframes(df_clean_full, sp500_members)
    )
    sy = start_date[:4]

    _, _, _, _, signals_clean = get_dataframes(
        df_clean, sp500_members
    )  # construct signals use non-extreme returns

    # Save to processed data directory
    dir_path = DATA / "processed"
    os.makedirs(dir_path, exist_ok=True)
    save_path = dir_path / f"rets_monthly_{sy}.csv"
    rets_monthly.to_csv(save_path)
    print(f"Saved monthly returns to {relative_path(save_path)}")
    save_path = dir_path / f"caps_monthly_{sy}.csv"
    caps_monthly.to_csv(save_path)
    print(f"Saved monthly market caps to {relative_path(save_path)}")
    save_path = dir_path / f"mkt_wgts_monthly_{sy}.csv"
    mkt_wgts_monthly.to_csv(save_path)
    print(f"Saved monthly market weights to {relative_path(save_path)}")
    save_path = dir_path / f"eql_wgts_monthly_{sy}.csv"
    eql_wgts_monthly.to_csv(save_path)
    print(f"Saved monthly equal weights to {relative_path(save_path)}")
    save_path = dir_path / f"signals_{sy}.csv"
    signals.to_csv(save_path)
    print(f"Saved signals to {relative_path(save_path)}")
    save_path = dir_path / f"signals_clean_{sy}.csv"
    signals_clean.to_csv(save_path)
    print(f"Saved cleaned signals to {relative_path(save_path)}")

    print("Done.")


def fetch_dspx(timeout: int = 60) -> None:
    """Downloading DSPX data from DSPX_URL and saving to disk."""

    req = urllib.request.Request(DSPX_URL)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")

    except urllib.error.URLError as err:
        sys.exit(f"Failed to download data from {DSPX_URL}: {err}")

    dspx_df = pd.read_csv(io.StringIO(raw))
    dspx_df.columns = ["Date", "DSPX"]
    dspx_df["Date"] = pd.to_datetime(dspx_df["Date"], format="%m/%d/%Y")
    dspx_df.set_index("Date", inplace=True)
    dspx_df.sort_index(inplace=True)
    START = dspx_df.index[0].year

    os.makedirs(DATA / "external", exist_ok=True)
    save_path = DATA / "external" / f"DSPX_{START}.csv"
    dspx_df.to_csv(save_path)
    print(f"Saved DSPX data to {relative_path(save_path)}")


def fetch_fama_french(timeout: int = 60) -> None:
    """Download Fama-French data from the given URL and return it as a tidy DataFrame."""

    print(f"Downloading Fama-French data from {FAMA_FRENCH_URL}...")

    req = urllib.request.Request(FAMA_FRENCH_URL)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            zip = zipfile.ZipFile(io.BytesIO(resp.read()))

    except urllib.error.URLError as err:
        sys.exit(f"Failed to download data from {FAMA_FRENCH_URL}: {err}")

    lines = zip.read(zip.namelist()[0]).decode("latin-1").splitlines()

    start = next(i for i, l in enumerate(lines) if l.strip().startswith(",Mkt"))
    end = next(i for i, l in enumerate(lines) if l.strip().startswith("Annual"))

    ff5_df = pd.read_csv(io.StringIO("\n".join(lines[start:end])), index_col=0)

    ff5_df.index = pd.to_datetime(ff5_df.index, format="%Y%m")
    ff5_df.index = ff5_df.index + pd.offsets.MonthEnd(0)  # convert to end of month
    ff5_df = ff5_df / 100.0
    ff5_df = ff5_df.loc[ff5_df.index >= START_DATE]

    os.makedirs(DATA / "external", exist_ok=True)
    save_path = DATA / "external" / f"FF5_{ff5_df.index[0].year}.csv"
    ff5_df.to_csv(save_path)
    print(f"Saved Fama-French data to {relative_path(save_path)}")


def main() -> None:
    """Download, clean and save the CRSP, CBOE and Fama-French datasets used throughout the paper."""
    df_raw = download_sp500_raw(START_DATE, END_DATE)

    sp500_members = download_sp500_membership(START_DATE)

    df_clean = clean_raw_data(data_raw=df_raw)

    format_data(
        df_clean_full=df_clean, sp500_members=sp500_members, start_date=START_DATE
    )

    fetch_dspx()

    fetch_fama_french()


if __name__ == "__main__":
    main()
