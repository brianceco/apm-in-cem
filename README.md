# Active Portfolio Management in Concentrated Equity Markets

Official implementation of experiments from the paper, [Citation] ((link)). Inspired by Stochastic Portfolio Theory (SPT), we study the two major drivers of performance for the equal-weighted portfolio relative to the market (capitalization-weighted) portfolio, *market diversity* and *(equal-weighted) dispersion*. We develop and calibrate a continuous-time stochastic model of diversity and dispersion. We backtest a Markowitz-style optimal portfolio under this model and show that, under both mean-reverting and trending modelling assumptions, we outperform the market and the equal-weighted portfolio between 1995-2024. All experiments are carried out within the S&P 500 universe.

## Organization

| Path | Description |
| --- | --- |
| `src/` | Core Python modules: the CRSP/CBOE/Fama-French data pipeline (`data.py`), the backtesting engine (`backtest.py`), the SDD model and simulation (`simulate.py`), plotting utilities (`plotting.py`), helper functions (`utils.py`), and project configuration and paths (`config.py`, `paths.py`). |
| `notebooks/` | Jupyter notebooks reproducing experiments within the paper. |
| `tests/` | Pytest unit tests for the data pipeline, backtesting engine, and utility functions. Must run python src/data.py to download and process data before running tests/test.data.py and tests/test_backtest.py. |
| `configs/` | `config.json`, holding tunable model and backtest parameters loaded by `src/config.py`. |

The contents of the three Jupyter notebooks in `notebooks/` are as follows:
1. `1_diversity_and_dispersion.ipynb`: Plots market diversity and equal-weighted realized dispersion at monthly frequencies, and calibrates parameter $c_\delta$.
2. `2_stylized_facts.ipynb`: Calibrates mean-reverting SDD model in sample (1977-2004) via MLE and assesses reproducibility of several stylized facts out of sample (2005-2024).
3. `3_backtest.ipynb`: Backtests optimal portfolio under mean-reverting and trending SDD models. Uses 1977-1994/1995-2024 as in-sample/out-of-sample split. Carries out sensitivity analysis of trading cost penalty parameters, and Fama-French 3-factor regressions. Use calibrated trading cost parameters from mean-reverting model to simulate performance of optimal lambda-tilt portfolio under simulated SDD model. 

**Warnings:** 
1. The notebooks **must** be run in the order in which they are numbered as they have (partial) logical dependencies on each other.**

2. Rendering notebook figures **requires** $\LaTeX$ installation on local machine by default. Set `usetex=False` when calling `use_paper_style` at top of notebooks to turn off TeX formatting.


## Installation
Run the following command in your terminal:
```
conda env create -f environment.yml      # or: pip install -r requirements.txt
```
Then activate the virtual environment via
```
conda activate apm-in-cem
```

## Quickstart
After installation, download and process relevant data by calling data module:
```
python src/data.py
```
You will be prompted to enter your WRDS username and password to download CRSP return and market cap data. Data is preprocessed following notebook 6 in [CRSP_on_WRDS_introduction](https://github.com/johruf/CRSP_on_WRDS_introduction). Publicly available S&P 500 implied dispersion index (DSPX) will be downloaded from [https://www.cboe.com](CBOE) and Fama-French monthly five-factor data will be downloaded from Ken French's [data library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
