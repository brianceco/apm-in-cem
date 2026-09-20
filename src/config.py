"""Project configuration loaded from configs/config.json.

The JSON holds `days_per_year` and `months_per_year` as integers rather than decimal
time steps, so DT_DAILY and DT_MONTHLY are the exact reciprocals the code has always
used. Keep it that way: simulate.sample derives its days-per-step from
int(dt / DT_DAILY), which a truncated decimal would silently floor to 20.
"""

import json

from paths import CONFIGS


def load_config(name: str = "config.json") -> dict:
    """Load a JSON config file from configs/."""
    with open(CONFIGS / name) as f:
        return json.load(f)


CONFIG = load_config()

DT_DAILY = 1 / CONFIG["days_per_year"]
DT_MONTHLY = 1 / CONFIG["months_per_year"]
