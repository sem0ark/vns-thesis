import json
import re
from pathlib import Path

import numpy as np


def parse_time_string(time_str: str) -> int:
    """Parses a time string like '5s', '2m', '1h' into seconds."""
    if not time_str:
        raise ValueError(
            "Invalid time format. Use digits followed by 's' (seconds), 'm' (minutes), or 'h' (hours)."
        )

    match = re.match(r"(\d+)([smh])", time_str)
    if not match:
        raise ValueError(
            "Invalid time format. Use digits followed by 's' (seconds), 'm' (minutes), or 'h' (hours). "
            "Example: '30s', '5m', '1h'."
        )
    value = int(match.group(1))
    unit = match.group(2)
    return {
        "s": 1,
        "m": 60,
        "h": 3600,
    }[unit] * value


class NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (Path)):
            return None
        if isinstance(o, (np.bool_, np.bool)):
            return bool(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super(NpEncoder, self).default(o)
