import re


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
