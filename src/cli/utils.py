import json
import re
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Callable, NamedTuple, ParamSpec, TypeVar

import numpy as np

P = ParamSpec("P")
R = TypeVar("R")


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


class CacheKeyWrapper(NamedTuple):
    key: Any

    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CacheKeyWrapper):
            return self.key == other.key
        return NotImplemented


def lru_cache_custom_hash_key(
    maxsize: int = 128, typed: bool = False, key_func: Callable[..., Any] | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    if key_func is None:
        return lru_cache(maxsize=maxsize, typed=typed)  # type: ignore

    def decorator(user_function: Callable[P, R]) -> Callable[P, R]:
        @lru_cache(maxsize=maxsize, typed=typed)
        def _wrapper(key_wrapper: CacheKeyWrapper) -> R:
            kwargs_dict = dict(key_wrapper.kwargs)
            return user_function(*key_wrapper.args, **kwargs_dict)

        @wraps(user_function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            custom_key = key_func(*args, **kwargs)
            key_wrapper = CacheKeyWrapper(key=custom_key, args=args, kwargs=kwargs)
            return _wrapper(key_wrapper)

        wrapper.cache_info = _wrapper.cache_info  # type: ignore
        wrapper.cache_clear = _wrapper.cache_clear  # type: ignore

        return wrapper

    return decorator
