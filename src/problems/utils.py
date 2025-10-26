from functools import lru_cache, wraps
from typing import Callable, NamedTuple, ParamSpec, TypeVar, Any

P = ParamSpec("P")
R = TypeVar("R")


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
