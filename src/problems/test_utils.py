from unittest.mock import MagicMock

from src.problems.utils import CacheKeyWrapper, lru_cache_custom_hash_key


class Unhashable:
    """A class that is deliberately unhashable."""

    def __init__(self, key: str):
        self.key = key

    def __repr__(self) -> str:
        return f"<Unhashable key={self.key}>"

    # Note: __hash__ is not defined, making it unhashable by default if __eq__ is not defined.


def custom_key_generator(unhashable_obj: Unhashable, factor: int) -> tuple[str, int]:
    """Generates a hashable key from an unhashable object."""
    return (unhashable_obj.key, factor)


def test_custom_cache_hit_on_unhashable_type():
    """Test that cache hits correctly when arguments are unhashable but the key is the same."""
    mock_func = MagicMock(side_effect=lambda data, factor: data.key + str(factor))

    @lru_cache_custom_hash_key(key_func=custom_key_generator)
    def compute_data(data: Unhashable, factor: int) -> str:
        return mock_func(data, factor)

    # 1. First call (Cache Miss)
    obj_a1 = Unhashable("A")
    result1 = compute_data(obj_a1, 10)

    assert result1 == "A10"
    assert mock_func.call_count == 1

    # 2. Second call with a DIFFERENT object instance but the SAME key (Cache Hit)
    obj_a2 = Unhashable("A")
    result2 = compute_data(obj_a2, 10)

    assert result2 == "A10"
    assert mock_func.call_count == 1  # Should NOT be called again

    # 3. Third call with a DIFFERENT key (Cache Miss)
    result3 = compute_data(obj_a1, 20)

    assert result3 == "A20"
    assert mock_func.call_count == 2

    # 4. Check cache info
    info = compute_data.cache_info()
    assert info.hits == 1
    assert info.misses == 2
    assert info.maxsize == 128
    assert info.currsize == 2


def test_custom_cache_hit_on_kwargs():
    """Test that cache hits correctly when using kwargs in the key generation."""
    mock_func = MagicMock(return_value=1)

    def kwarg_key(a, b, **kwargs):
        # Key depends only on a and the keyword 'id'
        return (a, kwargs.get("id"))

    @lru_cache_custom_hash_key(key_func=kwarg_key)
    def my_func(a, b, **kwargs):
        return mock_func(a, b, **kwargs)

    # 1. Call 1: Key (1, "X")
    my_func(1, 5, id="X", unused=100)
    assert mock_func.call_count == 1

    # 2. Call 2: Same Key (1, "X"), different positional and unused kwarg -> Cache Hit
    my_func(1, 99, id="X", unused=200)
    assert mock_func.call_count == 1

    # 3. Call 3: Different Key (1, "Y") -> Cache Miss
    my_func(1, 5, id="Y")
    assert mock_func.call_count == 2

    info = my_func.cache_info()
    assert info.hits == 1
    assert info.misses == 2


def test_custom_cache_clear():
    """Test cache_clear method."""
    mock_func = MagicMock(return_value=1)

    @lru_cache_custom_hash_key(key_func=lambda a, b: a)
    def func_to_clear(a: int, b: int) -> int:
        return mock_func(a, b)

    func_to_clear(1, 2)
    func_to_clear(1, 3)  # Cache Hit
    assert func_to_clear.cache_info().currsize == 1

    func_to_clear.cache_clear()

    assert func_to_clear.cache_info().currsize == 0

    func_to_clear(1, 2)  # Should be a Cache Miss after clearing
    assert mock_func.call_count == 2  # Called twice in total


def test_cache_key_wrapper_hash_and_eq():
    """Test the core logic of the CacheKeyWrapper class."""

    # Two wrappers with the same custom key
    wrapper1 = CacheKeyWrapper(key=("user", 42), args=(1,), kwargs={})
    wrapper2 = CacheKeyWrapper(key=("user", 42), args=(2,), kwargs={})

    # Two wrappers with different keys
    wrapper3 = CacheKeyWrapper(key=("admin", 1), args=(1,), kwargs={})

    # Test Hash (should be equal if key is equal)
    assert hash(wrapper1) == hash(wrapper2)
    assert hash(wrapper1) != hash(wrapper3)

    # Test Equality (should be equal if key is equal, regardless of args/kwargs)
    assert wrapper1 == wrapper2
    assert wrapper1 != wrapper3

    # Test against different type
    assert wrapper1 != "a string"
