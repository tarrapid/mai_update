import functools


def sgn(x: float) -> int:
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0


def cache(func):
    """Кэш предыдущих вызовов функций"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key not in wrapper.cache:
            wrapper.cache[cache_key] = func(*args, **kwargs)
        return wrapper.cache[cache_key]
    wrapper.cache = dict()
    return wrapper
