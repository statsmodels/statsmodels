from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload

_T = TypeVar("_T")

__all__ = [
    "ResettableCache",
    "cache_readonly",
    "cache_writable",
    "cached_data",
    "cached_value",
    "deprecated_alias",
]

# Typed as property so that the type of the cached attribute is the type that
# the decorated method returns
cache_readonly = property
cached_data = property
cached_value = property

class ResettableCache(dict[str, Any]):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

def deprecated_alias(
    old_name: str,
    new_name: str,
    remove_version: str | None = None,
    msg: str | None = None,
    warning: type[Warning] = ...,
) -> property: ...

class CachedWritableAttribute(Generic[_T]):
    @overload
    def __get__(self, obj: None, type: type | None = None) -> Callable[[Any], _T]: ...
    @overload
    def __get__(self, obj: object, type: type | None = None) -> _T: ...
    def __set__(self, obj: object, value: Any) -> None: ...

class cache_writable:
    def __init__(self, cachename: str | None = None) -> None: ...
    def __call__(self, func: Callable[[Any], _T]) -> CachedWritableAttribute[_T]: ...
