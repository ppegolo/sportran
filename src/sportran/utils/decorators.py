# -*- coding: utf-8 -*-

from functools import wraps
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def add_method(cls: type) -> Callable[[F], F]:
    """A decorator to dynamically add a method to a class."""

    def decorator(func: F) -> F:

        @wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


## Example:
## let A be a class
##
## Non-decorator way (note the function must accept self)
## def foo(self):
##     print('hello world!')
## setattr(A, 'foo', foo)
#
## def bar(self, s):
##     print(f'Message: {s}')
## setattr(A, 'bar', bar)
#
## Decorator can be written to take normal functions and make them methods
# @add_method(A) def foo(): print('hello world!')
#
# @add_method(A) def bar(s): print(f'Message: {s}')
