# -*- coding: utf-8 -*-

from typing import Any


class AttributeDict(dict):
    """
    This class internally stores values in a dictionary, but exposes the keys also as
    attributes, i.e. asking for attrdict.key will return the value of attrdict['key']
    and so on.

    Raises an AttributeError if the key does not exist, when called as an attribute,
    while the usual KeyError if the key does not exist and the dictionary syntax is
    used.
    """

    def __init__(self, dictionary: dict | None = None) -> None:
        super().__init__()
        if dictionary is None:
            dictionary = {}

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self[key] = AttributeDict(value)
            else:
                self[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({dict.__repr__(self)})"

    def __getattr__(self, attr: str) -> Any:
        try:
            return self[attr]
        except KeyError:
            errmsg = f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            raise AttributeError(errmsg)

    def __setattr__(self, attr: str, value: Any) -> None:
        try:
            self[attr] = value
        except KeyError:
            raise AttributeError(
                f"AttributeError: '{attr}' is not a valid attribute of the object '{self.__class__.__name__}'"
            )

    def __delattr__(self, attr: str) -> None:
        try:
            del self[attr]
        except KeyError:
            errmsg = f"'{self.__class__.__name__}' object has no attribute '{attr}'"
            raise AttributeError(errmsg)

    def __deepcopy__(self, memo: dict | None = None) -> "AttributeDict":
        from copy import deepcopy

        if memo is None:
            memo = {}
        retval = deepcopy(dict(self))
        return self.__class__(retval)

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__.copy()

    def __setstate__(self, dictionary: dict[str, Any]) -> None:
        self.__dict__.update(dictionary)

    def __dir__(self) -> list[str]:
        return list(self.keys())
