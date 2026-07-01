# -*- coding: utf-8 -*-

from typing import IO, Any, Callable, ClassVar, Optional


class PrintMethod:
    """
    This class is intended to manage all the messages that should be put in a file,
    in the stdout with print, or in any other place. Is has to be used as a global thing.
    It is just a routing point. In the whole library please use this.
    By default it is equivalent to a call to print()
    """

    _print_func: ClassVar[Optional[Callable[..., None]]] = (
        None  #: print function called by :func:`write_log`
    )
    _METHOD: ClassVar[list[str]] = ["bash"]  #: methods to call by :func:`write_log`
    lfile: ClassVar[Optional[IO[str]]] = None

    def __init__(self) -> None:
        pass

    @classmethod
    def open_file(cls, fname: str) -> None:
        """opens the log file that will be used globally"""
        cls.lfile = open(fname, "w")

    @classmethod
    def close_file(cls) -> None:
        if cls.lfile is not None:
            cls.lfile.close()
            cls.lfile = None

    @classmethod
    def write_log(cls, *args: Any, **kwargs: Any) -> None:
        """
        Calls all the methods added by :func:`append_method`
        """
        if "bash" in cls._METHOD:
            print(*args, **kwargs)
        if "file" in cls._METHOD:
            s = ""
            for a in args:
                s += str(a)
            if cls.lfile is not None:
                cls.lfile.write(s + "\n")
        if "other" in cls._METHOD:
            if cls._print_func:
                cls._print_func(*args, **kwargs)
            else:
                print(*args, **kwargs)

    @classmethod
    def set_func(cls, func: Callable[..., None]) -> None:
        """Set the function to call when :func:`write_log` is called.
        The function is called only if the 'other' method is setted by :func:`append_method`
        or :func:`set_method`"""
        cls._print_func = func

    @classmethod
    def append_method(cls, method: str) -> None:
        """append the method to the list.
        :param method: the method to be added to the method list, can be any of 'bash', 'file' or 'other'
        :type method: str
        if 'file' remember to call :func:`open_file` somewhere
        """
        cls._METHOD.append(method)

    @classmethod
    def set_method(cls, method: str) -> None:
        """Removes all method and set only the provided one"""
        cls._METHOD = [method]
