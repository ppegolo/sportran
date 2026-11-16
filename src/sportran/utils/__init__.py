# -*- coding: utf-8 -*-
"""
Various utilities like for example a global logger instance
"""

from .logger import PrintMethod

try:
    log = PrintMethod()
except Exception:
    raise RuntimeError("PrintMethod not defined.")

__all__ = ["log"]
