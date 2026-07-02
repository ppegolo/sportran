# -*- coding: utf-8 -*-
"""
==========================
Sportran library
==========================

This is the core library containing all the code necessary to perform a nice and fast
cepstral analysis on your time series.
"""

from importlib.metadata import version as _pkg_version

from . import current, i_o, md, plotter, utils
from .current import *

__all__ = current.__all__ + md.__all__

__license__ = "GPL-3.0 license, see LICENSE.txt file."
try:
    __version__ = _pkg_version("sportran")
except Exception:  # pragma: no cover
    __version__ = "1.0.0rc4"
__authors__ = "Loris Ercole, Riccardo Bertossa, Sebastiano Bisacchi"
__paper__ = (
    "L. Ercole, R. Bertossa, S. Bisacchi, S. Baroni, "
    '"SporTran: a code to estimate transport coefficients from the cepstral analysis of '
    '(multivariate) current time series", arXiv:2202.11571 (2022), '
    "https://doi.org/10.48550/arXiv.2202.11571"
)
__paper_short__ = "L. Ercole et al., arXiv:2202.117571 (2022)"
