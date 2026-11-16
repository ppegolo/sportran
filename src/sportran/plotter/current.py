# -*- coding: utf-8 -*-
"""
Defines the CurrentPlotter class.
"""

__all__ = ["CurrentPlotter"]

from . import plotter
from .plotter import (
    plot_cepstral_spectrum as _plot_cepstral_spectrum,
)
from .plotter import (
    plot_ck as _plot_ck,
)
from .plotter import (
    plot_cospectrum_component as _plot_cospectrum_component,
)
from .plotter import (
    plot_kappa_Pstar as _plot_kappa_Pstar,
)
from .plotter import (
    plot_L0_Pstar as _plot_L0_Pstar,
)
from .plotter import (
    plot_periodogram as _plot_periodogram,
)
from .plotter import (
    plot_resample as _plot_resample,
)
from .plotter import (
    plot_trajectory as _plot_trajectory,
)


class CurrentPlotter(plotter.Plotter):
    """
    A Plotter subclass containing the plot functions used by an object of type Current.
    """

    plot_cepstral_spectrum = _plot_cepstral_spectrum
    plot_ck = _plot_ck
    plot_cospectrum_component = _plot_cospectrum_component
    plot_kappa_Pstar = _plot_kappa_Pstar
    plot_L0_Pstar = _plot_L0_Pstar
    plot_periodogram = _plot_periodogram
    plot_resample = _plot_resample
    plot_trajectory = _plot_trajectory

    _plot_style = "api_style.mplstyle"
    pass


# # alternative method:
# for funcname in ( 'plot_periodogram', 'plot_ck', 'plot_L0_Pstar', 'plot_kappa_Pstar',
#        'plot_cepstral_spectrum', 'plot_resample'):
#
#    # get the function from the plotter module, and make it an attribute of CurrentPlotter
#    setattr(CurrentPlotter, funcname, getattr(plotter, funcname))
