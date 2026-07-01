# -*- coding: utf-8 -*-
"""
Defines the CurrentPlotter class.
"""

__all__ = ["CurrentPlotter"]

from . import plotter


class CurrentPlotter(plotter.Plotter):
    """
    A Plotter subclass containing the plot functions used by an object of type Current.
    """

    from .plotter import (
        plot_cepstral_spectrum,
        plot_ck,
        plot_cospectrum_component,
        plot_kappa_Pstar,
        plot_L0_Pstar,
        plot_periodogram,
        plot_resample,
        plot_trajectory,
    )

    _plot_style = "api_style.mplstyle"
    pass


# # alternative method:
# for funcname in (
#        'plot_periodogram',
#        'plot_ck',
#        'plot_L0_Pstar',
#        'plot_kappa_Pstar',
#        'plot_cepstral_spectrum',
#        'plot_resample'):
#
#    # get the function from the plotter module, and make it an attribute of CurrentPlotter
#    setattr(CurrentPlotter, funcname, getattr(plotter, funcname))
