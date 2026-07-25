# -*- coding: utf-8 -*-
"""
Defines the CLIPlotter class.
"""

__all__ = ["CLIPlotter"]

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
    plot_fstar_analysis as _plot_fstar_analysis,
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
    plot_psd as _plot_psd,
)
from .plotter import (
    plot_resample as _plot_resample,
)


class CLIPlotter(plotter.Plotter):
    """
    A Plotter subclass containing the plot functions used by the command-line interface.
    """

    plot_cepstral_spectrum = _plot_cepstral_spectrum
    plot_ck = _plot_ck
    plot_cospectrum_component = _plot_cospectrum_component
    plot_fstar_analysis = _plot_fstar_analysis
    plot_kappa_Pstar = _plot_kappa_Pstar
    plot_L0_Pstar = _plot_L0_Pstar
    plot_periodogram = _plot_periodogram
    plot_psd = _plot_psd
    plot_resample = _plot_resample

    _plot_style = "cli_style.mplstyle"  # TODO define style for CLI
    pass


# probably we should decorate all these functions with addPlotToPdf and other decorators
# that allow any changes of style needed
