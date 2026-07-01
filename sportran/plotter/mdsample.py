# -*- coding: utf-8 -*-
"""
Defines the MDSamplePlotter class.
"""

__all__ = ["MDSamplePlotter"]

from . import plotter


class MDSamplePlotter(plotter.Plotter):
    """
    A Plotter subclass containing the plot functions used by an object of type MDSample.
    """

    from .plotter import plot_resample, plot_trajectory

    _plot_style = "api_style.mplstyle"

    def plot_periodogram(
        current,
        PSD_FILTER_W=None,
        *,
        freq_units="THz",
        freq_scale=1.0,
        axes=None,
        FIGSIZE=None,
        mode="log",
        **plot_kwargs,
    ):
        """
        Plot an ``MDSample`` periodogram.

        Parameters match :func:`sportran.plotter.plotter.plot_periodogram`,
        except that ``kappa_units`` is always disabled for ``MDSample``.
        """
        # kappa_units is not supported by MDSample
        from .plotter import plot_periodogram

        plot_kwargs.pop("kappa_units")
        return plot_periodogram(
            current,
            PSD_FILTER_W=PSD_FILTER_W,
            freq_units=freq_units,
            freq_scale=freq_scale,
            axes=axes,
            kappa_units=False,
            FIGSIZE=FIGSIZE,
            mode=mode,
            **plot_kwargs,
        )

    pass
