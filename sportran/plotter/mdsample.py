# -*- coding: utf-8 -*-

from typing import Any

from . import plotter
from .plotter import plot_resample as _plot_resample
from .plotter import plot_trajectory as _plot_trajectory

__all__ = ["MDSamplePlotter"]


class MDSamplePlotter(plotter.Plotter):
    plot_resample = _plot_resample
    plot_trajectory = _plot_trajectory

    _plot_style = "api_style.mplstyle"

    def plot_periodogram(
        current: Any,
        PSD_FILTER_W: int | float | None = None,
        *,
        freq_units: str = "THz",
        freq_scale: float = 1.0,
        axes: Any = None,
        FIGSIZE: tuple[float, float] | None = None,
        mode: str = "log",
        **plot_kwargs: Any,
    ) -> Any:
        """
        Plot an ``MDSample`` periodogram.

        Parameters match :func:`sportran.plotter.plotter.plot_periodogram`, except that
        ``kappa_units`` is always disabled for ``MDSample``.
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
