# -*- coding: utf-8 -*-
"""
Defines an (abstract) Plotter class and all the plot functions that its subclasses can
import.
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import MultipleLocator

from . import plt

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
iter_colors = iter(colors)


class Plotter:
    _plot_style: str | None = None


def _n_tick_in_range(beg: float, end: float, n: float) -> tuple[float, float]:
    size = end - beg
    n_cifre = math.floor(math.log(size / n, 10.0))
    delta = math.ceil((size / n) / 10**n_cifre) * 10**n_cifre
    return delta, delta / 2


def _index_cumsum(arr: np.ndarray, p: float) -> int:
    if p > 1 or p < 0:
        raise ValueError("p must be between 0 and 1")
    arr_int: np.ndarray = np.cumsum(arr)
    arr_int = arr_int / arr_int[-1]
    idx = 0
    while arr_int[idx] < p:
        idx = idx + 1
    return idx


def addPlotToPdf(func: Callable[..., Any], pdf: Any, *args: Any, **kwargs: Any) -> Any:
    """Render plot function, save current figure into PDF, and close it."""
    result = func(*args, **kwargs)
    pdf.savefig()
    plt.close()
    return result


def plot_trajectory(
    x: Any,
    *,
    axis: Axes | None = None,
    FIGSIZE: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> Any:
    if x.traj is None:
        raise ValueError("Trajectory not defined.")
    if axis is None:
        figure, axis = plt.subplots(1, figsize=FIGSIZE)
    axis.plot(x.traj, **plot_kwargs)
    axis.set_xlabel(r"$t$ [ps]")
    axis.grid()
    return axis


def plot_periodogram(
    current: Any,
    PSD_FILTER_W: int | float | None = None,
    *,
    freq_units: str = "THz",
    freq_scale: float = 1.0,
    axes: Any = None,
    kappa_units: bool = True,
    FIGSIZE: tuple[float, float] | None = None,
    mode: str = "log",
    **plot_kwargs: Any,
) -> Any:
    if current.psd is None:
        current.compute_psd()
    # (re)compute filtered psd, if a window has been defined
    if (PSD_FILTER_W is not None) or (current.PSD_FILTER_W is not None):
        current.filter_psd(PSD_FILTER_W, freq_units)
    else:  # use a zero-width (non-filtering) window
        current.filter_psd(0.0)
    if kappa_units:  # plot psd in units of kappa - the log(psd) is not converted
        psd_scale = 0.5 * current.KAPPA_SCALE
    else:
        psd_scale = 1.0

    if axes is None:
        figure, axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)
        plt.subplots_adjust(hspace=0.1)
    if freq_units in ("THz", "thz"):
        axes[0].plot(current.freqs_THz, psd_scale * current.fpsd, **plot_kwargs)
        axes[0].set_xlim([0.0, current.Nyquist_f_THz])
        if mode == "log":
            axes[1].plot(current.freqs_THz, current.flogpsd, **plot_kwargs)
            axes[1].set_xlim([0.0, current.Nyquist_f_THz])
            axes[1].set_xlabel(r"$f$ [THz]")
    elif freq_units == "red":
        axes[0].plot(
            current.freqs / freq_scale, psd_scale * current.fpsd, **plot_kwargs
        )
        axes[0].set_xlim([0.0, 0.5 / freq_scale])
        if mode == "log":
            axes[1].plot(current.freqs / freq_scale, current.flogpsd, **plot_kwargs)
            axes[1].set_xlim([0.0, 0.5 / freq_scale])
            axes[1].set_xlabel(r"$f$ [$\omega$*DT/2$\pi$]")
    else:
        raise ValueError("Frequency units not valid.")
    axes[0].xaxis.set_ticks_position("top")
    if kappa_units:
        axes[0].set_ylabel(r"PSD [{}]".format(current._KAPPA_SI_UNITS))
    else:
        axes[0].set_ylabel(r"PSD")
    axes[0].grid()
    if mode == "log":
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].set_ylabel(r"log(PSD)")
        axes[1].grid()
    return axes


def plot_cospectrum_component(
    current: Any,
    idx1: int,
    idx2: int,
    *,
    axis: Axes | None = None,
    FIGSIZE: tuple[float, float] | None = None,
    f_THz_max: float | None = None,
    k_SI_max: float | None = None,
    k_SI_min: float | None = None,
    k_tick: float | None = None,
    f_tick: float | None = None,
) -> Any:
    if axis is None:
        figure, axis = plt.subplots(1, figsize=FIGSIZE)
    color1 = next(iter_colors)
    color2 = next(iter_colors)
    axis.plot(
        current.freqs_THz,
        np.real(current.fcospectrum[idx1][idx2]) * current.KAPPA_SCALE * 0.5,
        c=color1,
    )
    axis.plot(
        current.freqs_THz,
        np.imag(current.fcospectrum[idx1][idx2]) * current.KAPPA_SCALE * 0.5,
        c=color2,
    )

    if f_THz_max is None:
        f_THz_max = current.freqs_THz[
            _index_cumsum(np.abs(current.fcospectrum[idx1][idx2]), 0.95)
        ]
    else:
        f_THz_max = min(f_THz_max, current.freqs_THz[-1])
    axis.set_xlim((0, f_THz_max))
    if k_SI_max is None:
        k_SI_max = (
            np.max(
                np.abs(current.fcospectrum[idx1][idx2])[
                    : int(current.NFREQS * f_THz_max / current.freqs_THz[-1])
                ]
                * current.KAPPA_SCALE
                * 0.5
            )
            * 1.3
        )
    if k_SI_min is None:
        k_SI_min = -k_SI_max
    axis.set_ylim((k_SI_min, k_SI_max))
    axis.set_xlabel(r"$\omega/2\pi$ (THz)")
    axis.set_ylabel(r"$S^{{{}{}}}$".format(idx1, idx2))

    if f_tick is None:
        dx1, dx2 = _n_tick_in_range(0, f_THz_max, 5)
    else:
        dx1, dx2 = (f_tick, f_tick / 2)
    if k_tick is None:
        dy1, dy2 = _n_tick_in_range(0, k_SI_max, 5)
    else:
        dy1, dy2 = (k_tick, k_tick / 2)

    axis.xaxis.set_major_locator(MultipleLocator(dx1))
    axis.xaxis.set_minor_locator(MultipleLocator(dx2))
    axis.yaxis.set_major_locator(MultipleLocator(dy1))
    axis.yaxis.set_minor_locator(MultipleLocator(dy2))


def plot_ck(
    current: Any,
    *,
    axis: Axes | None = None,
    label: str | None = None,
    FIGSIZE: tuple[float, float] | None = None,
) -> Any:

    if axis is None:
        figure, axis = plt.subplots(1, figsize=FIGSIZE)
    color = next(iter_colors)
    axis.plot(current.cepf.logpsdK, "o-", c=color, label=label)

    axis.plot(current.cepf.logpsdK + current.cepf.logpsdK_THEORY_std, "--", c=color)
    axis.plot(current.cepf.logpsdK - current.cepf.logpsdK_THEORY_std, "--", c=color)
    axis.axvline(x=current.cepf.aic_Kmin, ls=":", c=color)
    axis.axvline(x=current.cepf.cutoffK, ls="--", c=color)
    axis.set_xlabel(r"$k$")
    axis.set_ylabel(r"$c_k$")
    return axis


def plot_L0_Pstar(
    current: Any,
    *,
    axis: Axes | None = None,
    label: str | None = None,
    FIGSIZE: tuple[float, float] | None = None,
) -> Any:
    if axis is None:
        figure, axis = plt.subplots(1, figsize=FIGSIZE)
    color = next(iter_colors)  # quick fix to avoid error with mlp>=3.8
    axis.plot(
        np.arange(current.NFREQS) + 1, current.cepf.logtau, ".-", c=color, label=label
    )
    axis.plot(
        np.arange(current.NFREQS) + 1,
        current.cepf.logtau + current.cepf.logtau_THEORY_std,
        "--",
        c=color,
    )
    axis.plot(
        np.arange(current.NFREQS) + 1,
        current.cepf.logtau - current.cepf.logtau_THEORY_std,
        "--",
        c=color,
    )
    axis.axvline(x=current.cepf.aic_Kmin + 1, ls=":", c=color)
    axis.axvline(x=current.cepf.cutoffK + 1, ls="--", c=color)
    axis.set_xlim((0, 3 * current.cepf.cutoffK))
    max_y = np.amax(
        (current.cepf.logtau + current.cepf.logtau_THEORY_std)[
            current.cepf.cutoffK : 3 * current.cepf.cutoffK
        ]
    )
    min_y = np.amin(
        (current.cepf.logtau - current.cepf.logtau_THEORY_std)[
            current.cepf.cutoffK : 3 * current.cepf.cutoffK
        ]
    )
    axis.set_ylim((min_y * 0.8, max_y * 1.2))
    axis.set_xlabel(r"$P^*$")
    axis.set_ylabel(r"$L_0(P*)$")
    return axis


def plot_kappa_Pstar(
    current: Any,
    *,
    axis: Axes | None = None,
    label: str | None = None,
    FIGSIZE: tuple[float, float] | None = None,
    pstar_max: int | None = None,
    kappa_SI_min: float | None = None,
    kappa_SI_max: float | None = None,
    pstar_tick: float | None = None,
    kappa_tick: float | None = None,
) -> Any:
    if axis is None:
        figure, axis = plt.subplots(1, figsize=FIGSIZE)
    color = next(iter_colors)
    axis.fill_between(
        np.arange(current.NFREQS) + 1,
        (current.cepf.tau - current.cepf.tau_THEORY_std) * current.KAPPA_SCALE * 0.5,
        (current.cepf.tau + current.cepf.tau_THEORY_std) * current.KAPPA_SCALE * 0.5,
        alpha=0.3,
        color=color,
    )
    axis.plot(
        np.arange(current.NFREQS) + 1,
        current.cepf.tau * current.KAPPA_SCALE * 0.5,
        "o-",
        c=color,
        label=label,
    )
    axis.axvline(x=current.cepf.aic_Kmin + 1, ls=":", c=color)
    axis.axvline(x=current.cepf.cutoffK + 1, ls="--", c=color)
    axis.axhline(y=current.kappa, ls="--", c=color)
    if pstar_max is None:
        pstar_max = int(round((current.cepf.cutoffK + 1) * 2.5))
    axis.set_xlim((0, pstar_max))
    if kappa_SI_max is None:
        kappa_SI_max = 1.2 * np.amax(
            current.KAPPA_SCALE
            * 0.5
            * (current.cepf.tau + current.cepf.tau_THEORY_std)[
                current.cepf.cutoffK : pstar_max
            ]
        )
    if kappa_SI_min is None:
        kappa_SI_min = 0.8 * np.amin(
            current.KAPPA_SCALE
            * 0.5
            * (current.cepf.tau - current.cepf.tau_THEORY_std)[
                current.cepf.cutoffK : pstar_max
            ]
        )
    axis.set_ylim((kappa_SI_min, kappa_SI_max))
    axis.set_xlabel(r"$P^*$")
    axis.set_ylabel(r"$\kappa(P^*)$ [{}]".format(current._KAPPA_SI_UNITS))
    if pstar_tick is None:
        dx1, dx2 = _n_tick_in_range(0, pstar_max, 5)
    else:
        dx1, dx2 = (pstar_tick, pstar_tick / 2)
    if kappa_tick is None:
        dy1, dy2 = _n_tick_in_range(0, kappa_SI_max, 5)
    else:
        dy1, dy2 = (kappa_tick, kappa_tick / 2)
    axis.xaxis.set_major_locator(MultipleLocator(dx1))
    axis.xaxis.set_minor_locator(MultipleLocator(dx2))
    axis.yaxis.set_major_locator(MultipleLocator(dy1))
    axis.yaxis.set_minor_locator(MultipleLocator(dy2))
    return axis


def plot_cepstral_spectrum(
    current: Any,
    *,
    freq_units: str = "THz",
    freq_scale: float = 1.0,
    axes: Any = None,
    kappa_units: bool = True,
    FIGSIZE: tuple[float, float] | None = None,
    mode: str = "log",
    **plot_kwargs: Any,
) -> Any:
    if axes is None:
        figure, axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)
    plt.subplots_adjust(hspace=0.1)
    if kappa_units:
        psd_scale = 0.5 * current.KAPPA_SCALE
    else:
        psd_scale = 1.0
    if freq_units in ("THz", "thz"):
        axes[0].plot(current.freqs_THz, current.cepf.psd * psd_scale, **plot_kwargs)
        axes[0].set_xlim([0.0, current.Nyquist_f_THz])
        if mode == "log":
            axes[1].plot(current.freqs_THz, current.cepf.logpsd, **plot_kwargs)
            axes[1].set_xlim([0.0, current.Nyquist_f_THz])
            axes[1].set_xlabel(r"$f$ [THz]")
    elif freq_units == "red":
        axes[0].plot(
            current.freqs / freq_scale, current.cepf.psd * psd_scale, **plot_kwargs
        )
        axes[0].set_xlim([0.0, 0.5 / freq_scale])
        if mode == "log":
            axes[1].plot(current.freqs / freq_scale, current.cepf.logpsd, **plot_kwargs)
            axes[1].set_xlim([0.0, 0.5 / freq_scale])
            axes[1].set_xlabel(r"$f$ [$\omega$*DT/2$\pi$]")
    else:
        raise ValueError("Units not valid.")
    axes[0].xaxis.set_ticks_position("top")
    axes[0].set_ylabel(r"PSD")
    if kappa_units:
        axes[0].set_ylabel(r"PSD [{}]".format(current._KAPPA_SI_UNITS))
    else:
        axes[0].set_ylabel(r"PSD")
    axes[0].grid()
    if mode == "log":
        axes[1].xaxis.set_ticks_position("bottom")
        axes[1].set_ylabel(r"log(PSD)")
        axes[1].grid()
    return axes


def plot_fstar_analysis(
    currents: Any,
    FSTAR_THZ_LIST: list[float] | np.ndarray,
    original_current: Any = None,
    *,
    axes: Any = None,
    FIGSIZE: tuple[float, float] | None = None,
    **plot_kwargs: Any,
) -> Any:
    if axes is None:
        figure, axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)
        return_axes = True
    else:
        return_axes = False
    axes[0].errorbar(
        FSTAR_THZ_LIST,
        [xff.kappa for xff in currents],
        yerr=[xff.kappa_std for xff in currents],
        zorder=-1,
        **plot_kwargs,
    )
    axes[1].errorbar(
        FSTAR_THZ_LIST,
        [xff.cepf.logtau_cutoffK for xff in currents],
        yerr=[xff.cepf.logtau_std_cutoffK for xff in currents],
        zorder=-1,
        **plot_kwargs,
    )
    axes[0].xaxis.set_ticks_position("top")
    axes[0].set_ylabel(r"PSD")
    axes[0].grid()
    axes[1].xaxis.set_ticks_position("bottom")
    axes[1].set_xlabel(r"$f$ [THz]")
    axes[1].set_ylabel(r"log(PSD)")
    axes[1].grid()
    if original_current is not None:
        ax2 = [axes[0].twinx(), axes[1].twinx()]
        plot_periodogram(original_current, axes=ax2, c="0.6")
        axes[0].set_ylabel(r"$\kappa$ [{}]".format(original_current._KAPPA_SI_UNITS))
        axes[1].set_ylabel(r"$\kappa$ [{}]".format(original_current._KAPPA_SI_UNITS))
        axes[0].set_zorder(ax2[0].get_zorder() + 1)
        axes[1].set_zorder(ax2[1].get_zorder() + 1)
        axes[0].set_frame_on(False)
        axes[1].set_frame_on(False)
    if return_axes:
        return currents, axes, figure
    else:
        return currents, axes


def plot_resample(
    x: Any,
    xf: Any,
    PSD_FILTER_W: int | float | None = None,
    *,
    freq_units: str = "THz",
    axes: Any = None,
    FIGSIZE: tuple[float, float] | None = None,
    mode: str = "log",
) -> Any:
    fstar_THz = xf.Nyquist_f_THz
    TSKIP = int(x.Nyquist_f_THz / xf.Nyquist_f_THz)

    from sportran.current import Current

    plot_kappa_units = isinstance(x, Current)
    if not axes:
        figure, axes = plt.subplots(2, sharex=True, figsize=FIGSIZE)
        axes = plot_periodogram(
            x,
            PSD_FILTER_W=PSD_FILTER_W,
            freq_units=freq_units,
            axes=axes,
            mode=mode,
            kappa_units=plot_kappa_units,
        )  # this also updates x.PSD_FILTER_W
    xf.plot_periodogram(
        freq_units=freq_units,
        freq_scale=TSKIP,
        axes=axes,
        mode=mode,
        kappa_units=plot_kappa_units,
    )
    if freq_units in ("THz", "thz"):
        axes[0].axvline(x=fstar_THz, ls="--", c="k")
        axes[0].set_xlim([0.0, x.Nyquist_f_THz])
        if mode == "log":
            axes[1].axvline(x=fstar_THz, ls="--", c="k")
            axes[1].set_xlim([0.0, x.Nyquist_f_THz])
    elif freq_units == "red":
        axes[0].axvline(x=0.5 / TSKIP, ls="--", c="k")
        axes[0].set_xlim([0.0, 0.5])
        if mode == "log":
            axes[1].axvline(x=0.5 / TSKIP, ls="--", c="k")
            axes[1].set_xlim([0.0, 0.5])
    return axes


################################################################################
## DUPLICATE FUNCTIONS THAT NEED TO BE MERGED IF POSSIBLE


def plot_psd(
    jf: Any,
    j2: Any | None = None,
    j2pl: Any | None = None,
    f_THz_max: float | None = None,
    k_SI_max: float | None = None,
    k_tick: float | None = None,
    f_tick: float | None = None,
) -> Any:
    """Plot legacy PSD view for filtered and optional comparison datasets."""
    if f_THz_max is None:
        idx_max = _index_cumsum(jf.psd, 0.95)
        f_THz_max = jf.freqs_THz[idx_max]
    else:
        maxT = jf.freqs_THz[-1]
        if j2 is not None:
            if j2.freqs_THz[-1] > maxT:
                maxT = j2.freqs_THz[-1]
        if j2pl is not None:
            if j2pl.freqs_THz[-1] > maxT:
                maxT = j2pl.freqs_THz[-1]
        if maxT < f_THz_max:
            f_THz_max = maxT

    if k_SI_max is None:
        k_SI_max = (
            np.max(
                jf.fpsd[: int(jf.freqs_THz.shape[0] * f_THz_max / jf.freqs_THz[-1])]
                * jf.KAPPA_SCALE
                * 0.5
            )
            * 1.3
        )

    figure, ax = plt.subplots(1, 1)  # figsize=(3.8, 2.3)
    ax.plot(jf.freqs_THz, jf.psd * jf.KAPPA_SCALE * 0.5, lw=0.2, c="0.8", zorder=0)
    ax.plot(jf.freqs_THz, jf.fpsd * jf.KAPPA_SCALE * 0.5, c=colors[0], zorder=2)
    if j2 is not None:
        plt.axvline(x=j2.Nyquist_f_THz, ls="--", c="k", dashes=(1.4, 0.6), zorder=3)
    if j2pl is not None:
        plt.plot(
            j2pl.freqs_THz,
            j2pl.cepf.psd * j2pl.KAPPA_SCALE * 0.5,
            c=colors[1],
            zorder=1,
        )
    try:
        plt.plot(
            jf.freqs_THz,
            np.real(jf.fcospectrum[0][0]) * jf.KAPPA_SCALE * 0.5,
            c=colors[3],
            lw=1.0,
            zorder=1,
        )
    except Exception:
        pass

    ax.set_ylim([0, k_SI_max])
    ax.set_xlim([0, f_THz_max])
    ax.set_xlabel(r"$\omega/2\pi$ (THz)")
    ax.set_ylabel(r"${{}}^{{\ell}}\hat{{S}}_{{\,k}}$ [{}]".format(jf._KAPPA_SI_UNITS))

    if f_tick is None:
        dx1, dx2 = _n_tick_in_range(0, f_THz_max, 5)
    else:
        dx1 = f_tick
        dx2 = dx1 / 2
    if k_tick is None:
        dy1, dy2 = _n_tick_in_range(0, k_SI_max, 5)
    else:
        dy1 = k_tick
        dy2 = dy1 / 2

    ax.xaxis.set_major_locator(MultipleLocator(dx1))
    ax.xaxis.set_minor_locator(MultipleLocator(dx2))
    ax.yaxis.set_major_locator(MultipleLocator(dy1))
    ax.yaxis.set_minor_locator(MultipleLocator(dy2))
