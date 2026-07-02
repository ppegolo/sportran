# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
from scipy.signal import periodogram

from sportran.plotter import Plotter, use_plot_style
from sportran.plotter.mdsample import MDSamplePlotter
from sportran.utils import log
from sportran.utils.decorators import add_method

from .resample import resample_timeseries
from .tools.acf import acovf, integrate_acf
from .tools.filter import runavefilter
from .tools.spectrum import freq_red_to_THz, freq_THz_to_red

__all__ = ["MDSample"]


class MDSample(object):
    """
    Representation of a single molecular-dynamics sample.

    The object can store trajectory, spectrum, and periodogram data while keeping
    derived quantities internally consistent.

    Main attributes include ``traj``, ``spectr``, ``psd``, ``freqs``, ``freqs_THz``,
    ``DT_FS``, ``fpsd``, ``flogpsd``, and ``acf``.

    """

    _default_plotter: type[Plotter] = MDSamplePlotter

    def __init__(
        self,
        traj: np.ndarray | None = None,
        spectr: np.ndarray | None = None,
        psd: np.ndarray | None = None,
        freqs: np.ndarray | None = None,
        DT_FS: float = 1.0,
    ) -> None:
        self.DT_FS = DT_FS
        self.initialize_traj(traj)
        self.initialize_spectrum(spectr)
        self.initialize_psd(freqs=freqs, psd=psd, DT_FS=DT_FS)

    def __repr__(self) -> str:
        msg = (
            "MDSample:\n"
            + "  DT_FS:  {}  fs\n".format(self.DT_FS)
            + "  traj:   {}  steps  *  {} equivalent components\n".format(
                self.N, self.N_EQUIV_COMPONENTS
            )
            + "          {}  fs\n".format(
                None if self.traj is None else self.DT_FS * self.N
            )
        )
        if self.spectr is not None:
            msg += "  spectr: {}  frequencies\n".format(self.NFREQS)
        if self.psd is not None:
            msg += (
                "  psd:    {}  frequencies\n".format(self.psd.size)
                + "      DF =      {}  [omega*DT/(2*pi)]\n".format(self.DF)
                + "                {}  [THz]\n".format(self.DF_THZ)
                + "      Nyquist Frequency = {}  [THz]\n".format(self.Nyquist_f_THz)
            )
        if self.fpsd is not None:
            msg += (
                "  fpsd:   {}  frequencies\n".format(self.fpsd.size)
                + "      PSD_FILTER_W  = {} [omega*DT/(2*pi)]\n".format(
                    self.PSD_FILTER_W
                )
                + "                    = {} [THz]\n".format(self.PSD_FILTER_W_THZ)
                + "      PSD_FILTER_WF = {} frequencies\n".format(self.PSD_FILTER_WF)
            )
        if self.acf is not None:
            msg += "  acf:    {}  lags\n".format(self.NLAGS)
        return msg

    def _get_builder(self) -> tuple[type, dict[str, np.ndarray | float | None]]:
        kwargs: dict[str, np.ndarray | float | None] = dict(
            traj=self.traj, DT_FS=self.DT_FS
        )
        return type(self), kwargs

    @classmethod
    def set_plotter(cls, plotter: type[Plotter] | None = None) -> None:
        if plotter is None:
            plotter = cls._default_plotter
        if not (isinstance(plotter, Plotter) or issubclass(plotter, Plotter)):
            raise TypeError("Invalid plotter")

        cls._plotter = plotter  # type: ignore[attr-defined]
        use_plot_style(plotter._plot_style)

        # delete any plot function already present in this class
        for funcname in filter(
            lambda name: name.startswith("plot_"), dir(cls)
        ):  # same as [name for name in dir(plotter) if name.startswith('plot_')
            obj = getattr(cls, funcname)
            if callable(obj):
                # print('deleting {} from class {}'.format(obj, cls))
                try:
                    delattr(cls, funcname)
                except AttributeError:
                    pass

        # loop over all functions of the plotter class, and transform them into methods
        # of Current
        for funcname in filter(lambda name: name.startswith("plot_"), dir(plotter)):
            obj = getattr(plotter, funcname)
            if callable(obj):
                add_method(cls)(obj)
                # print('{} added to class {}'.format(obj, cls))

    ############################################# ##################################
    ###  INITIALIZE METHODS
    ################################### ############################################

    def initialize_traj(self, array: list | np.ndarray | tuple | None) -> None:
        if not isinstance(array, (list, np.ndarray, tuple)):
            raise TypeError("Input trajectory must be an array.")
        if array is not None:
            array = np.array(array, dtype=float)
            if isinstance(array, np.ndarray) and len(array.shape) == 1:
                self.MANY_EQUIV_COMPONENTS = False
                self.traj = array[:, np.newaxis]
            elif isinstance(array, np.ndarray) and len(array.shape) == 2:
                self.MANY_EQUIV_COMPONENTS = True
                if array.shape[0] % 2 == 1:
                    self.traj = array[:-1]
                    log.write_log(
                        "Trajectory has an odd number of points. Removing the last one."
                    )
                else:
                    self.traj = array
            else:
                raise TypeError("Input trajectory array has > 2 dimensions.")
            self.N, self.N_EQUIV_COMPONENTS = self.traj.shape
            if (self.N < 2) or (self.N_EQUIV_COMPONENTS < 1):
                raise ValueError(
                    "Input trajectory size too small (N = {}, N_EQUIV_COMPONENTS = {}).".format(
                        self.N, self.N_EQUIV_COMPONENTS
                    )
                )
        else:
            self.traj = None
            self.N = None
            self.N_EQUIV_COMPONENTS = None
        self.acf: np.ndarray | None = None
        self.NLAGS: int | None = None

    def initialize_spectrum(self, array: np.ndarray | list | None) -> None:
        if array is not None:
            self.spectr = np.array(array, dtype=complex)
            self.NFREQS = self.spectr.size
            self.DF = 0.5 / (self.NFREQS - 1)
        else:
            self.spectr = None
            self.NFREQS = None
            self.DF = None

    def initialize_psd(
        self,
        freq_psd: np.ndarray | tuple | None = None,
        psd: np.ndarray | None = None,
        freqs: np.ndarray | None = None,
        DT_FS: float | None = None,
    ) -> None:
        # frequencies
        if freq_psd is not None:  # use freq_psd variable
            if len(freq_psd) == 2:  # (freqs, psd) tuple was passed
                if (freqs is not None) or (psd is not None):
                    raise ValueError("Too many arguments.")
                frequencies = freq_psd[0]
                array = freq_psd[1]
            elif len(freq_psd) > 2:  # array used as psd or freqs
                if psd is None:  # only psd was passed
                    if freqs is not None:
                        raise ValueError("Too many arguments.")
                    frequencies = None
                    array = freq_psd
                else:  # freqs and psd passed separately
                    if freqs is not None:
                        raise ValueError("Too many arguments.")
                    frequencies = freq_psd
                    array = psd
            else:
                raise ValueError("arguments not valid")
        else:  # ignore freq_psd variable
            frequencies = freqs
            array = psd

        self.psd = None
        self.freqs = None
        self.fpsd: np.ndarray | None = None
        self.flogpsd: np.ndarray | None = None
        self.Nyquist_f_THz: float | None = None
        self.PSD_FILTER_W: float | None = None
        self.PSD_FILTER_W_THZ: float | None = None
        self.PSD_FILTER_WF: int | None = None

        # PSD
        if array is None:
            return
        self.psd = np.array(array, dtype=float)
        self.logpsd = np.log(self.psd)
        self.logpsd_min = np.min(self.psd)

        # frequencies
        self.NFREQS = self.psd.size
        if frequencies is None:  # recompute frequencies
            self.freqs = np.linspace(0.0, 0.5, self.NFREQS)
        else:
            self.freqs = np.array(frequencies, dtype=float)
            if self.freqs.size != self.NFREQS:
                raise ValueError("Number of frequencies different from PSD array size.")

        # freqs conversions to THz
        if DT_FS is not None:
            self.DT_FS = DT_FS
        self.freqs_THz: np.ndarray = freq_red_to_THz(self.freqs, self.DT_FS)
        self.Nyquist_f_THz = float(self.freqs_THz[-1])
        self.DF = 0.5 / (self.NFREQS - 1)
        self.DF_THZ = freq_red_to_THz(self.DF, self.DT_FS)

    ############################################# ##################################
    ###  COMPUTE METHODS
    ################################### ############################################

    def timeseries(self) -> np.ndarray:
        return np.arange(self.N) * self.DT_FS

    def compute_trajectory(self) -> None:
        if self.spectr is None:
            raise ValueError("Spectrum not defined.")
        full_spectr: np.ndarray = np.append(self.spectr, self.spectr[-2:0:-1].conj())
        self.traj = np.real(np.fft.ifft(full_spectr))  # *np.sqrt(self.NFREQS-1)
        self.N = self.traj.size

    def compute_spectrum(self) -> None:
        if self.traj is None:
            raise ValueError("Trajectory not defined.")
        full_spectr = np.fft.fft(self.traj)
        self.spectr = full_spectr[: self.N / 2 + 1]
        self.NFREQS = self.spectr.size
        self.DF = 0.5 / (self.NFREQS - 1)

    def compute_psd(
        self,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        method: str = "trajectory",
        DT_FS: float | None = None,
        normalize: bool = False,
    ) -> None:
        if DT_FS is not None:
            self.DT_FS = DT_FS
        if method == "trajectory":
            if self.traj is None:
                raise ValueError("Trajectory not defined.")
            self.freqs, self.psdALL = periodogram(self.traj, detrend=None, axis=0)
            self.psd = np.mean(self.psdALL, axis=1)
            self.psd[1:-1] = self.psd[1:-1] * 0.5
            self.psd *= self.DT_FS
            self.NFREQS = self.freqs.size
            self.DF = 0.5 / (self.NFREQS - 1)
            self.DF_THZ = freq_red_to_THz(self.DF, self.DT_FS)
        elif method == "spectrum":
            if self.spectr is None:
                raise ValueError("Spectrum not defined.")
            self.psd = self.DT_FS * np.abs(self.spectr) ** 2 / (2 * (self.NFREQS - 1))
            self.freqs = np.linspace(0.0, 0.5, self.NFREQS)
        else:
            raise KeyError("method not understood")

        self.freqs_THz = self.freqs / self.DT_FS * 1000.0
        self.Nyquist_f_THz = float(self.freqs_THz[-1])
        if normalize:
            self.psd = self.psd / np.trapezoid(self.psd) / self.N / self.DT_FS
        self.logpsd = np.log(self.psd)
        self.psd_min = np.min(self.psd)
        self.psd_power = np.trapezoid(self.psd)  # one-side PSD power

        # (re)compute filtered psd, if a window has been defined
        if (PSD_FILTER_W is not None) or (self.PSD_FILTER_W is not None):
            self.filter_psd(PSD_FILTER_W, freq_units)

    def filter_psd(
        self,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        window_type: str = "rectangular",
        logpsd_filter_type: int = 1,
    ) -> None:
        if self.psd is None:
            raise ValueError("Periodogram is not defined.")
        assert self.DT_FS is not None
        if PSD_FILTER_W is not None:
            if freq_units in ("THz", "thz"):
                self.PSD_FILTER_W_THZ = PSD_FILTER_W
                self.PSD_FILTER_W = freq_THz_to_red(PSD_FILTER_W, self.DT_FS)
            elif freq_units == "red":
                self.PSD_FILTER_W = PSD_FILTER_W
                self.PSD_FILTER_W_THZ = freq_red_to_THz(PSD_FILTER_W, self.DT_FS)
            else:
                raise ValueError("Freq units not valid.")
        else:
            pass  # try to use the internal value
        if self.PSD_FILTER_W is not None:
            self.PSD_FILTER_WF = int(round(self.PSD_FILTER_W * self.NFREQS * 2.0))
        else:
            raise ValueError("Filter window width not defined.")

        if window_type == "rectangular":
            assert self.PSD_FILTER_WF is not None
            self.fpsd = runavefilter(self.psd, self.PSD_FILTER_WF)

            # filter log-psd
            if logpsd_filter_type == 1:
                assert self.PSD_FILTER_WF is not None
                self.flogpsd = runavefilter(self.logpsd, self.PSD_FILTER_WF)
            else:
                self.flogpsd = np.log(self.fpsd)
        else:
            raise KeyError("Window type unknown.")

    def compute_acf(self, NLAGS: int | None = None) -> None:
        if NLAGS is not None:
            self.NLAGS = NLAGS
        else:
            self.NLAGS = self.N
        n_lags = self.NLAGS if NLAGS is None else NLAGS
        self.NLAGS = n_lags
        self.acf = np.zeros((n_lags, self.N_EQUIV_COMPONENTS))
        for d in range(self.N_EQUIV_COMPONENTS):
            self.acf[:, d] = acovf(self.traj[:, d], unbiased=True, fft=True)[:n_lags]
        self.acfm = np.mean(self.acf, axis=1)  # average acf

    def compute_gkintegral(self) -> None:
        if self.acf is None:
            raise RuntimeError("Autocovariance is not defined.")
        self.tau = integrate_acf(self.acf)
        self.taum = np.mean(self.tau, axis=1)  # average tau

    def resample(
        self,
        TSKIP: int | None = None,
        fstar_THz: float | None = None,
        FILTER_W: int | None = None,
        plot: bool = False,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        FIGSIZE: tuple[float, float] | None = None,
        verbose: bool = True,
    ) -> Any:
        return resample_timeseries(
            self,
            TSKIP,
            fstar_THz,
            FILTER_W,
            plot,
            PSD_FILTER_W,
            freq_units,
            FIGSIZE,
            verbose,
        )


################################################################################

# set the default plotter of this class
MDSample.set_plotter()

################################################################################
