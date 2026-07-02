import abc
import inspect
import warnings
from typing import Any, Callable

import numpy as np

from sportran.md.bayes import BayesFilter
from sportran.md.cepstral import CepstralFilter, multicomp_cepstral_parameters
from sportran.md.maxlike import MaxLikeFilter
from sportran.md.mdsample import MDSample
from sportran.md.tools.filter import runavefilter
from sportran.md.tools.spectrum import freq_red_to_THz
from sportran.plotter import Plotter
from sportran.plotter.current import CurrentPlotter
from sportran.utils import log

from . import units

__all__ = ["Current"]


class Current(MDSample, abc.ABC):
    """
    Current abstract class for thermo-cepstral analysis. Defines a Current object with
    useful tools to perform analysis.

    INPUT parameters:
     - traj          the current time series (N, N_EQUIV_COMPONENTS) array For a
       multi-component fluid use a (N_CURRENTS, N, N_EQUIV_COMPONENTS) array
     - DT_FS         MD time step [fs]
     - KAPPA_SCALE   the GK conversion factor, multiplies the GK integral

    OPTIONAL parameters:
     - PSD_FILTER_W  PSD filter window [freq_units] (optional)
     - FREQ_UNITS    frequency units   [THz or red] (optional)
     - MAIN_CURRENT_INDEX for a multi-current time series, the index of the "main"
       current (e.g. energy) [0]
     - MAIN_CURRENT_FACTOR factor to be multiplied by the main current [1.0]

    The default plotter is `plotter.CurrentPlotter`. It can be set by
    `Current.set_plotter`.

    The `_current_type`, `_input_parameters`, and  `_KAPPA_SI_UNITS` attributes, and the
    `_builder` method must be defined at the subclass level.
    """

    # parameters are class-specific (a HeatCurrent may use different ones wrt
    # ElectricCurrent) and case-insensitive
    _current_type: str | None = None
    _input_parameters = {"DT_FS", "KAPPA_SCALE"}
    _optional_parameters = {
        "PSD_FILTER_W",
        "FREQ_UNITS",
        "MAIN_CURRENT_INDEX",
        "MAIN_CURRENT_FACTOR",
    }
    _KAPPA_SI_UNITS: str = ""
    _default_plotter = CurrentPlotter
    otherMD: list[MDSample] | None
    cospectrum: np.ndarray | None
    fcospectrum: np.ndarray | None
    cepf: CepstralFilter | None
    kappa: float
    kappa_std: float
    psd: np.ndarray
    logpsd: np.ndarray

    def __init__(self, traj: np.ndarray, **params: Any) -> None:
        # e.g. params: (DT_FS, UNITS, TEMPERATURE, VOLUME, PSD_FILTER_W=None,
        # FREQ_UNITS='THz') validate input parameters

        params = {k.upper(): v for k, v in params.items()}  # convert keys to uppercase
        keyset = set(params.keys())
        if not self._input_parameters.issubset(keyset):
            raise ValueError(
                "The input parameters {} must be defined.".format(
                    self._input_parameters - keyset
                )
            )
        if not keyset.issubset(self._input_parameters | self._optional_parameters):
            raise ValueError(
                "The input parameters {} are not valid.".format(
                    keyset - (self._input_parameters | self._optional_parameters)
                )
            )

        # pop non unit-specific parameters
        PSD_FILTER_W = params.pop("PSD_FILTER_W", None)
        FREQ_UNITS = params.pop("FREQ_UNITS", "THz")

        DT_FS = params.pop("DT_FS")
        MAIN_CURRENT_INDEX = params.pop("MAIN_CURRENT_INDEX", 0)
        MAIN_CURRENT_FACTOR = params.pop("MAIN_CURRENT_FACTOR", 1.0)
        self.otherMD = None
        self.cospectrum = None
        self.fcospectrum = None
        self.cepf = None
        self.initialize_currents(traj, DT_FS, MAIN_CURRENT_INDEX, MAIN_CURRENT_FACTOR)
        self.initialize_units(
            **params
        )  # KAPPA_SCALE or (e.g. UNITS, TEMPERATURE, VOLUME)
        self.TEMPERATURE: float  # set by initialize_units via __setattr__
        self.VOLUME: float  # set by initialize_units via __setattr__
        if self.traj is not None:
            self.compute_psd(PSD_FILTER_W, FREQ_UNITS)
            self.initialize_cepstral_parameters()
        else:
            log.write_log(
                "Warning: trajectory not initialized. You should manually initialize what you need."
            )
        self.cepf = None

    def __repr__(self) -> str:
        msg = (
            type(self).__name__
            + "\n  N_CURRENTS  =  {}\n".format(self.N_CURRENTS)
            + "  KAPPA_SCALE =  {}\n".format(self.KAPPA_SCALE)
        )
        for key in self._input_parameters - {"DT_FS", "KAPPA_SCALE"}:
            msg += "  {:11} =  {}\n".format(key, getattr(self, key))
        msg += super().__repr__()
        if self.otherMD:
            msg += "additional currents:\n"
            for current in self.otherMD:
                msg += current.__repr__()
        if self.cepf:
            msg += self.cepf.__repr__()
        try:
            msg += "\n  kappa* = {:18f} +/- {:10f}  {}\n".format(
                self.kappa, self.kappa_std, self._KAPPA_SI_UNITS
            )
        except AttributeError:
            pass
        return msg

    @property
    @abc.abstractmethod
    def _builder(self) -> dict[str, float | str | None]:
        """
        Returns a dictionary of all keyworded parameters needed to rebuild an identical
        object of the same class. The trajectory is excluded. Used by
        self._get_builder().

        This is a virtual method that must be defined in a subclass of Current.
        """
        raise NotImplementedError

    def _get_builder(self) -> tuple[type["Current"], dict[str, Any]]:
        """
        Get a tuple (class, builder) that can be used to build a new object with same
        parameters:
          TimeSeries, builder = self._get_builder() new_ts = TimeSeries(**builder)
        """
        if self.MANY_CURRENTS:
            if self.otherMD is None:
                raise RuntimeError("Other currents not initialized.")
            traj_array = np.vstack(([self.traj], [j.traj for j in self.otherMD]))
        else:
            traj_array = self.traj
        kwargs = self._builder
        kwargs.update(traj=traj_array)
        return type(self), kwargs

    @classmethod
    def set_plotter(cls, plotter: type[Plotter] | None = None) -> None:
        """
        Set the plotter class. The _plotter attribute will contain the selected plotter
        class. All the plot functions of plotter (named 'plot_*') will be transformed
        into methods of Current.

        **NOTE** If called by a subclass, it will change the plotter of the base class
        (Current) and all its subclasses. If this is not a good behavior, we should
        change it in the future.
        """
        # if called by a subclass of Current, change the base class (Current)
        if issubclass(cls, Current) and cls != Current:
            cls = Current  # type: ignore[type-abstract]
        # (note: it is not possible to delete the parent class' attributes from a child.
        # But here we forcibly do this operation on cls = Current)
        super().set_plotter(plotter)

    def initialize_currents(
        self,
        j: np.ndarray,
        DT_FS: float,
        main_current_index: int = 0,
        main_current_factor: float = 1.0,
    ) -> None:
        # check if we have a multicomponent fluid
        j = np.array(j, dtype=float)
        if len(j.shape) == 3:
            self.N_CURRENTS = j.shape[0]
            if self.N_CURRENTS == 1:
                self.MANY_CURRENTS = False
                j = np.squeeze(j, axis=0)
            else:
                self.MANY_CURRENTS = True
        elif len(j.shape) <= 2:
            self.N_CURRENTS = 1
            self.MANY_CURRENTS = False
        else:
            raise ValueError("Shape of j {} not valid.".format(j.shape))

        if self.MANY_CURRENTS:
            log.write_log("Using multicomponent code.")
            super().__init__(
                traj=(j[main_current_index] * main_current_factor), DT_FS=DT_FS
            )
            # initialize other MDSample currents
            other_currents_idxs = (
                np.arange(self.N_CURRENTS) != main_current_index
            )  # select the other currents
            self.otherMD = [
                MDSample(traj=js, DT_FS=DT_FS) for js in j[other_currents_idxs]
            ]
        else:
            log.write_log("Using single component code.")
            super().__init__(traj=(j * main_current_factor), DT_FS=DT_FS)
            self.otherMD = None

        # initialize cospectrum, that is not initialized by MDSample.initialize_psd
        self.cospectrum = None
        self.fcospectrum = None

    @classmethod
    def _get_units(cls) -> dict[str, Callable[[float, float], float]]:
        try:
            # get the units submodule corresponding to this class
            if cls._current_type is None:
                raise TypeError
            units_module = getattr(units, cls._current_type)
        except AttributeError:
            print(
                'No units submodule defined for the current type "{}". Add units to a file "current/units/{}.py".'.format(
                    cls._current_type, cls._current_type
                )
            )
            return {}
        except TypeError:
            raise RuntimeError(
                'No units can be defined for a generic Current. Define a "KAPPA_SCALE" instead.'
            )

        # get all functions that start with "scale_kappa_" into a dictionary {"name":
        # function}
        units_prefix = "scale_kappa_"
        units_d = {
            name.replace(units_prefix, ""): function
            for name, function in inspect.getmembers(
                units_module,
                predicate=lambda f: (
                    inspect.isfunction(f) and f.__name__.startswith(units_prefix)
                ),
            )
        }
        if not units_d:
            print(
                'Warning: No units defined for a current type "{}". Add them to the module "current/units/{}.py'.format(
                    cls._current_type, cls._current_type
                )
            )
        return units_d

    @classmethod
    def get_units_list(cls) -> Any:
        """
        Get the list of supported units. Units are defined in the module
        current/units/{current_type}.py, where {current_type} is the _current_type
        attribute of this class ('heat', 'electric', ...).
        """
        return cls._get_units().keys()

    def initialize_units(self, **parameters: Any) -> None:
        """
        Initializes the units and defines the KAPPA_SCALE.
        """
        self.UNITS = parameters.pop("UNITS", None)

        # set unit-specific parameters
        for param, value in parameters.items():
            self.__setattr__(param, value)

        # validate units and define KAPPA_SCALE from units conversion function
        if self.UNITS:
            units_list = self.get_units_list()
            if len(units_list) == 0:
                raise RuntimeError(
                    'No units defined for a current type "{}". Add them to the module "current/units/{}.py'.format(
                        self._current_type, self._current_type
                    )
                )
            elif self.UNITS in units_list:
                units_conversion_func = self._get_units()[self.UNITS]
                self.KAPPA_SCALE = units_conversion_func(**parameters)  # type: ignore[call-arg]
            else:
                raise ValueError(
                    'Units "{}" not valid. Valid units are:\n  {}'.format(
                        self.UNITS, self.get_units_list()
                    )
                )

    def compute_psd(
        self,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        method: str = "fft",
        DT_FS: float | None = None,
        normalize: bool = False,
    ) -> None:
        # overrides MDSample method
        """
        Compute the periodogram from the heat current time series. If a PSD_FILTER_W
        (expressed in freq_units) is known or given, the psd is also filtered. The PSD
        is multiplied by DT_FS at the end.
        """
        # number of degrees of freedom of the chi-square distribution of the psd / 2
        self.ndf_chi = self.N_EQUIV_COMPONENTS - self.N_CURRENTS + 1
        if self.ndf_chi <= 0:
            warnings.warn(
                "The number of degrees of freedom of the chi-squared distribution is <=0. The number of "
                "equivalent (Cartesian) components of the input current must be >= number of currents.",
                RuntimeWarning,
                stacklevel=2,
            )

        if self.MANY_CURRENTS:
            if self.otherMD is None:
                raise RuntimeError(
                    "self.otherMD cannot be None (wrong/missing initialization?)"
                )
            self._compute_psd_multi(self.otherMD, PSD_FILTER_W, freq_units)
        else:
            super().compute_psd(PSD_FILTER_W, freq_units)

    def _compute_psd_multi(
        self,
        others: list[MDSample] | tuple[MDSample, ...] | np.ndarray,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        normalize: bool = False,
        call_other: bool = True,
    ) -> None:
        """
        For multi-component (many-current) systems: compute the cospectrum matrix and
        the transport coefficient. The results have almost the same statistical
        properties. The chi-square distribution has ndf = 2(ndf_chi) = 2(l - M + 1),
        where l is the number of time series for each current (N_EQUIV_COMPONENTS), M is
        the number of currents (N_CURRENTS). ! NOTICE: if l < M this will not work.

        In this routine the mean over the number of temporal series is already
        multiplied by the correct factor (the transport coefficient will be obtained by
        multiplying the result by 0.5, as in the one-component case). The output arrays
        are the same as in the one-component case. The elements of the matrix are
        multiplied by DT_FS at the end. If a PSD_FILTER_W is known or given, the psd is
        also filtered. others is a list of other currents, i.e. MDSample objects.

        For example, in the case of 4 currents, of which j is the energy current and j1,
        j2, j3 are mass currents:
           j._compute_psd_multi([j1,j2,j3], PSD_FILTER_W, freq_units)
        """
        # check if others is an array
        if not isinstance(others, (list, tuple, np.ndarray)):
            others = [others]
        if self.traj is None:
            raise ValueError("Trajectory not defined.")

        self.spectrALL = np.fft.rfft(self.traj, axis=0)
        self.NFREQS = self.spectrALL.shape[0]
        self.freqs = np.linspace(0.0, 0.5, self.NFREQS)
        self.DF = 0.5 / (self.NFREQS - 1)
        self.DF_THZ = freq_red_to_THz(self.DF, self.DT_FS)
        self.freqs_THz = self.freqs / self.DT_FS * 1000.0
        self.Nyquist_f_THz = float(self.freqs_THz[-1])

        # calculate the same thing on the other trajectory
        if call_other:
            for other in others:  # call other._compute_psd_multi (MDsample method)
                Current._compute_psd_multi(
                    other,  # type: ignore[arg-type]
                    [self],
                    PSD_FILTER_W,
                    freq_units,
                    normalize,
                    False,
                )
        else:
            return

        # define the cospectrum matrix. Its shape is (2, 2, NFREQS, n_spatial_dim) [
        #  self.spectrALL*self.spectrALL.conj()
        #  self.spectrALL*other.spectrALL.conj() ] [
        #  other.spectrALL*self.spectrALL.conj()
        #  other.spectrALL*other.spectrALL.conj() ]
        other_spectrALL = []
        for other in others:
            other_spectrALL.append(other.spectrALL)  # type: ignore[union-attr]

        # compute the matrix defined by the outer product of only the first indexes of
        # the two arrays
        covarALL = (
            self.DT_FS
            / (2.0 * (self.NFREQS - 1.0))
            * np.einsum(
                "a...,b...->ab...",
                np.array([self.spectrALL] + other_spectrALL),
                np.array([self.spectrALL] + other_spectrALL).conj(),
            )
        )

        # number of degrees of freedom of the chi-square distribution of the psd / 2
        assert self.ndf_chi == (covarALL.shape[3] - len(other_spectrALL))

        # compute the sum over the last axis (equivalent Cartesian components):
        self.cospectrum = covarALL.sum(axis=3)

        # compute the element 1/"(0,0) of the inverse" (aka the transport coefficient)
        # the diagonal elements of the inverse have very convenient statistical
        # properties
        if self.cospectrum is None:
            raise RuntimeError("Cospectrum not initialized.")
        multi_psd = (
            np.linalg.inv(self.cospectrum.transpose((2, 0, 1)))[:, 0, 0] ** -1
        ).real / self.ndf_chi

        if normalize:
            multi_psd = multi_psd / np.trapezoid(multi_psd) / self.N / self.DT_FS

        self.psd = multi_psd
        self.logpsd = np.log(self.psd)
        self.psd_min = np.min(self.psd)
        self.psd_power = np.trapezoid(self.psd)  # one-side PSD power
        if (PSD_FILTER_W is not None) or (self.PSD_FILTER_W is not None):
            self.filter_psd(PSD_FILTER_W, freq_units)

    def filter_psd(
        self,
        PSD_FILTER_W: float | None = None,
        freq_units: str = "THz",
        window_type: str = "rectangular",
        logpsd_filter_type: int = 1,
    ) -> None:
        """
        Filter the periodogram with the given PSD_FILTER_W [freq_units].
          - PSD_FILTER_W  PSD filter window [freq_units]
          - freq_units    frequency units   ['THz', 'red' (default)]
          - window_type   filtering window type ['rectangular']
        """
        super().filter_psd(PSD_FILTER_W, freq_units, window_type, logpsd_filter_type)

        if window_type == "rectangular":
            # try to filter the other currents (if present)
            if self.cospectrum is not None:
                assert self.PSD_FILTER_WF is not None
                fcospectrum: list[list[np.ndarray]] = []
                for i in range(self.cospectrum.shape[0]):
                    fcospectrum.append([])
                    for j in range(self.cospectrum.shape[1]):
                        ffpsd = runavefilter(self.cospectrum[i, j], self.PSD_FILTER_WF)
                        fcospectrum[i].append(ffpsd / self.N_EQUIV_COMPONENTS)
                self.fcospectrum = np.asarray(fcospectrum)

    def initialize_cepstral_parameters(self) -> None:
        """
        Defines the parameters of the theoretical distribution of the cepstrum.
        """
        if not self.MANY_CURRENTS:
            self.ck_THEORY_var, self.psd_THEORY_mean = multicomp_cepstral_parameters(
                self.NFREQS, self.N_EQUIV_COMPONENTS
            )
        else:
            if self.ndf_chi is None:
                raise RuntimeError("self.ndf_chi cannot be None.")
            self.ck_THEORY_var, self.psd_THEORY_mean = multicomp_cepstral_parameters(
                self.NFREQS, self.ndf_chi
            )

    def bayesian_analysis(
        self,
        model: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
        n_parameters: int,
        is_restart: bool = False,
        n_steps: int = 2000000,
        backend: str = "chain.h5",
        burn_in: int | None = None,
        thin: int | None = None,
        mask: np.ndarray | None = None,
        log_like: str = "off",
    ) -> None:
        assert self.cospectrum is not None
        self.bayes = BayesFilter(
            self.cospectrum,
            model,
            n_parameters,
            self.N_EQUIV_COMPONENTS,
            is_restart=is_restart,
            n_steps=n_steps,
            backend=backend,
            burn_in=burn_in,
            thin=thin,
            mask=mask,
        )
        self.bayes.run_mcmc(log_like=log_like)

        self.offdiag = self.bayes.parameters_mean[0] * self.bayes.factor
        self.offdiag_std = self.bayes.parameters_std[0] * self.bayes.factor

        self.bayesian_log = (
            "-----------------------------------------------------\n"
            + "  BAYESIAN ANALYSIS\n"
            + "-----------------------------------------------------\n"
        )
        self.bayesian_log += (
            "  L_01   = {:18f} +/- {:10f}\n".format(self.offdiag, self.offdiag_std)
            + "-----------------------------------------------------\n"
        )
        log.write_log(self.bayesian_log)
        with open("bayesian_analysis_{}".format(n_parameters), "w+") as g:
            g.write("{}\t{}\n".format(self.offdiag, self.offdiag_std))

    ####################################################################################
    # MAXLIKE methods
    def maxlike_estimate(
        self,
        model: Callable[[np.ndarray, np.ndarray], Callable[[np.ndarray], np.ndarray]],
        n_parameters: str | int = "AIC",
        mask: np.ndarray | None = None,
        likelihood: str = "wishart",
        solver: str = "BFGS",
        guess_runave_window: int = 50,
        minimize_kwargs: dict[str, Any] | None = None,
        ext_guess: np.ndarray | None = None,
        limits: list[tuple[float, float]] | None = None,
        omega_fixed: np.ndarray | None = None,
    ) -> None:
        """
        Perform maximum likelihood estimation and optionally select the optimal number
        of parameters using AIC.
        """
        minimize_kwargs = minimize_kwargs or {}

        # Get the appropriate data based on likelihood type
        data = self._get_data_by_likelihood(likelihood)

        # Initialize MaxLikeFilter object
        self.maxlike = MaxLikeFilter(
            data=data,
            model=model,
            n_components=self.N_EQUIV_COMPONENTS,
            n_currents=self.N_CURRENTS,
            likelihood=likelihood,
            solver=solver,
            ext_guess=ext_guess,
            omega_fixed=omega_fixed,
        )

        # Run the maximum likelihood estimation
        self.maxlike.maxlike(
            n_parameters=n_parameters,
            mask=mask,
            guess_runave_window=guess_runave_window,
            minimize_kwargs=minimize_kwargs,
            limits=limits,
        )

        # Extract and scale results
        self.maxlike.extract_and_scale_results()

        # Access the results from self.maxlike
        self.NLL_mean = self.maxlike.NLL_mean[0]
        self.NLL_std: float | np.ndarray | None = None
        try:
            self.NLL_std = self.maxlike.NLL_std[0]
        except AttributeError:
            pass
        # self.NLL_upper = getattr(self.maxlike, "NLL_upper", None) self.NLL_lower =
        # getattr(self.maxlike, "NLL_lower", None)

        # Store additional results if needed
        self.optimal_nparameters = getattr(self.maxlike, "optimal_nparameters", None)
        self.aic_values = getattr(self.maxlike, "aic_values", None)

        # Add logging for the MLE results
        self.mle_log = (
            "-----------------------------------------------------\n"
            + "  MAXIMUM LIKELIHOOD ESTIMATION\n"
            + "-----------------------------------------------------\n"
        )

        if isinstance(n_parameters, str) and n_parameters.lower() == "aic":
            if self.optimal_nparameters is None:
                self.mle_log += "  Optimal n_parameters (AIC) = N/A\n"
            else:
                self.mle_log += "  Optimal n_parameters (AIC) = {:d}\n".format(
                    int(self.optimal_nparameters)
                )
        else:
            self.mle_log += "  Fixed n_parameters = {:d}\n".format(
                int(self.maxlike.n_parameters)  # type: ignore[arg-type]
            )

        if likelihood == "wishart":
            # Iterate over the upper triangle (including the diagonal)
            for i in range(self.N_CURRENTS):
                for j in range(i, self.N_CURRENTS):
                    mean_val = self.NLL_mean[i, j]
                    if self.NLL_std is None:
                        std_val_w: np.ndarray | int | float = 0
                    else:
                        assert isinstance(self.NLL_std, np.ndarray)
                        std_val_w = self.NLL_std[i, j]

                    self.mle_log += (
                        f"  S_{{{i}{j}}} = {mean_val:18f} +/- {std_val_w:10f}\n"
                    )
        else:
            mean_val = self.NLL_mean * self.KAPPA_SCALE / 2
            if self.NLL_std is None:
                std_val_nw = 0.0
            else:
                std_val_nw = self.NLL_std * self.KAPPA_SCALE / 2

            self.mle_log += "  kappa* = {:18f} +/- {:10f}  {}\n".format(
                mean_val, std_val_nw, self._KAPPA_SI_UNITS
            )
            self.NLL_mean = mean_val
            self.NLL_std = std_val_nw

        self.mle_log += "-----------------------------------------------------\n"

        log.write_log(self.mle_log)

    def _get_data_by_likelihood(self, likelihood: str) -> np.ndarray:
        """
        Get the data to be used for the likelihood estimation based on the provided
        likelihood type.
        """
        likelihood = likelihood.lower()
        if likelihood == "wishart":
            if self.cospectrum is None:
                raise RuntimeError("Cospectrum not initialized.")
            return self.cospectrum.real * self.N_CURRENTS
        elif likelihood in ["chisquare", "chisquared"]:
            return self.psd
        elif likelihood in ["variancegamma", "variance-gamma"]:
            if self.cospectrum is None:
                raise RuntimeError("Cospectrum not initialized.")
            return self.cospectrum.real[0, 1]  # * self.N_CURRENTS
        else:
            raise ValueError(
                "Likelihood must be Wishart, Chi-square, or Variance-Gamma."
            )

    def cepstral_analysis(
        self,
        aic_type: str = "aic",
        aic_Kmin_corrfactor: float = 1.0,
        manual_cutoffK: int | None = None,
    ) -> None:
        """
        Perform cepstral analysis of the current trajectory.

        ``cutoffK`` (``P*-1``) is the number of retained cepstral coefficients. By
        default, it is selected by minimizing the chosen AIC variant and then scaled by
        ``aic_Kmin_corrfactor``.

        Parameters
        ----------
        aic_type : str
            Akaike criterion variant used to choose the cutoff (``'aic'`` or
            ``'aicc'``).
        aic_Kmin_corrfactor : float
            Multiplicative correction applied to the AIC minimum cutoff.
        manual_cutoffK : int or None
            Manual ``P*-1`` cutoff. If provided, the AIC cutoff is ignored.

        Notes
        -----
        Results are stored in ``self.kappa`` and ``self.kappa_std`` (in
        ``self._KAPPA_SI_UNITS``), and the text summary is stored in
        ``self.cepstral_log``.
        """

        self.cepf = CepstralFilter(
            self.logpsd,
            ck_theory_var=self.ck_THEORY_var,
            psd_theory_mean=self.psd_THEORY_mean,
            aic_type=aic_type,
        )
        if self.cepf is None:
            raise RuntimeError("Cepstral filter initialization failed.")
        self.cepf.scan_filter_tau(
            cutoffK=manual_cutoffK, aic_Kmin_corrfactor=aic_Kmin_corrfactor
        )
        self.kappa = self.cepf.tau_cutoffK * self.KAPPA_SCALE * 0.5
        self.kappa_std = self.cepf.tau_std_cutoffK * self.KAPPA_SCALE * 0.5

        assert self.cepf.cutoffK is not None
        assert self.cepf.aic_Kmin is not None
        self.cepstral_log = (
            "-----------------------------------------------------\n"
            + "  CEPSTRAL ANALYSIS\n"
            + "-----------------------------------------------------\n"
        )
        if not self.cepf.manual_cutoffK_flag:
            self.cepstral_log += "  cutoffK = (P*-1) = {:d}  (auto, AIC_Kmin = {:d}, corr_factor = {:4})\n".format(
                int(self.cepf.cutoffK),
                int(self.cepf.aic_Kmin),
                self.cepf.aic_Kmin_corrfactor,
            )
        else:
            self.cepstral_log += (
                "  cutoffK  = (P*-1) = {:d}  (manual, AIC_Kmin = {:d})\n".format(
                    int(self.cepf.cutoffK),
                    int(self.cepf.aic_Kmin),
                )
            )
        self.cepstral_log += (
            "  L_0*   = {:18f} +/- {:10f}\n".format(
                self.cepf.logtau_cutoffK, self.cepf.logtau_std_cutoffK
            )
            + "  S_0*   = {:18f} +/- {:10f}\n".format(
                self.cepf.tau_cutoffK, self.cepf.tau_std_cutoffK
            )
            + "-----------------------------------------------------\n"
            + "  kappa* = {:18f} +/- {:10f}  {}\n".format(
                self.kappa, self.kappa_std, self._KAPPA_SI_UNITS
            )
            + "-----------------------------------------------------\n"
        )
        log.write_log(self.cepstral_log)

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
    ) -> Any:  # yapf: disable
        """
        Simulate the resampling of the time series.

        Parameters
        ----------
        TSKIP        = sampling time [steps] fstar_THz    = target cutoff frequency
        [THz] TSKIP and fstar_THZ are mutually exclusive.

        FILTER_W     = pre-sampling filter window width [steps] plot         = plot the
        PSD [True] PSD_FILTER_W = PSD filtering window width [chosen frequency units]
        freq_units   = 'thz'  [THz]
                       'red'  [omega*DT/(2*pi)]
        FIGSIZE      = plot figure size verbose      = print log [True]

        Returns
        -------
        xf : a filtered & resampled time series object ax : an array of plot axes,
        optional (if plot=True)
        """
        xf = super().resample(
            TSKIP, fstar_THz, FILTER_W, False, PSD_FILTER_W, freq_units, None, verbose
        )

        if plot:
            try:
                axs = self.plot_resample(  # type: ignore[attr-defined]
                    xf=xf,
                    freq_units=freq_units,
                    PSD_FILTER_W=PSD_FILTER_W,
                    FIGSIZE=FIGSIZE,
                )
                return xf, axs
            except AttributeError:
                print("Plotter does not support the plot_resample method")
        else:
            return xf

    def fstar_analysis(
        self,
        TSKIP_LIST: list[int] | np.ndarray,
        aic_type: str = "aic",
        aic_Kmin_corrfactor: float = 1.0,
        manual_cutoffK: int | None = None,
        plot: bool = True,
        axes: Any = None,
        FIGSIZE: tuple[float, float] | None = None,
        verbose: bool = False,
        **plot_kwargs: Any,
    ) -> Any:  # yapf: disable
        from sportran.current.tools.fstar_analysis import fstar_analysis

        return fstar_analysis(
            self,
            TSKIP_LIST,
            aic_type,
            aic_Kmin_corrfactor,
            manual_cutoffK,
            plot,
            axes,
            FIGSIZE,
            verbose,
            **plot_kwargs,
        )


################################################################################

# set the default plotter of this class
Current.set_plotter()
