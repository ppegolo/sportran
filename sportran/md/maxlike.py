# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as sp
from scipy.special import multigammaln
from scipy.optimize import minimize
from scipy.linalg import cholesky
import opt_einsum
from sportran.utils import log
from .tools.filter import runavefilter

EULER_GAMMA = (
    0.57721566490153286060651209008240243104215933593992  # Euler-Mascheroni constant
)
LOG2 = np.log(2)


class MaxLikeFilter:
    """
    Maximum-likelihood estimate of the Onsager or transport coefficient.

    ** INPUT VARIABLES:
    data            = The noisy data. Either the spectral matrix, or one of its components.
    model           = Function that models the data (for now only spline)
    n_parameters    = Number of parameters to be used for the fit
    n_components    = Number of independent samples the data is generated from.
    n_currents      = Number of independent flux types.
    """

    def __init__(
        self,
        data=None,
        model=None,
        n_parameters=None,
        n_components=None,
        n_currents=None,
        likelihood=None,
        solver=None,
        omega_fixed=None,
    ):
        """
        Initialize the MaxLikeFilter class with the provided parameters.
        """
        log.write_log("MaxLikeFilter Initialization")

        self.data = data
        self.model = model
        self.n_parameters = n_parameters
        self.n_components = n_components
        self.n_currents = n_currents
        self.solver = solver
        self.omega_fixed = omega_fixed
        self.omega = np.arange(data.shape[-1]) if data is not None else None

        # Set likelihood function
        self.log_like = self._get_likelihood_function(likelihood)

        # Store optimization results
        self.parameters_mean = None
        self.parameters_std = None
        self.optimizer_res = None
        self.log_likelihood_value = None

    def _get_likelihood_function(self, likelihood):
        """
        Get the likelihood function based on the provided likelihood type.
        """
        if likelihood is None:
            return None
        likelihood = likelihood.lower()
        likelihoods = {
            "wishart": self.log_likelihood_wishart,
            "chisquare": self.log_likelihood_diag,
            "chisquared": self.log_likelihood_diag,
            "variancegamma": self.log_likelihood_offdiag,
            "variance-gamma": self.log_likelihood_offdiag,
        }
        return likelihoods.get(likelihood) or self._unsupported_likelihood_error()

    def _unsupported_likelihood_error(self):
        raise NotImplementedError(
            "Supported likelihoods: wishart, chisquare, variance-gamma"
        )

    def _validate_parameters(self):
        """
        Ensure that all necessary parameters are set before running maxlike.
        """
        assert (
            self.n_parameters is not None
        ), "Number of parameters (n_parameters) must be provided"
        assert self.solver is not None, "Solver must be provided"
        assert self.data is not None, "`data` must be provided"
        assert self.log_like is not None, "Likelihood must be set"
        assert self.model is not None, "Model must be provided"

    def maxlike(
        self,
        data=None,
        model=None,
        n_parameters=None,
        likelihood=None,
        solver=None,
        mask=None,
        n_components=None,
        n_currents=None,
        guess_runave_window=50,
        omega_fixed=None,
        write_log=True,
        minimize_kwargs=None,
    ):
        """
        Perform the maximum-likelihood estimation.
        """
        # Update instance variables if provided
        self._update_parameters(
            data,
            model,
            n_parameters,
            likelihood,
            solver,
            n_components,
            n_currents,
            omega_fixed,
        )

        # Validate necessary variables
        self._validate_parameters()

        if write_log:
            log.write_log(
                "Maximum-likelihood estimate with {} parameters".format(
                    self.n_parameters
                )
            )

        self._prepare_data(mask)
        self._initialize_spline_nodes(write_log)
        guess_data = self.guess_data(
            self.data,
            self.omega,
            self.omega_fixed,
            self.n_components,
            self.n_currents,
            window=guess_runave_window,
        )
        self.data = np.moveaxis(
            self.data, self.data.shape.index(max(self.data.shape)), 0
        )

        # Perform optimization
        self._optimize_parameters(guess_data, minimize_kwargs, write_log)

    def _update_parameters(
        self,
        data,
        model,
        n_parameters,
        likelihood,
        solver,
        n_components,
        n_currents,
        omega_fixed,
    ):
        """
        Update class parameters with new values if provided.
        """
        if data is not None:
            self.data = data
            self.omega = np.arange(data.shape[-1])
        if model is not None:
            self.model = model
        if n_parameters is not None:
            self.n_parameters = n_parameters
        if likelihood is not None:
            self.log_like = self._get_likelihood_function(likelihood)
        if solver is not None:
            self.solver = solver
        if n_components is not None:
            self.n_components = n_components
        if n_currents is not None:
            self.n_currents = n_currents
        # if omega_fixed is not None:
        self.omega_fixed = omega_fixed
        # else:
        #     self.omega_fixed = None

    def _prepare_data(self, mask):
        """
        Prepare data for processing, applying mask if provided.
        """
        if mask is not None:
            self.data = self.data[mask]

        # Validate data shape
        self._validate_data_shape()

        self.omega = np.arange(self.data.shape[-1])

    def _validate_data_shape(self):
        """
        Validate the shape of the input data.
        """
        if len(self.data.shape) == 3:
            assert (
                self.log_like == self.log_likelihood_wishart
            ), "Misshaped `data` for likelihood"
            assert (
                self.data.shape[0] == self.data.shape[1]
            ), "Data for a Wishart estimate must be a (n,n,N) array."
        elif len(self.data.shape) != 1:
            raise ValueError("`data` should be a 1d or 3d array")

    def _initialize_spline_nodes(self, write_log):
        """
        Initialize spline nodes for the model.
        """
        if self.omega_fixed is None:
            if write_log:
                log.write_log(
                    "Spline nodes are equispaced from 0 to the Nyquist frequency."
                )
            args = np.int32(
                np.linspace(
                    0, self.data.shape[-1] - 1, self.n_parameters, endpoint=True
                )
            )
            self.omega_fixed = self.omega[args]
        assert self.omega_fixed.shape[0] == self.n_parameters

    def _optimize_parameters(self, guess_data, minimize_kwargs, write_log):
        """
        Perform the optimization to find the parameters that maximize the likelihood.
        """
        self._guess_data = guess_data
        res = minimize(
            fun=self.log_like,
            x0=guess_data,
            args=(
                self.model,
                self.omega,
                self.omega_fixed,
                self.data,
                self.n_currents,
                self.n_components,
            ),
            method=self.solver,
            **(minimize_kwargs or {}),
        )

        self._store_optimization_results(res, write_log)

    def _store_optimization_results(self, res, write_log):
        """
        Store the results of the optimization.
        """
        self.parameters_mean = res.x
        if hasattr(res, "hess_inv"):
            cov = res.hess_inv
            if write_log:
                log.write_log(
                    (
                        f"The {self.solver} solver provides Hessian. "
                        "Covariance matrix estimated through Laplace approximation."
                    )
                )
            self.parameters_cov = cov
            self.parameters_std = np.sqrt(cov.diagonal())
        else:
            if write_log:
                log.write_log(
                    (
                        f"The {self.solver} solver does not provide Hessian. "
                        "No covariance matrix output."
                    )
                )
            self.parameters_std = None

        self.optimizer_res = res
        self.log_likelihood_value = -self.log_like(
            res.x,
            self.model,
            self.omega,
            self.omega_fixed,
            self.data,
            self.n_currents,
            self.n_components,
        )

    def guess_data(self, data, omega, omega_fixed, ell, nu, window=10):
        """
        Moving average of the input data as initial guess for the parameter estimation.
        """
        try:
            guess_data = self._compute_moving_average(data, window)
        except Exception as e:
            log.write_log(
                f"Guessing data failed with exception: {e}. Window changed to 10."
            )
            guess_data = self._compute_moving_average(data, 10)

        return self._process_guess_data(guess_data, omega, omega_fixed, ell, nu)

    def _compute_moving_average(self, data, window):
        """
        Compute the moving average of the data.
        """
        shape = data.shape
        return np.array(
            [runavefilter(c, window) for c in data.reshape(-1, shape[-1])]
        ).reshape(shape)

    def _process_guess_data(self, guess_data, omega, omega_fixed, ell, nu):
        """
        Process guess data based on the likelihood.
        """
        guess_data = np.array(
            [
                guess_data[..., j]
                for j in [
                    np.argmin(np.abs(omega - omega_fixed[i]))
                    for i in range(len(omega_fixed))
                ]
            ]
        )

        if self.log_like == self.log_likelihood_wishart:
            guess_data = self._process_wishart_guess_data(guess_data, nu)

        return guess_data

    def _process_wishart_guess_data(self, guess_data, nu):
        """
        Process the guess data for Wishart likelihood.
        """
        guess_data = np.array([cholesky(g, lower=False) for g in guess_data])
        upper_triangle_indices = np.triu_indices(nu)
        nw = self.omega_fixed.shape[0]
        return self._flatten_wishart_parameters(
            guess_data, upper_triangle_indices, nw, nu
        )

    def _flatten_wishart_parameters(self, guess_data, upper_triangle_indices, nw, nu):
        """
        Flatten the wishart parameters for optimization.
        """
        guess_params = np.zeros((nw, nu * (nu + 1) // 2))
        ie = 0
        for i, j in zip(*upper_triangle_indices):
            guess_params[:, ie] = guess_data[:, i, j]
            ie += 1
        return guess_params.flatten()

    def log_likelihood_wishart(
        self, w, model, omega, omega_fixed, data_, nu, ell, eps=1e-3
    ):
        """
        Logarithm of the Wishart probability density function.
        """
        n = ell
        p = nu

        # Compute scale matrix from the model
        # (symmetrize to ensure positive definiteness)
        V = scale_matrix(model, w, omega, omega_fixed, p)
        X = data_

        if n < p:
            S = np.linalg.svd(X, hermitian=True, compute_uv=False)
            detX = np.array([np.prod(s[abs(s) > eps]) for s in S])
        else:
            detX = np.linalg.det(X)

        invV = np.linalg.inv(V)
        detV = np.linalg.det(V)

        trinvV_X = opt_einsum.contract("wab,wba->w", invV, X)

        # Version with all the (irrelevant) constant factors
        # logG = multigammaln(0.5 * n, p)
        # log_pdf = (
        #     0.5
        #     *((n - p - 1) * np.log(detX) - trinvV_X - n * np.log(detV) - n * p * LOG2)
        #     - logG
        # )

        log_pdf = (n - p - 1) * np.log(detX) - trinvV_X - n * np.log(detV)

        return -np.sum(log_pdf)

    def log_likelihood_diag(self, w, model, omega, omega_fixed, data_, nu, ell):
        """
        Negative of the logarithm of the Chi-squared probability density function.
        """
        spline = model(omega_fixed, w)
        spectrum = 2 * spline(omega)
        dof = 2 * (ell - nu + 1)
        z = np.abs(dof * data_ / spectrum)  # This is chi-squared distributed

        logz = np.log(z)
        log_pdf = (2 - dof) * logz + z

        return np.sum(log_pdf)

    def log_likelihood_offdiag(self, w, model, omega, omega_fixed, data_, nu, ell):
        """
        Negative of the logarithm of the Variance-Gamma probability density function.
        """
        spline = model(omega_fixed, w)
        rho = np.clip(spline(omega), -0.98, 0.98)
        _alpha = 1 / (1 - rho**2)
        _beta = rho / (1 - rho**2)
        _lambda = 0.5 * ell * nu
        _gamma2 = _alpha**2 - _beta**2
        _lambda_minus_half = _lambda - 0.5

        z = data_ * nu * ell
        absz = np.abs(z)
        term1 = _lambda * np.log(_gamma2)
        term2 = _lambda_minus_half * np.log(absz)
        term3 = np.log(sp.kv(_lambda_minus_half, _alpha * absz))
        term4 = _beta * z
        term5 = -_lambda_minus_half * np.log(2 * _alpha)
        log_pdf = term1 + term2 + term3 + term4 + term5
        return -np.sum(log_pdf)

    def __repr__(self):
        return "MaxLikeFilter:\n"


def scale_matrix(model, w, omega, omega_fixed, n):
    """
    Compute the scale matrix from the model.
    """
    elements = model(omega_fixed, w)(omega)
    is_complex = elements.dtype == np.complex128

    # Precompute the upper triangle indices
    triu_indices = np.triu_indices(n)

    # Preallocate L
    L = np.zeros((n, n, omega.shape[0]), dtype=elements.dtype)

    # Assign elements to L using vectorized operations
    if is_complex:
        for idx, (i, j) in enumerate(zip(*triu_indices)):
            if i == j:
                L[i, j] = elements[:, idx]
            else:
                L[i, j] = elements[:, idx] + 1j * elements[:, idx + 1]
    else:
        for idx, (i, j) in enumerate(zip(*triu_indices)):
            L[i, j] = elements[:, idx]

    # Compute the scale matrix S using einsum
    S = np.einsum("jiw,jkw->wik", L.conj() if is_complex else L, L)

    return S


def scale_matrix_std_mc(model, w, omega, omega_fixed, n, cov_w, size=1000):
    sample = w + np.random.multivariate_normal(
        mean=np.zeros_like(w), cov=cov_w, size=size
    )
    sample_S = np.stack(
        [scale_matrix(model, ww, omega, omega_fixed, 2) for ww in sample]
    )
    S_std = sample_S.std(axis=0)
    return S_std


def normalize_parameters(p, guess):
    mini, maxi = 1.2 * np.abs(guess), -1.2 * np.abs(guess)
    return (p - mini) / (maxi - mini)


def denormalize_parameters(p, guess):
    mini, maxi = 1.2 * np.abs(guess), -1.2 * np.abs(guess)
    return p * (maxi - mini) + mini
