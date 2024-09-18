import numpy as np
import emcee
from multiprocessing import Pool
import scipy.special as sp
from scipy.special import multigammaln
from scipy.optimize import minimize
from scipy.linalg import cholesky
import opt_einsum
import time
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

    ** INTERNAL VARIABLES:

    TODO
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
        log.write_log("MaxLikeFilter")

        self.data = data
        self.model = model
        self.n_parameters = n_parameters
        self.n_components = n_components
        self.n_currents = n_currents
        self.likelihood = likelihood
        self.solver = solver
        self.omega_fixed = omega_fixed

        if data is not None:
            self.omega = np.arange(data.shape[-1])
        else:
            self.omega = None

        if likelihood is not None:
            self.set_likelihood(likelihood)
        else:
            self.log_like = None

        self.parameters_mean = None
        self.parameters_std = None
        self.optimizer_res = None
        self.log_likelihood_value = None

    def set_likelihood(self, likelihood):
        """
        Set the likelihood function based on the provided likelihood type.
        """
        likelihood = likelihood.lower()
        if likelihood == "wishart":
            self.log_like = self.log_likelihood_wishart
        elif likelihood == "chisquare" or likelihood == "chisquared":
            self.log_like = self.log_likelihood_diag
        elif likelihood == "variancegamma" or likelihood == "variance-gamma":
            self.log_like = self.log_likelihood_offdiag
        else:
            raise NotImplementedError(
                "Currently implemented likelihoods are: `wishart`, `chisquare`, `variance-gamma`"
            )

    def __repr__(self):
        msg = "MaxLikeFilter:\n"
        return msg

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

        if data is not None:
            self.data = data
            self.omega = np.arange(data.shape[-1])
        if model is not None:
            self.model = model
        if n_parameters is not None:
            self.n_parameters = n_parameters
        if likelihood is not None:
            self.set_likelihood(likelihood)
        if solver is not None:
            self.solver = solver
        if n_components is not None:
            self.n_components = n_components
        if n_currents is not None:
            self.n_currents = n_currents
        if omega_fixed is not None:
            self.omega_fixed = omega_fixed

        # Validate necessary variables
        assert (
            self.n_parameters is not None
        ), "Number of parameters (n_parameters) must be provided"
        assert self.solver is not None, "Solver must be provided"
        assert self.data is not None, "`data` must be provided"
        assert self.log_like is not None, "Likelihood must be set"

        if write_log:
            log.write_log(
                "Maximum-likelihood estimate with {} parameters".format(
                    self.n_parameters
                )
            )

        ell = self.n_components

        if mask is not None:
            self.mask = mask
            self.data = self.data[mask]

        nu = self.n_currents

        # Check data shape
        assert (
            len(self.data.shape) == 1 or len(self.data.shape) == 3
        ), "`data` should be a 1d array (diagonal or off-diagonal estimates) or a 3d array (Wishart matrix estimate)"
        if len(self.data.shape) == 3:
            assert (
                self.likelihood == "wishart"
            ), f"Misshaped `data` for {self.likelihood} likelihood."
            assert (
                self.data.shape[0] == self.data.shape[1]
            ), f"data.shape={self.data.shape}. `data` for a Wishart estimate must be a (n,n,N) array."
        else:
            assert (
                self.likelihood != "wishart"
            ), f"Misshaped `data` for {self.likelihood} likelihood."

        # Frequency
        omega = np.arange(self.data.shape[-1])
        self.omega = omega

        # Spline nodes
        if omega_fixed is None:
            try:
                assert self.omega_fixed.shape[0] == n_parameters
            except:
                if write_log:
                    log.write_log(
                        "Spline nodes are equispaced from 0 to the Nyquist frequency."
                    )
                args = np.int32(
                    np.linspace(
                        0, self.data.shape[-1] - 1, self.n_parameters, endpoint=True
                    )
                )
                self.omega_fixed = omega[args]
        else:
            self.omega_fixed = omega_fixed
        assert self.omega_fixed.shape[0] == n_parameters

        # Spline model
        assert self.model is not None, "Model must be provided"

        # Initial guess for optimization
        guess_data = self.guess_data(
            self.data,
            omega,
            self.omega_fixed,
            ell,
            nu,
            window=guess_runave_window,
            loglike=self.likelihood,
        )
        self.data = np.moveaxis(
            self.data, self.data.shape.index(max(self.data.shape)), 0
        )

        # Minimize the negative log-likelihood
        self._guess_data = guess_data
        guess_par = guess_data  # normalize_parameters(guess_data, guess_data)
        # print(guess_data - denormalize_parameters(guess_par, guess_data))
        res = minimize(
            fun=self.log_like,
            x0=guess_par,
            args=(self.model, omega, self.omega_fixed, self.data, nu, ell),
            method=self.solver,
            **minimize_kwargs,
        )

        # Covariance of the parameters
        try:
            cov = res.hess_inv
            if write_log:
                log.write_log(
                    f"The {self.solver} solver features the calculation of the Hessian. The covariance matrix will be estimated through the Laplace approximation."
                )
        except:
            if write_log:
                log.write_log(
                    f"The {self.solver} solver does not feature the calculation of the Hessian. No covariance matrix will be output."
                )
            cov = None

        self.parameters_mean = res.x
        # self.parameters_mean = denormalize_parameters(res.x, self._guess_data)
        if cov is not None:
            self.parameters_std = np.sqrt(cov.diagonal())

        self.optimizer_res = res
        self.log_likelihood_value = -self.log_like(
            res.x, self.model, omega, self.omega_fixed, self.data, nu, ell
        )

    ################################################
    # Helper functions

    def guess_data(
        self, data, omega, omega_fixed, ell, nu, window=10, loglike="wishart"
    ):
        """
        Moving average of the input data as initial guess for the parameter estimation.
        """
        shape = data.shape
        try:
            guess_data = np.array(
                [runavefilter(c, window) for c in data.reshape(-1, shape[-1])]
            ).reshape(shape)
        except:
            log.write_log(
                f"Guessing data with a running average with a {window} large window failed. Window changed to 10."
            )
            window = 10
            guess_data = np.array(
                [runavefilter(c, window) for c in data.reshape(-1, shape[-1])]
            ).reshape(shape)

        guess_data = np.array(
            [
                guess_data[..., j]
                for j in [
                    np.argmin(np.abs(omega - omega_fixed[i]))
                    for i in range(len(omega_fixed))
                ]
            ]
        )
        # print(guess_data.shape)
        if loglike == "wishart":
            guess_data = np.array(
                [cholesky(g, lower=False) for g in guess_data]
            )  # / np.sqrt(ell)
            upper_triangle_indices = np.triu_indices(nu)

            nw = omega_fixed.shape[0]
            ie = 0
            if guess_data.dtype == np.complex128:
                guess_params = np.zeros((nw, nu**2))
                for i, j in zip(*upper_triangle_indices):
                    if i == j:
                        guess_params[:, ie] = guess_data[:, i, j].real
                        ie += 1
                    else:
                        guess_params[:, ie] = guess_data[:, i, j].real
                        ie += 1
                        guess_params[:, ie] = guess_data[:, i, j].imag
                        ie += 1
                guess_data = guess_params.flatten()
            else:
                guess_params = np.zeros((nw, nu * (nu + 1) // 2))
                for i, j in zip(*upper_triangle_indices):
                    guess_params[:, ie] = guess_data[:, i, j]
                    ie += 1
                guess_data = guess_params.flatten()

        return guess_data

    def log_likelihood_wishart(
        self, w, model, omega, omega_fixed, data_, nu, ell, eps=1e-3
    ):
        """
        Logarithm of the Wishart probability density function.
        """
        n = ell
        p = nu

        # Compute scale matrix from the model (symmetrize to ensure positive definiteness)
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

        # coeff_detV = -n
        # coeff_detX = n - p - 1
        logG = multigammaln(0.5 * n, p)

        log_pdf = (
            0.5
            * (
                (n - p - 1) * np.log(detX)
                - trinvV_X
                - n * np.log(detV)
                - n * p * np.log(2)
            )
            - logG
        )

        # log_pdf = (
        #     0.5*(coeff_detV*np.log(detV + eps) + coeff_detX*np.log(detX + eps) - trinvV_X) - logG
        # )

        # return the negative LL because of `minimize`
        assert np.sum(log_pdf) > 0
        tot = -np.mean(log_pdf)
        return tot

    def log_likelihood_complex_wishart(
        self, w, model, omega, omega_fixed, data_, nu, ell, eps=1e-9
    ):
        """
        Logarithm of the Complex Wishart probability density function.
        """
        n = ell
        p = nu

        # Compute scale matrix from the model (symmetrize to ensure positive definiteness)
        S = scale_matrix(model, w, omega, omega_fixed, p)
        # nw = w.shape[0]//2
        # real_part = model(omega_fixed, w[:nw])
        # imag_part = model(omega_fixed, w[nw:])

        # # Lower Cholesky factor of S
        # L = (real_part(omega) + 1j*imag_part(omega))
        # S = opt_einsum.contract('wba,wbc->wac', L.conj(), L)

        # The distribution refers to the sample covariance matrix of the (complex) multinormal vectors, not their average
        X = data_

        if n < p:
            raise ValueError("n must be greater or equal than p")
            # Singular Wishart
            multig = multigammaln(0.5 * n, n)
            S = np.linalg.svd(X, hermitian=True, compute_uv=False)
            detX = np.array([np.prod(s[abs(s) > eps]) for s in S])
        else:
            # multig = multigammaln(0.5 * n, p)
            logdetX = np.log(np.abs(np.linalg.det(X).real))

        invS = np.linalg.inv(S)
        logdetS = np.log(np.abs(np.linalg.det(S).real))

        trinvS_X = opt_einsum.contract("wab,wba->w", invS, X).real

        # coeff_detV = -n
        # coeff_detX = n-p-1
        log_pdf = (n - p) * logdetX - trinvS_X - n * logdetS

        # log_pdf = coeff_detV * np.log(detV) + coeff_detX * np.log(detX) - trinvV_X
        # # print(-np.sum(log_pdf))
        return -np.sum(log_pdf)

    def log_likelihood_offdiag(self, w, model, omega, omega_fixed, data_, nu, ell):
        """
        Negative of the logarithm of the Variance-Gamma probability density function.
        """
        # print(w)
        spline = model(omega_fixed, w)
        rho = np.clip(spline(omega), -0.98, 0.98)
        _alpha = 1 / (1 - rho**2)
        _beta = rho / (1 - rho**2)
        _lambda = 0.5 * ell * nu
        _gamma2 = _alpha**2 - _beta**2
        _lambda_minus_half = _lambda - 0.5

        # Non sono più sicuro sia sensata questa definizione di z.
        # Non è semplicemtente data_? AH! Forse è la stessa cosa che succede al Chi2, va moltiplicato per il numero di dof. Capire meglio e fare prove.
        z = data_ * nu * ell
        absz = np.abs(z)
        term1 = _lambda * np.log(_gamma2)
        term2 = _lambda_minus_half * np.log(absz)
        term3 = np.log(sp.kv(_lambda_minus_half, _alpha * absz))
        term4 = _beta * z
        term5 = -_lambda_minus_half * np.log(2 * _alpha)
        # log_pdf = _lambda * np.log(_gamma2) + _lambda_minus_half * np.log(absz) + np.log(sp.kv(_lambda_minus_half, _alpha * absz)) + \
        # _beta * z - _lambda_minus_half * np.log(2 * _alpha) # + const
        log_pdf = term1 + term2 + term3 + term4 + term5
        return -np.sum(log_pdf)

    def log_likelihood_diag(self, w, model, omega, omega_fixed, data_, nu, ell):
        """
        Negative of the logarithm of the Chi-squared probability density function.
        """
        spline = model(omega_fixed, w)
        spectrum = 2 * spline(omega)
        # spectrum = np.clip(spline(omega), 1e-60, 1e60)/2
        dof = 2 * (ell - nu + 1)
        z = np.abs(dof * data_ / spectrum)  # This is chi-squared distributed

        logz = np.log(z)
        log_pdf = (2 - dof) * logz + z

        return np.sum(log_pdf)

    def log_likelihood_normal(self, w, omega, omega_fixed, data_, nu, ell):
        """
        Logarithm of the Normal probability density function.
        """
        spline = self.model(omega_fixed, w)
        rho = np.clip(spline(omega), -0.98, 0.98)

        log_pdf = -((data_ - rho) ** 2)
        return np.sum(log_pdf)

    def log_prior_offdiag(self, w):
        """
        Uniform prior for off-diagonal elements.
        """
        if np.all((w >= -1) & (w <= 1)):
            return 1
        else:
            return -np.inf

    def log_prior_diag(self, w):
        """
        Uniform prior for diagonal elements.
        """
        if np.all((w >= 1e-6) & (w <= 1e6)):
            return 1
        else:
            return -np.inf

    def log_posterior_offdiag(self, w, model, omega, omega_fixed, data, nu=6, ell=3):
        """
        Log-posterior for off-diagonal elements.
        """
        return self.log_prior_offdiag(w) + self.log_likelihood_offdiag(
            w, omega, model, omega_fixed, data, nu, ell
        )

    def log_posterior_diag(self, w, model, omega, omega_fixed, data, nu=6, ell=3):
        """
        Log-posterior for diagonal elements.
        """
        return self.log_prior_diag(w) + self.log_likelihood_diag(
            w, model, omega, omega_fixed, data, nu, ell
        )

    def log_posterior_normal(self, w, omega, omega_fixed, data, nu=6, ell=3):
        """
        Log-posterior for normal distribution.
        """
        return self.log_prior_offdiag(w) + self.log_likelihood_normal(
            w, omega, omega_fixed, data, nu, ell
        )


def scale_matrix(model, w, omega, omega_fixed, n):
    """ """
    elements = model(omega_fixed, w)(omega)
    ie = 0
    if elements.dtype == np.complex128:
        L = np.zeros((n, n, omega.shape[0]), dtype=np.complex128)
        for i, j in zip(*np.triu_indices(n)):
            if i == j:
                L[i, j] = elements[:, ie]
                ie += 1
            else:
                L[i, j] = elements[:, ie] + 1j * elements[:, ie + 1]
                ie += 2

        S = np.einsum("jiw,jkw->wik", L.conj(), L)
    else:
        L = np.zeros((n, n, omega.shape[0]))
        for i, j in zip(*np.triu_indices(n)):
            L[i, j] = elements[:, ie]
            ie += 1

        S = np.einsum("jiw,jkw->wik", L, L)
    return S


def normalize_parameters(p, guess):
    mini, maxi = 1.2 * np.abs(guess), -1.2 * np.abs(guess)
    return (p - mini) / (maxi - mini)


def denormalize_parameters(p, guess):
    mini, maxi = 1.2 * np.abs(guess), -1.2 * np.abs(guess)
    return p * (maxi - mini) + mini


# # Methods to perform a bayesian estimation of the transport coefficients

# import numpy as np

# import emcee


# from . import aic
# from .cepstral import dct_coefficients, dct_filter_psd, dct_filter_tau, CepstralFilter, multicomp_cepstral_parameters
# from .tools.filter import runavefilter

# from sportran.utils import log
# from multiprocessing import Pool

# import scipy.special as sp
# from scipy.special import multigammaln
# from scipy.optimize import minimize
# from scipy.linalg import cholesky

# import opt_einsum

# import time

# __all__ = ['MaxLikeFilter']
# EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992   # Euler-Mascheroni constant
# LOG2 = np.log(2)

# class MaxLikeFilter(object):
#     """
#     Maximum-likelihood estimate of the Onsager or transport coefficient.

#     ** INPUT VARIABLES:
#     data            = The noisy data. Either the spectral matrix, or one of its components.
#     model           = Function that models the data (for now only spline)
#     n_parameters    = Number of parameters to be used for the fit
#     n_components    = Number of independent samples the data is generated from.

#     ** INTERNAL VARIABLES:

#     TODO
#     """

#     def __init__(self):
#         # TODO: it's likely better to define some quantities here rather than in self.maxlike
#         log.write_log("MaxLikeFilter")

#     def __repr__(self):
#         msg = 'MaxLikeFilter:\n' #+ \
#             #   '  AIC type  = {:}\n'.format(self.aic_type) + \
#             #   '  AIC min   = {:f}\n'.format(self.aic_min) + \
#             #   '  AIC_Kmin  = {:d}\n'.format(self.aic_Kmin)
#         # if self.cutoffK is not None:
#         #     msg += \
#         #         '  AIC_Kmin_corrfactor = {:f}\n'.format(self.aic_Kmin_corrfactor) + \
#         #         '  cutoffK = (P*-1) = {:d} {:}\n'.format(self.cutoffK, '(manual)' if self.manual_cutoffK_flag else '(auto)') + \
#         #         '  L_0*   = {:15f} +/- {:10f}\n'.format(self.logtau_cutoffK, self.logtau_std_cutoffK) + \
#         #         '  S_0*   = {:15f} +/- {:10f}\n'.format(self.tau_cutoffK, self.tau_std_cutoffK)
#         return msg


#     ################################################

#     def maxlike(
#             self,
#             data = None,
#             model = None,
#             n_parameters = None,
#             likelihood = None,
#             solver = None,
#             mask = None,
#             n_components = None,
#             guess_runave_window = 50,
#             omega_fixed = None,
#             write_log = True
#             ):

#         # Define internal variables for consistent notation
#         assert n_parameters is not None
#         self.n_parameters = n_parameters

#         assert solver is not None

#         if write_log:
#             log.write_log('Maximum-likelihood estimate with {} parameters'.format(n_parameters))

#         nu = 2
#         ell = n_components

#         assert likelihood is not None
#         likelihood = likelihood.lower()
#         if likelihood == 'wishart':
#             log_like = self.log_likelihood_wishart
#         elif likelihood == 'chisquare' or likelihood == 'chisquared':
#             log_like = self.log_likelihood_diag
#         elif likelihood == 'variancegamma' or likelihood == 'variance-gamma':
#             log_like = self.log_likelihood_offdiag
#         else:
#             raise NotImplementedError("Currently implemented likelihoods are: `wishart`, `chisquare`, `variance-gamma`")

#         assert data is not None, "`data` must be provided"
#         if mask is not None:
#             self.data = data[mask]
#         else:
#             self.data = data

#         # Check data is consistently shaped
#         assert len(data.shape) == 1 or len(data.shape) == 3, "`data` should be a 1d array (diagonal or off-diagonal estimates) or a 3d array (Wishart matrix estimate)"
#         if len(data.shape) == 3:
#             assert likelihood == 'wishart', f"Misshaped `data` for {likelihood} likelihood."
#             assert data.shape[0] == data.shape[1] == 2, "`data` for a Wishart esimate must be a (2,2,N) array."
#         else:
#             assert likelihood != 'wishart', f"Misshaped `data` for {likelihood} likelihood."

#         # Frequency
#         omega = np.arange(data.shape[-1])
#         self.omega = omega

#         # Spline nodes
#         if omega_fixed is None:
#             if write_log:
#                 log.write_log("Spline nodes are equispaced from 0 to the Nyquist frequency.")
#             args = np.int32(np.linspace(0, data.shape[-1] - 1, n_parameters, endpoint = True))
#             omega_fixed = omega[args]
#             self.omega_fixed = omega_fixed

#         # Spline model
#         assert model is not None
#         self.model = model

#         # Define initial guess for the optimization
#         guess_data = self.guess_data(data, omega, omega_fixed, ell, window = guess_runave_window)
#         data = np.moveaxis(data, data.shape.index(max(data.shape)), 0)

#         # Minimize the negative log-likelihood
#         res = minimize(fun = log_like,
#                        x0 = guess_data,
#                        args = (model, omega, omega_fixed, data, nu, ell),
#                        method = solver)

#         # Evaluate the covariance of the parameters, if the selected solver allows it
#         try:
#             cov = res.hess_inv
#             if write_log:
#                 log.write_log(f'The {solver} solver features the calculation of the Hessian. The covariance matrix will be estimated through the Laplace approximation.')
#         except:
#             if write_log:
#                 log.write_log(f'The {solver} solver does not feature the calculation of the Hessian. No covariance matrix will be output.')
#             cov = None

#         self.parameters_mean = res.x
#         if cov is not None:
#             self.parameters_std = np.sqrt(cov.diagonal())

#         self.optimizer_res = res
#         self.log_likelihood_value = -log_like(res.x, model, omega, omega_fixed, data, nu, ell)

#   ################################################

#     # Helper functions

#     def guess_data(self, data, omega, omega_fixed, ell, window = 10, loglike = 'wishart'):
#         '''
#         Moving average of the input data as initial guess for the parameter esimation
#         '''

#         shape = data.shape

#         try:
#             guess_data = np.array([runavefilter(c, window) for c in data.reshape(-1, shape[-1])]).reshape(shape)
#         except:
#             log.write_log(f'Guessing data with a running average with a {window} large window failed. Window changed to 10.')
#             window = 10
#             guess_data = np.array([runavefilter(c, window) for c in data.reshape(-1, shape[-1])]).reshape(shape)

#         guess_data = np.array([guess_data[...,j] for j in [np.argmin(np.abs(omega - omega_fixed[i])) for i in range(len(omega_fixed))]])

#         if loglike == 'wishart':
#             guess_data = np.array([cholesky(g, lower = False) for g in guess_data]) / np.sqrt(ell)
#             guess_data = np.array([guess_data[:, 0, 0], guess_data[:, 0, 1], guess_data[:, 1, 1]]).reshape(-1)

#         return guess_data

#     def log_likelihood_wishart(self, w, model, omega, omega_fixed, data_, nu, ell):
#         '''
#         Logarithm of the Wishart probability density function.
#         '''
#         n = ell
#         p = nu
#         multig = multigammaln(0.5*n, p)

#         # Compute scale matrix from the model (symmetrize to ensure positive definiteness)
#         spline = model(omega_fixed, w)
#         V = spline(omega)
#         V = opt_einsum.contract('wba,wbc->wac', V, V) / n # equiv to V.T@V for each frequency

#         # The argument of the PDF is the data
#         X = data_

#         # Determinant of X
#         a, b, d = X[...,0,0], X[...,0,1], X[...,1,1]
#         detX = a*d - b**2

#         # Determinant and inverse of V
#         a, b, d = V[...,0,0], V[...,0,1], V[...,1,1]
#         invV = (1/(a*d - b**2)*np.array([[d, -b],[-b, a]])).transpose(2,0,1)
#         detV = a*d - b**2

#         # Trace of the matrix product between the inverse of V and X
#         trinvV_X = opt_einsum.contract('wab,wba->w', invV, X)

#         # if detV.min() < 0 or detX.min() < 0:
#         #         print(detV.min(), detX.min())

#         # Sum pieces of the log-likelihood
#         log_pdf = - (0.5*(-n*p*LOG2 - n*np.log(detV) + (n-p-1)*np.log(detX) - trinvV_X) - multig)

#         return np.sum(log_pdf)

#     def log_likelihood_offdiag(self, w, omega, omega_fixed, data_, nu, ell):
#         '''
#         Logarithm of the Variance-Gamma probability density function.
#         '''
#         spline = self.model(omega_fixed, w)
#         rho = np.clip(spline(omega), -0.98, 0.98)
#         _alpha = 1/(1-rho**2)
#         _beta = rho/(1-rho**2)
#         _lambda = 0.5*ell*nu
#         _gamma2 = _alpha**2 - _beta**2
#         _lambda_minus_half = _lambda-0.5

#         # Data is distributed according to a Variance-Gamma distribution with parameters (notation as in Wikipedia):
#         # mu = 0; alpha = 1/(1-rho**2); beta = rho/(1-rho**2); lambda = ell*nu/2
#         # Its expectation value is ell*nu*rho
#         z = data_*ell*nu
#         absz = np.abs(z)
#         # z = data
#         log_pdf = _lambda*np.log(_gamma2) + _lambda_minus_half*np.log(absz) + np.log(sp.kv(_lambda_minus_half, _alpha*absz)) + \
#             _beta*z - 0.5*np.log(np.pi) - np.log(sp.gamma(_lambda)) - _lambda_minus_half*np.log(2*_alpha)

#         res = np.sum(log_pdf)
#         return res

#     def log_likelihood_diag(self, w, omega, omega_fixed, data_,  ell):
#         spline = self.model(omega_fixed, w)
#         rho = np.clip(spline(omega), 1e-6, 1e6)

#         # Data is distributed according to a Chi-squared distribution with parameters (notation as in Wikipedia):
#         # Its expectation value is ell*rho
#         z = data_*ell/rho
#         absz = np.abs(z)
#         # z = data
#         log_pdf = (ell / 2 - 1)*np.log(absz) - absz/2 - np.log(rho)

#         res = np.sum(log_pdf)
#         return res

#     def log_likelihood_normal(self, w, omega, omega_fixed, data_, nu, ell):
#         spline = self.model(omega_fixed, w)
#         rho = np.clip(spline(omega), -0.98, 0.98)

#         log_pdf = -(data_ - rho)**2
#         return np.sum(log_pdf)

#     # The log-prior function
#     def log_prior_offdiag(self, w):
#         # Uniform prior
#         if np.all((w>=-1)&(w<=1)):
#             return 1
#         else:
#             return -np.inf

#     # The log-prior function
#     def log_prior_diag(self, w):
#         # Uniform prior
#         if np.all((w>=1e-6)&(w<=1e6)):
#             return 1
#         else:
#             return -np.inf

#     # The log-posterior function
#     def log_posterior_offdiag(self, w, omega, omega_fixed, data, nu = 6, ell = 3):
#         return self.log_prior_offdiag(w) + self.log_likelihood_offdiag(w, omega, omega_fixed, data, nu, ell)

#     # The log-posterior function
#     def log_posterior_diag(self, w, omega, omega_fixed, data, ell = 3):
#         return self.log_prior_diag(w) + self.log_likelihood_diag(w, omega, omega_fixed, data, ell)

#     # The log-posterior function
#     def log_posterior_normal(self, w, omega, omega_fixed, data, nu=6, ell=3):
#         return self.log_prior_offdiag(w) + self.log_likelihood_normal(w, omega, omega_fixed, data, nu, ell)
