# Methods to perform a bayesian estimation of the transport coefficients

import numpy as np

import emcee


from . import aic
from .cepstral import dct_coefficients, dct_filter_psd, dct_filter_tau, CepstralFilter, multicomp_cepstral_parameters
from .tools.filter import runavefilter

from sportran.utils import log
from multiprocessing import Pool

import scipy.special as sp
from scipy.special import multigammaln
from scipy.optimize import minimize
from scipy.linalg import cholesky

import opt_einsum

import time

__all__ = ['MaxLikeFilter']
EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992   # Euler-Mascheroni constant
LOG2 = np.log(2)

class MaxLikeFilter(object):
    """
    Maximum-likelihood estimate of the Onsager or transport coefficient.

    ** INPUT VARIABLES:
    data            = The noisy data. Either the spectral matrix, or one of its components.
    model           = Function that models the data (for now only spline)
    n_parameters    = Number of parameters to be used for the fit
    n_components    = Number of independent samples the data is generated from.

    ** INTERNAL VARIABLES:

    TODO
    """

    def __init__(self):
        # TODO: it's likely better to define some quantities here rather than in self.maxlike
        log.write_log("MaxLikeFilter")
 
    def __repr__(self):
        msg = 'MaxLikeFilter:\n' #+ \
            #   '  AIC type  = {:}\n'.format(self.aic_type) + \
            #   '  AIC min   = {:f}\n'.format(self.aic_min) + \
            #   '  AIC_Kmin  = {:d}\n'.format(self.aic_Kmin)
        # if self.cutoffK is not None:
        #     msg += \
        #         '  AIC_Kmin_corrfactor = {:f}\n'.format(self.aic_Kmin_corrfactor) + \
        #         '  cutoffK = (P*-1) = {:d} {:}\n'.format(self.cutoffK, '(manual)' if self.manual_cutoffK_flag else '(auto)') + \
        #         '  L_0*   = {:15f} +/- {:10f}\n'.format(self.logtau_cutoffK, self.logtau_std_cutoffK) + \
        #         '  S_0*   = {:15f} +/- {:10f}\n'.format(self.tau_cutoffK, self.tau_std_cutoffK)
        return msg


    ################################################

    def maxlike(
            self,
            data = None,
            model = None,
            n_parameters = None,
            likelihood = None,
            solver = None,
            mask = None,
            n_components = None,
            guess_runave_window = 50,
            omega_fixed = None,
            write_log = True
            ):
        
        # Define internal variables for consistent notation 
        assert n_parameters is not None
        self.n_parameters = n_parameters

        assert solver is not None

        if write_log:
            log.write_log('Maximum-likelihood estimate with {} parameters'.format(n_parameters))

        nu = 2
        ell = n_components

        assert likelihood is not None
        likelihood = likelihood.lower()
        if likelihood == 'wishart':
            log_like = self.log_likelihood_wishart
        elif likelihood == 'chisquare' or likelihood == 'chisquared':
            log_like = self.log_likelihood_diag
        elif likelihood == 'variancegamma' or likelihood == 'variance-gamma':
            log_like = self.log_likelihood_offdiag
        else:
            raise NotImplementedError("Currently implemented likelihoods are: `wishart`, `chisquare`, `variance-gamma`")
        
        assert data is not None, "`data` must be provided"
        if mask is not None:
            self.data = data[mask]
        else:
            self.data = data
        
        # Check data is consistently shaped
        assert len(data.shape) == 1 or len(data.shape) == 3, "`data` should be a 1d array (diagonal or off-diagonal estimates) or a 3d array (Wishart matrix estimate)"
        if len(data.shape) == 3:
            assert likelihood == 'wishart', f"Misshaped `data` for {likelihood} likelihood."
            assert data.shape[0] == data.shape[1] == 2, "`data` for a Wishart esimate must be a (2,2,N) array."
        else:
            assert likelihood != 'wishart', f"Misshaped `data` for {likelihood} likelihood."

        # Frequency
        omega = np.arange(data.shape[-1])
        self.omega = omega

        # Spline nodes
        if omega_fixed is None:
            if write_log:
                log.write_log("Spline nodes are equispaced from 0 to the Nyquist frequency.")
            args = np.int32(np.linspace(0, data.shape[-1] - 1, n_parameters, endpoint = True))
            omega_fixed = omega[args]
            self.omega_fixed = omega_fixed

        # Spline model
        assert model is not None
        self.model = model

        # Define initial guess for the optimization
        guess_data = self.guess_data(data, omega, omega_fixed, ell, window = guess_runave_window)
        data = np.moveaxis(data, data.shape.index(max(data.shape)), 0)

        # Minimize the negative log-likelihood
        res = minimize(fun = log_like,
                       x0 = guess_data,  
                       args = (model, omega, omega_fixed, data, nu, ell),
                       method = solver)
        
        # Evaluate the covariance of the parameters, if the selected solver allows it
        try:
            cov = res.hess_inv
            if write_log:
                log.write_log(f'The {solver} solver features the calculation of the Hessian. The covariance matrix will be estimated through the Laplace approximation.')
        except:
            if write_log:
                log.write_log(f'The {solver} solver does not feature the calculation of the Hessian. No covariance matrix will be output.')
            cov = None
        
        self.parameters_mean = res.x
        if cov is not None:
            self.parameters_std = np.sqrt(cov.diagonal())

        self.optimizer_res = res
        self.log_likelihood_value = -log_like(res.x, model, omega, omega_fixed, data, nu, ell)

  ################################################

    # Helper functions

    def guess_data(self, data, omega, omega_fixed, ell, window = 10, loglike = 'wishart'):
        '''
        Moving average of the input data as initial guess for the parameter esimation
        '''

        shape = data.shape
        
        try:
            guess_data = np.array([runavefilter(c, window) for c in data.reshape(-1, shape[-1])]).reshape(shape)
        except:
            log.write_log(f'Guessing data with a running average with a {window} large window failed. Window changed to 10.')
            window = 10
            guess_data = np.array([runavefilter(c, window) for c in data.reshape(-1, shape[-1])]).reshape(shape)
        
        guess_data = np.array([guess_data[...,j] for j in [np.argmin(np.abs(omega - omega_fixed[i])) for i in range(len(omega_fixed))]])

        if loglike == 'wishart':
            guess_data = np.array([cholesky(g, lower = False) for g in guess_data]) / np.sqrt(ell)
            guess_data = np.array([guess_data[:, 0, 0], guess_data[:, 0, 1], guess_data[:, 1, 1]]).reshape(-1)
        
        return guess_data
    
    def log_likelihood_wishart(self, w, model, omega, omega_fixed, data_, nu, ell):
        '''
        Logarithm of the Wishart probability density function.
        '''        
        n = ell
        p = nu
        multig = multigammaln(0.5*n, p)

        # Compute scale matrix from the model (symmetrize to ensure positive definiteness)
        spline = model(omega_fixed, w)
        V = spline(omega)
        V = opt_einsum.contract('wba,wbc->wac', V, V) / n # equiv to V.T@V for each frequency

        # The argument of the PDF is the data
        X = data_ 
        
        # Determinant of X
        a, b, d = X[...,0,0], X[...,0,1], X[...,1,1]
        detX = a*d - b**2
        
        # Determinant and inverse of V
        a, b, d = V[...,0,0], V[...,0,1], V[...,1,1]
        invV = (1/(a*d - b**2)*np.array([[d, -b],[-b, a]])).transpose(2,0,1)
        detV = a*d - b**2

        # Trace of the matrix product between the inverse of V and X
        trinvV_X = opt_einsum.contract('wab,wba->w', invV, X)

        # if detV.min() < 0 or detX.min() < 0:
        #         print(detV.min(), detX.min())

        # Sum pieces of the log-likelihood
        log_pdf = - (0.5*(-n*p*LOG2 - n*np.log(detV) + (n-p-1)*np.log(detX) - trinvV_X) - multig)
        
        return np.sum(log_pdf)

    def log_likelihood_offdiag(self, w, omega, omega_fixed, data_, nu, ell):
        '''
        Logarithm of the Variance-Gamma probability density function.
        '''
        spline = self.model(omega_fixed, w)
        rho = np.clip(spline(omega), -0.98, 0.98)
        _alpha = 1/(1-rho**2)
        _beta = rho/(1-rho**2)
        _lambda = 0.5*ell*nu
        _gamma2 = _alpha**2 - _beta**2
        _lambda_minus_half = _lambda-0.5
        
        # Data is distributed according to a Variance-Gamma distribution with parameters (notation as in Wikipedia):
        # mu = 0; alpha = 1/(1-rho**2); beta = rho/(1-rho**2); lambda = ell*nu/2
        # Its expectation value is ell*nu*rho
        z = data_*ell*nu
        absz = np.abs(z)
        # z = data 
        log_pdf = _lambda*np.log(_gamma2) + _lambda_minus_half*np.log(absz) + np.log(sp.kv(_lambda_minus_half, _alpha*absz)) + \
            _beta*z - 0.5*np.log(np.pi) - np.log(sp.gamma(_lambda)) - _lambda_minus_half*np.log(2*_alpha)

        res = np.sum(log_pdf)
        return res

    def log_likelihood_diag(self, w, omega, omega_fixed, data_,  ell):
        spline = self.model(omega_fixed, w)
        rho = np.clip(spline(omega), 1e-6, 1e6)

        # Data is distributed according to a Chi-squared distribution with parameters (notation as in Wikipedia):
        # Its expectation value is ell*rho
        z = data_*ell/rho
        absz = np.abs(z)
        # z = data 
        log_pdf = (ell / 2 - 1)*np.log(absz) - absz/2 - np.log(rho)

        res = np.sum(log_pdf)
        return res

    def log_likelihood_normal(self, w, omega, omega_fixed, data_, nu, ell):
        spline = self.model(omega_fixed, w)
        rho = np.clip(spline(omega), -0.98, 0.98)

        log_pdf = -(data_ - rho)**2
        return np.sum(log_pdf)

    # The log-prior function
    def log_prior_offdiag(self, w):
        # Uniform prior
        if np.all((w>=-1)&(w<=1)):
            return 1
        else: 
            return -np.inf

    # The log-prior function
    def log_prior_diag(self, w):
        # Uniform prior
        if np.all((w>=1e-6)&(w<=1e6)):
            return 1
        else:
            return -np.inf

    # The log-posterior function
    def log_posterior_offdiag(self, w, omega, omega_fixed, data, nu = 6, ell = 3):
        return self.log_prior_offdiag(w) + self.log_likelihood_offdiag(w, omega, omega_fixed, data, nu, ell)

    # The log-posterior function
    def log_posterior_diag(self, w, omega, omega_fixed, data, ell = 3):
        return self.log_prior_diag(w) + self.log_likelihood_diag(w, omega, omega_fixed, data, ell)

    # The log-posterior function
    def log_posterior_normal(self, w, omega, omega_fixed, data, nu=6, ell=3):
        return self.log_prior_offdiag(w) + self.log_likelihood_normal(w, omega, omega_fixed, data, nu, ell)