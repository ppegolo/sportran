# Methods to perform a bayesian estimation of the transport coefficients

import numpy as np
import emcee
import scipy.special as sp
from . import aic
from .cepstral import dct_coefficients, dct_filter_psd, dct_filter_tau, CepstralFilter, multicomp_cepstral_parameters
from .tools.filter import runavefilter
from sportran.utils import log
from multiprocessing import Pool
from scipy.special import multigammaln
from scipy.optimize import minimize
import opt_einsum

import time

__all__ = ['MaxLikeFilter']
EULER_GAMMA = 0.57721566490153286060651209008240243104215933593992   # Euler-Mascheroni constant
LOG2 = np.log(2)

class MaxLikeFilter(object):
    """
    BAYESIAN ANALYSIS based filtering.

    ** INPUT VARIABLES:
    spectrum        = the original periodogram (if single-component) of spectral matrix (if multi-component)
    is_offdiag      = If True, estimate the off-diagonal matrix element of the spectral matrix (default = True)
    is_diag         = If True, estimate the diagonal matrix elements of the spectral matrix (default = False)
    model           = the function that models the data (for now only spline)
    n_parameters    = the number of parameters to be used for the fit

    ** INTERNAL VARIABLES:
    samplelogpsd  = the original sample log-PSD - logpsd_THEORY_mean

    logpsdK  = the cepstrum of the data, \\hat{C}_n (i.e. the DCT of samplelogpsd)
    aic_min  = minimum value of the AIC
    aic_Kmin = cutoffK that minimizes the AIC
    aic_Kmin_corrfactor = aic_Kmin cutoff correction factor (default: 1.0)
    cutoffK  = (P*-1) = cutoff used to compute logtau and logpsd (by default = aic_Kmin * aic_Kmin_corrfactor)
    manual_cutoffK_flag = True if cutoffK was manually specified, False if aic_Kmin is being used

    logtau          = filtered log(tau) as a function of cutoffK, L_0(P*-1)
    logtau_cutoffK  = filtered log(tau) at cutoffK, L*_0
    logtau_var_cutoffK = theoretical L*_0 variance
    logtau_std_cutoffK = theoretical L*_0 standard deviation
    logpsd          = filtered log-PSD at cutoffK

    tau          = filtered tau as a function of cutoffK, S_0(P*-1)
    tau_cutoffK  = filtered tau at cutoffK, S*_0
    tau_var_cutoffK = theoretical S*_0 variance
    tau_std_cutoffK = theoretical S*_0 standard deviation
    psd          = filtered PSD at the specified cutoffK

    p_aic... = Bayesian AIC weighting stuff
    """

    def __init__(self, spectrum, model, n_parameters, n_components, 
                 mask = None):

        if not isinstance(spectrum, np.ndarray):
            raise TypeError('spectrum should be an object of type numpy.ndarray')
        if spectrum.shape[0] != 2 or spectrum.shape[1] != 2:
            raise TypeError('spectrum should be a 2x2xN numpy.ndarray')
        
        self.spectrum = spectrum/n_components
        self.model = model
        self.mask = mask
        self.n_components = n_components
        self.n_parameters = n_parameters

    def __repr__(self):
        msg = 'BayesFilter:\n' #+ \
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
            n_parameters = None,
            likelihood = 'wishart',
            solver = 'BFGS',
            mask = None
            ):

        if n_parameters is None:
            n_parameters = self.n_parameters

        nu = 2
        ell = self.n_components
        
        # Define noisy data
        if mask is not None:
            noisy_data = np.concatenate([self.spectrum.real[i, j][mask] for (i, j) in [[0,0], [0,1], [1,1]]])
        else:
            noisy_data = np.concatenate([self.spectrum.real[i, j] for (i, j) in [[0,0], [0,1], [1,1]]])

        # Define initial guess for the optimization
        try:
            guess_data = runavefilter(noisy_data, 100)
        except:
            guess_data = runavefilter(noisy_data, 10)

        args = np.int32(np.linspace(0, len(noisy_data) - 1, n_parameters, endpoint = True))

        log.write_log('Maximum-likelihood estimate with {} parameters'.format(n_parameters))

        p0 = guess_data
        
        omega = np.arange(noisy_data.size)
        omega_fixed = omega[args]

        self.omega = omega
        self.omega_fixed = omega_fixed

        if likelihood.lower() == 'wishart':
            log_like = self.log_likelihood_wishart
        
        mode = self.model
        res = minimize(fun = lambda w, mode, omega, omega_fixed, noisy_data, nu, ell: -log_like(w, mode, omega, omega_fixed, noisy_data, nu, ell),
                       x0 = p0,  
                       args = (self.model, omega, omega_fixed, noisy_data, nu, ell),
                       method = solver)
        
        params = res.x
        try:
            cov = res.hess_inv
        except:
            log.write_log(f'The selected solver {solver} does not feature the calculation of the Hessian. No covariance matrix will be output.')
            cov = None
        
        self.parameters_mean = res.x
        # self.parameters_args = args
        if cov is not None:
            self.parameters_std = cov.diagonal()**0.5
        self.noisy_data = noisy_data

  ################################################

    # Helper functions

    def log_likelihood_wishart(w, model, omega, omega_fixed, data_, nu, ell):
        '''
        Logarithm of the Wishart probability density function.
        '''        
        n = ell
        p = 2
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
        log_pdf = 0.5*(-n*p*LOG2 - n*np.log(detV) + (n-p-1)*np.log(detX) - trinvV_X) - multig
        
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