# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.fft import dct
from sportran.utils import log

def dct_filter_psd(y, K=None):
    # Copied from cepstral.py to avoid circular import
    # K=P*-1 is the maximum coefficient summed (c_k = 0 for k > K)
    if (K >= y.size):
        log.write_log('! Warning:  dct_filter_psd K value ({:}) out of range.'.format(K))
        return np.full(y.size, np.NaN)
    yk = dct(y, type=1)
    if K is not None:
        yk[K + 1:] = 0.
    ynew = dct(yk, type=1) / (y.size - 1) * 0.5
    return ynew

def dct_coefficients(y):
    # Copied from cepstral.py to avoid circular import
    """Compute the normalized Discrete Cosine Transform coefficients of y.
        yk = 0.5 * DCT(y) / (N-1)"""
    yk = dct(y, type=1) / (y.size - 1) * 0.5   # normalization
    return yk

def compute_bias(logpsd, theory_mean):
    # I am almost sure `theory_mean` should actually be zero, since the "trivial bias" is already included 
    # in the definition of the estimator of log(kappa).
    bias_ = np.copy(dct_coefficients(logpsd))[1:]
    bias_[:-1] *= 2
    bias = theory_mean - np.flip(np.cumsum(np.flip(bias_)))
    return bias

########################################################################################################################
# Minimize the Mean Square Error MSE[L_0^*] = bias[L_0^*]**2 + Var[L_0^*]

def dct_MSE(logpsd, theory_var=None, theory_mean=None, dt = 1, window_freq_THz = None, is_self_consistent = True, initial_P = None, is_savgol = False):
    
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    N = 2 * (logpsd.size - 1)
    
    
    # Compute mean and variance of the cepstral coefficients
    eulergamma = 0.57721566490153286060651209008240243104215933593992

    if theory_var is None:
        theory_var = np.pi**2 / 6. / N   # otherwise
    else:
        theory_var = theory_var[1]

    if theory_mean is None:
        theory_mean = -eulergamma
    else:
        theory_mean = theory_mean[1]

    pstar = np.arange(1, N // 2 + 1)
    var = theory_var * (4*pstar-2)
    MSE_bias = (N//2 - pstar) * N*theory_var  # This is the bias on the estimator of the bias (theory_var is already divided by N)

    if is_self_consistent:

        # 1. Select an initial value for P
        if initial_P is None:
            # Choose it of the order of 10~15 ps
            initial_P = np.min([np.random.randint(int(10000.0/dt), int(20000.0/dt)), N//2])
            #initial_P = N//2-1

        K_new = initial_P - 1
        K = -1
        log.write_log('scMMSE: initial K = {:d}\n'.format(K_new))

        biases = []
        max_num = 1000
        isc = 0
        while K_new != K and isc <= max_num:

            K = K_new
            # 2. Compute the filtered logspectrum
            filtered_logpsd = dct_filter_psd(logpsd, K)
            
            # 3. Compute the bias according to filtered_logpsd
            bias = compute_bias(filtered_logpsd, 0)
            #bias = compute_bias(filtered_logpsd, theory_mean)
            biases.append(bias)
            
            # 4. Minimize the MSE
            MSE = bias**2 + var - MSE_bias          # This is an unbiased estimator of the MSE
            MSE_min = np.min(MSE)
            MSE_Kmin = np.argmin(MSE)

            log.write_log('scMMSE: step {isc:>4d}: MSE_Kmin = {kmin:>{nnum:d}d}, MSE_min = {mmse:.2f}\n'.format(isc = isc+1, kmin = MSE_Kmin, nnum = len(str(logpsd.size)), mmse = MSE_min))

            K_new = MSE_Kmin
            isc += 1
        
        return MSE, bias, var, [MSE_bias, biases] 

    else:
        # 1. Provide a gross and uncontrolled estimate of the smooth PSD via a moving average

        df = 1/dt/N*1000 #THz
        if window_freq_THz is None:
            window_freq_THz = 1.0
       
        # The Savitzki-Golay filter is a piecewise polynomial fitting of the data
        if is_savgol:
            
            window = 2*(int(window_freq_THz / df) // 2) + 1

            log.write_log('Savitzki-Golay smoothing window width = {:.2f} THz = {:d} points\n'.format(window_freq_THz, window))
            if window < 10:
                log.write_log('The window is likely too narrow. Please set a larger value of `window_freq_THz`.\n')
        
            import warnings
            from scipy.signal import savgol_filter
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    smoothed_logpsd = savgol_filter(logpsd, window_length = window, polyorder = 3)
                except np.RankWarning:
                    smoothed_logpsd = pd.Series(logpsd).rolling(window = window).mean().shift(1-window).fillna(method = 'ffill').to_numpy()

        else:

            from scipy.signal import resample
            from scipy.interpolate import interp1d
            from scipy.signal import savgol_filter

            window = 2*(int(window_freq_THz / df) // 2)
            f = df*np.arange(logpsd.size)
            smoothed_logpsd, resampled_f = resample(logpsd, logpsd.size // window, t = f)

            df_ = df*logpsd.size/smoothed_logpsd.size
            window = 2*(int(window_freq_THz / df_) // 2) + 1
            if window <= 3:
                window = 5
            log.write_log('Window length = {}\n'.format(window))
            smoothed_logpsd = savgol_filter(smoothed_logpsd, window_length = window, polyorder = 3)

            logpsd_func = interp1d(resampled_f, smoothed_logpsd, kind = 'quadratic', fill_value = 'extrapolate')
            smoothed_logpsd = logpsd_func(f)
        
        # 2. Compute the gross cepstrum
        smoothed_cepstrum = dct(smoothed_logpsd, type = 1) / N

        # 3. Compute the bias
        bias = compute_bias(smoothed_logpsd, 0)
        #bias = compute_bias(smoothed_logpsd, theory_mean)
        MSE = bias**2 + var - MSE_bias          # This is an unbiased estimator of the MSE
        
        return MSE, bias, var, [smoothed_logpsd, smoothed_cepstrum] #, smoothed_cepstrum_2] #fit_variables, bias, var, bias_orig

########################################################################################################################

def dct_AIC(yk, theory_var=None):
    """AIC[K] = sum_{k>K} c_k^2/theory_var + 2*(K+1)
    Assumiamo di tenere tutti i k <= K."""
    aic = np.zeros(yk.size)
    if theory_var is None:
        N = 2 * (yk.size - 1)
        theory_var1 = np.pi**2 / 3. / N   # k = {0, N/2}
        theory_var2 = np.pi**2 / 6. / N   # otherwise
        for K in range(yk.size - 1):
            # aic[K] = np.sum(yk[K+1:]**2)/theory_var + 2.*K
            aic[K] = ((1. / theory_var2) * np.sum(yk[K + 1:-1]**2) + (1. / theory_var1) * yk[-1]**2) + 2. * (K + 1)
        aic[-1] = 2. * yk.size
    else:
        aic[-1] = 0.   # + (2*(yk.size+1))
        for K in range(yk.size - 2, -1, -1):   # N-2, N-3, ..., 0
            aic[K] = aic[K + 1] + yk[K + 1]**2 / theory_var[K + 1]
        aic = aic + 2. * (np.arange(yk.size) + 1)
    return aic


def dct_AICc(yk, theory_var=None):
    """AICc[K] = AIC[K] + 2*(K+1)*(K+2)/(NF-K-2)
    Assumiamo di tenere tutti i k <= K."""
    aic = dct_AIC(yk, theory_var)
    KK = np.arange(yk.size - 2)
    aic[:-2] = aic[:-2] + 2. * (KK + 1) * (KK + 2) / (yk.size - KK - 2.)
    aic[-2] = aic[-3]   # not defined
    aic[-1] = aic[-3]   # not defined
    return aic


def dct_aic_ab(yk, theory_var, A=1.0, B=2.0):
    """AIC[K] = sum_{k>K} c_k^2/theory_var + 2*K
    Assumiamo di tenere tutti i k <= K."""
    aic = np.zeros(yk.size)
    aic[-1] = 0.
    for K in range(yk.size - 2, -1, -1):   # N-2, N-3, ..., 0
        aic[K] = aic[K + 1] + yk[K + 1]**2 / theory_var[K + 1]
    aic = A * aic + B * (np.arange(yk.size) + 1)
    return aic


################################################################################
####  Bayesian method
################################################################################
#
# Produce_density takes as inputs the following numpy arrays:
#
#                INPUT :
#
# - aic: The vector with the estimate of the Akaike information aic(k), with k ranging from k_beg:k_max.
#         I think it is important to have values till k_max but the first k_beg can be different from one ;
# - sigma, mean : for the k_beg:k_max, in the same order, the mean and sigmas of the estimate
#                 of the transport coefficient ;
# - method : a string equal to 'one' or 'two' with two different methods of inizializing p[k], the estimate probability
#            of a given k. I hope in real cases the two methods should be equivalent.
#
#                RETURNS:
#
# The probability p[ik], a grid and a density for the probability of the transport coefficient on that grid, obtained as:
# density[igrid] \sim Sum_ik p[ik] * N[ mean[ik],sigma[ik] ].
# All pi factors are never inserted and taken care I hope by final normalization.
#
# The functions produce_p and produce_p_density distinguish the two conceptual steps of producing p and using this quantity
# to provide the final estimate on the transport coefficient
#


def produce_p(aic, method='ba', force_normalize=False):
    k0 = np.argmin(aic)
    kM = len(aic)
    p = np.zeros(kM)

    if (method == 'min'):
        # just the min(aic)
        p[k0] = 1.0

    elif (method == 'baroni'):
        for ik in range(kM):
            delta_aic = aic[ik] - aic[k0]
            GAMMA = 3.9215536345675050924567623117545   # (epsilon/sigma)^2
            p[ik] = np.exp(-0.25 * np.log(1. + GAMMA) * delta_aic)

    elif ((method == 'burnham-anderson') or (method == 'ba')):
        delta_aic = aic - aic[k0]
        p = np.exp(-0.5 * delta_aic)
        p = p / np.sum(p)

    elif ((method == 'burnham-anderson2') or (method == 'ba2')):
        delta_aic = aic - aic[k0]
        p = np.exp(-delta_aic)
        p = p / np.sum(p)

    elif (method == 'two'):
        for ik in range(kM):
            delta_aic = aic[ik] - aic[k0]
            p[ik] = np.exp(-delta_aic**2 / (2.0 * (kM - k0))) / np.sqrt(kM - k0)

    elif (method == 'four'):
        for ik in range(kM):
            delta_aic = aic[ik] - aic[k0]
            p[ik] = np.exp(-delta_aic**2 / (2.0 * np.abs(ik - k0)))
        p[k0] = 1.0

    else:
        raise KeyError('P_AIC METHOD not valid.')
    #p[ik] = np.exp(- delta_aic ** 2 / ( 2.0 * ( kM - ik) ) ) / np.sqrt(kM - ik)
    #p[ik] = np.exp(-delta_aic ** 2 / ( 2.0 * np.abs(ik - k0) )) / np.sqrt(np.abs(ik - k0))

    # normalize p
    if force_normalize:
        integral = np.trapz(p)
        for ik in range(kM):
            p[ik] = p[ik] / integral

    # checks
    if any(np.isnan(p)):
        raise Warning('WARNING: P contains NaN.')
    if (np.min(p < 0)):
        raise Warning('WARNING: P < 0.')
    return p


def produce_p_density(p, sigma, mean, grid=None, grid_size=1000):
    kM = len(mean)
    dm = np.min(mean)
    dM = np.max(mean)
    argdm = np.argmin(mean)
    argdM = np.argmax(mean)
    sigmam = sigma[argdm]
    sigmaM = sigma[argdM]
    if grid is None:
        return_grid = True
        # generate grid with grid_size points
        delta = ((dM + 5. * sigmaM) - (dm - 5. * sigmam)) / float(grid_size - 1)
        grid = np.linspace(dm - 5. * sigmam, dM + 5. * sigmaM, grid_size)
    else:
        return_grid = False
        delta = grid[1] - grid[0]
    density = np.zeros(len(grid))
    for ik in range(kM):
        density = density + p[ik] * np.exp(-(grid - mean[ik])**2 / (2.0 * (sigma[ik]**2))) / sigma[ik]
    somma = np.trapz(density) * delta
    density = density / somma
    if return_grid:
        return density, grid
    else:
        return density


def grid_statistics(grid, density, grid2=None):
    """Compute distribution mean and std.
      media   = \\sum_i (density[i] * grid[i])
      std     = sqrt( \\sum_i (density[i] * grid[i]^2) - media^2 )
       oppure = sqrt( \\sum_i (density[i] * grid2[i])  - media^2 )"""
    somma = np.sum(density)
    media = np.dot(density, grid) / somma
    if grid2 is None:
        var = np.dot(density, grid**2) / somma - media**2
    else:
        var = np.dot(density, grid2) / somma - media**2
    std = np.sqrt(var)
    return media, std
