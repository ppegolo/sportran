# -*- coding: utf-8 -*-

import numpy as np

########################################################################################################################
# Minimize the Mean Square Error MSE[L_0^*] = bias[L_0^*]**2 + Var[L_0^*]

def dct_MSE(ck, theory_var=None, theory_mean=None, init_pstar=None, decay = 'cosexp', decay_pars = None, acf = None, dt = None):
    
    from scipy.special import zeta
    from scipy.optimize import curve_fit
    from scipy.stats import linregress

    eulergamma = 0.57721566490153286060651209008240243104215933593992
    N = 2 * (ck.size - 1)

    if theory_var is None:
        theory_var = np.pi**2 / 6. / N   # otherwise
    else:
        theory_var = theory_var[1]

    if theory_mean is None:
        theory_mean = -eulergamma
    else:
        theory_mean = theory_mean[1]

    C0 = ck[0]

    if init_pstar is None:
        init_pstar = 100
    
    pstar = np.arange(1, ck.size -1)
    print(pstar)
    var = theory_var * (4*pstar-3)

    if decay == 'power law':
        where = slice(0, init_pstar)
        n = np.arange(len(ck))
        xf = np.log(n+1)[where]
        yf = np.log(np.abs(ck))[where]
        slope, intercept, r, p, se = linregress(xf, yf)
        print('B = {:.2e} +\- {:.2e}'.format(slope, se))
        print('r-value = {:.2e}'.format(r))
        B = -slope

        bias = theory_mean - C0*((1+N/2)**slope + 2*zeta(B, 1+pstar) - 2*zeta(B, 1+N/2))
        
        fit_variables = {'slope': slope, 'intercept': intercept}
    
    elif decay == 'exp':
        def expd(t, R0, tau):
            return R0*np.exp(-np.abs(t)/tau)
        def ck_exp(n, beta):
            from numpy import pi, exp
            n = np.asarray(n)
            output = np.full(n.shape, np.inf)
            n_ = n[n != 0]
            output[n != 0] = exp(-beta*n_)/n_ + (-1)**(n_+1)*2/n_**2/(pi**2+beta**2)
            return output

        if decay_pars is not None:
            R0 = decay_pars['R0']
            tau = decay_pars['tau']
            
            beta  = dt/tau

            n = np.arange(len(ck))
            print(R0, tau, beta)
            to_sum = ck_exp(n, beta)
            bias = theory_mean - np.flip(np.cumsum(np.flip(to_sum))) - ck_exp(N//2, beta)

            fit_variables = {'R0': R0, 'tau': tau}

        elif acf is not None and dt is not None:
            time = np.arange(len(acf))*dt

            try:
                popt, pcov = curve_fit(expd, time, acf, p0 = [acf[0], 1000.]) # TODO: find a way to make a good guess on the parameters (Hilbert transform?)
            except:
                raise ValueError('Parameters of the `exp` method cannot be estimated.')
            R0 = popt[0]
            tau = popt[1]

            beta  = dt/tau

            n = np.arange(len(ck))
            print(R0, tau, beta)
            to_sum = ck_exp(n, beta)
            bias = theory_mean - np.flip(np.cumsum(np.flip(to_sum))) - ck_exp(N//2, beta)

            fit_variables = {'R0': R0, 'tau': tau}

        else:
            raise ValueError('`acf` and `dt` must be provided to estimate the cepstral coefficients decay with the `exp` method, if `decay_pars` is not given.')
            


    elif decay == 'cosexp':
        def cosexp(t, R0, tau, f0):
            return R0*np.cos(2*np.pi*f0*t)*np.exp(-np.abs(t)/tau)
        # Analytic formula that works when the correlation function is cosexp
        def ck_cosexp(n, eps, tau, f0):
            from numpy import exp, pi, cos, sqrt
            n_ = np.asarray(n)
            output = np.full(n_.shape, np.inf)
            n = n_[n_ != 0]
            output[n_ != 0] = (2*exp(-n*eps/tau) * cos(2*pi*f0*n*eps) - exp(-n*eps/tau*sqrt(1 + (2*pi*tau*f0)**2)))/n
            return output
        # Overkill upper bound
        def ck_cosexp_ub_overkill(n, eps, tau, f0):
            from numpy import exp, pi, cos, sqrt
            n_ = np.asarray(n)
            output = np.full(n_.shape, np.inf)
            n = n_[n_ != 0]
            output[n_ != 0] = 3*exp(-n*eps/tau)/n
            return output
        
        if decay_pars is not None:
            dt_ps = dt/1000
            R0 = decay_pars['R0']
            tau = decay_pars['tau']
            f0 = decay_pars['f0']
            
            n = np.arange(ck.size - 2, -1, -1)
            print(n.size)
            print(n)
            to_sum = ck_cosexp(n, dt_ps, tau, f0)
            bias_orig = theory_mean - np.flip(2*(np.cumsum(to_sum))) - ck_cosexp(N//2, dt_ps, tau, f0)

            # Upper bound on bias (for large enough P...)
            #def sum1(eps, tau, f0, N):
            #    from numpy import exp, cos, sin, pi, sqrt
            #    return np.array([sqrt(np.sum([2*exp(-n*eps/tau)*cos(2*pi*f0*eps*(n-P))/n for n in range(P, N//2)])**2 + \
            #                     np.sum([2*exp(-n*eps/tau)*sin(2*pi*f0*eps*(n-P))/n for n in range(P, N//2)])**2)   \
            #                     for P in range(1, N//2)])
            def sum2(n, eps, tau, f0, N):
                from numpy import cumsum, flip, exp, sqrt, pi
                to_sum = -exp(-n*eps/tau*sqrt(1+(2*pi*f0*tau)**2))/n
                return flip(cumsum(to_sum))

            def CP(N, eps, f0):
                from numpy import cos, pi
                n = np.arange(1, N//2)
                return cos(2*pi*f0*eps*n)
            def SP(N, eps, f0):
                from numpy import sin, pi
                n = np.arange(1, N//2)
                return sin(2*pi*f0*eps*n)
            def SigmaCP(N, eps, tau, f0):
                from numpy import exp, pi, cos
                n = np.arange(N//2-1, 0, -1)
                to_sum = exp(-eps*n/tau)/n*cos(2*pi*f0*eps*n)
                return np.flip(np.cumsum(to_sum)) 
            def SigmaSP(N, eps, tau, f0):
                from numpy import exp, pi, sin
                n = np.arange(N//2-1, 0, -1)
                to_sum = exp(-eps*n/tau)/n*sin(2*pi*f0*eps*n)
                return np.flip(np.cumsum(to_sum)) 
            
            
            n = np.arange(N//2-1, 0, -1)

            #bias = theory_mean -2 * (sum1(dt_ps, tau, f0, N) + sum2(n, dt_ps, tau, f0, N)) - ck_cosexp(N//2, dt_ps, tau, f0)
            
            bias = theory_mean -2 * (CP(N, dt_ps, f0)*SigmaSP(N, dt_ps, tau, f0) - SP(N, dt_ps, f0)*SigmaCP(N, dt_ps, tau, f0))  +\
                    sum2(n, dt_ps, tau, f0, N)) - ck_cosexp(N//2, dt_ps, tau, f0)

            fit_variables = {'R0': R0, 'tau': tau, 'f0': f0}
        
        elif acf is not None and dt is not None:
            
            dt_ps = dt/1000
            time = np.arange(len(acf))*dt_ps
            time = time[time<=100]

            from scipy.signal import hilbert
            analytic_signal = hilbert(acf)
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)/dt_ps)

            envf = amplitude_envelope[:len(time)]
            print (time.shape, envf.shape)

            def exp_decay(t, tau):
                return np.exp(-t/tau)
            popt, pcov = curve_fit(exp_decay, time, envf/envf[0], p0 = [1])
            tau = popt[0]
            f0 = instantaneous_frequency.mean()
            print('Guess for tau: {:.2e} +/- {:.2e} ps'.format(tau, pcov[0,0]))
            print('Guess for f0: {:.2e} THz'.format(f0))

            try:
                acff = acf[:len(time)]
                popt, pcov = curve_fit(cosexp, time, acff, p0 = [acff[0], tau, f0])
            except:
                raise ValueError('Parameters of the `cosexp` method cannot be estimated.')
            R0 = popt[0]
            tau = popt[1]
            f0 = popt[2]
            fit_variables = {'R0': R0, 'tau': tau, 'f0': f0}

            print('Estimated parameters:')
            print('R0  = {:.2e} +/- {:.2e}'.format(R0,  np.sqrt(pcov[0,0])))
            print('tau = {:.2f} +/- {:.2f} ps'.format(tau, np.sqrt(pcov[1,1])))
            print('f0  = {:.2f} +/- {:.2f} THz'.format(f0,  np.sqrt(pcov[2,2])))
            
            #n = np.arange(ck.size - 2, -1, -1)
            #print(n.size)
            #print(n)
            #to_sum = ck_cosexp(n, dt_ps, tau, f0)
            #bias = theory_mean - np.flip(2*(np.cumsum(to_sum))) - ck_cosexp(N//2, dt_ps, tau, f0)

            pstar = np.arange(ck.size-2, 0, -1) # ck.size = N/2, so ck.size-2 lets the range start from N/2-1
            bias = theory_mean - 2*np.flip(np.cumsum(ck_cosexp_ub(pstar, dt_ps, tau, f0))) - 2*ck_cosexp_ub(ck.size, dt_ps, tau, f0)

        else:
            raise ValueError('`acf` and `dt` must be provided to estimate the cepstral coefficients decay with the `cosexp` method.')
        


    
    else:
        raise ValueError('The variable `decay` must either be equal to "cosexp" (default) or "power law".')

    return bias**2 + var, fit_variables, bias, var

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
