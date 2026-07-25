# -*- coding: utf-8 -*-

from typing import Any

import numpy as np
import scipy.stats as stats


class MissingDataError(ValueError):
    pass


def _bit_length_26(n: int) -> int:
    return n.bit_length()


def _acf_q_stat(acf: np.ndarray, nobs: int) -> tuple[np.ndarray, np.ndarray]:
    h = acf.shape[0]
    qstat: np.ndarray = (
        nobs * (nobs + 2) * np.cumsum(acf**2 / (nobs - np.arange(1, h + 1)))
    )
    pvalue = 1 - stats.chi2.cdf(qstat, df=np.arange(1, h + 1))
    return qstat, pvalue


def integrate_acf(acf: np.ndarray) -> np.ndarray:
    """Return cumulative integral of ``acf`` versus upper integration limit.

    Supports multi-component ``(N, N_COMP)`` arrays. Trapezoidal integration is used:
    ``tau[i] = trapz(acf[0:i+1])``.
    """

    N = acf.shape[0]
    tau = np.zeros(acf.shape)
    for i in range(1, N):
        tau[i] = tau[i - 1] + 0.5 * acf[i - 1] + 0.5 * acf[i]
    return tau


################################################################################
## functions copied from statstools.tsa


def acovf(
    x: np.ndarray,
    unbiased: bool = False,
    demean: bool = True,
    fft: bool = False,
    missing: str = "none",
) -> np.ndarray:
    """
    Autocovariance for 1D

    Parameters
    ----------
    x : array
        Time series data. Must be 1d.
    unbiased : bool
        If True, then denominators is n-k, otherwise n
    demean : bool
        If True, then subtract the mean x from each element of x
    fft : bool
        If True, use FFT convolution.  This method should be preferred for long time
        series.
    missing : str
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how the NaNs
        are to be treated.

    Returns
    -------
    acovf : array
        autocovariance function

    References
    -----------
    Parzen, E. (1963), "On spectral analysis with missing observations and amplitude
    modulation", Sankhya: The Indian Journal of Statistics, Series A.
    """
    x = np.squeeze(np.asarray(x))
    if x.ndim > 1:
        raise ValueError("x must be 1d. Got %d dims." % x.ndim)

    missing = missing.lower()
    if missing not in ["none", "raise", "conservative", "drop"]:
        raise ValueError("missing option %s not understood" % missing)
    if missing == "none":
        deal_with_masked = False
    else:
        deal_with_masked = has_missing(x)
    if deal_with_masked:
        if missing == "raise":
            raise MissingDataError("NaNs were encountered in the data")
        notmask_bool = ~np.isnan(x)  # bool
        if missing == "conservative":
            x[~notmask_bool] = 0
        else:  #'drop'
            x = x[notmask_bool]  # copies non-missing
        notmask_int = notmask_bool.astype(int)  # int

    if demean and deal_with_masked:
        # whether 'drop' or 'conservative':
        xo = x - x.sum() / notmask_int.sum()
        if missing == "conservative":
            xo[~notmask_bool] = 0
    elif demean:
        xo = x - x.mean()
    else:
        xo = x

    n = len(x)
    if unbiased and deal_with_masked and missing == "conservative":
        d = np.correlate(notmask_int, notmask_int, "full")
    elif unbiased:
        xi = np.arange(1, n + 1)
        d = np.hstack((xi, xi[:-1][::-1]))
    elif deal_with_masked:  # biased and NaNs given and ('drop' or 'conservative')
        d = notmask_int.sum() * np.ones(2 * n - 1)
    else:  # biased and no NaNs or missing=='none'
        d = n * np.ones(2 * n - 1)

    if fft:
        nobs = len(xo)
        n = _next_regular(2 * nobs + 1)
        Frf = np.fft.fft(xo, n=n)
        acov = np.fft.ifft(Frf * np.conjugate(Frf))[:nobs] / d[nobs - 1 :]
        acov = acov.real
    else:
        acov = (np.correlate(xo, xo, "full") / d)[n - 1 :]

    if deal_with_masked and missing == "conservative":
        # restore data for the user
        x[~notmask_bool] = np.nan

    return acov


# see for example http://www.itl.nist.gov/div898/handbook/eda/section3/autocopl.htm
def acf(
    x: np.ndarray,
    unbiased: bool = False,
    nlags: int = 40,
    qstat: bool = False,
    fft: bool = False,
    alpha: float | None = None,
    missing: str = "none",
) -> Any:
    """
    Autocorrelation function for 1d arrays.

    Parameters
    ----------
    x : array
       Time series data
    unbiased : bool
       If True, then denominators for autocovariance are n-k, otherwise n
    nlags: int, optional
        Number of lags to return autocorrelation for.
    qstat : bool, optional
        If True, returns the Ljung-Box q statistic for each autocorrelation coefficient.
        See q_stat for more information.
    fft : bool, optional
        If True, computes the ACF via FFT.
    alpha : scalar, optional
        If a number is given, the confidence intervals for the given level are returned.
        For instance if alpha=.05, 95 % confidence intervals are returned where the
        standard deviation is computed according to Bartlett\'s formula.
    missing : str, optional
        A string in ['none', 'raise', 'conservative', 'drop'] specifying how the NaNs
        are to be treated.

    Returns
    -------
    acf : array
        autocorrelation function
    confint : array, optional
        Confidence intervals for the ACF. Returned if confint is not None.
    qstat : array, optional
        The Ljung-Box Q-Statistic.  Returned if q_stat is True.
    pvalues : array, optional
        The p-values associated with the Q-statistics.  Returned if q_stat is True.

    Notes
    -----
    The acf at lag 0 (ie., 1) is returned.

    This is based np.correlate which does full convolution. For very long time series it
    is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted but the
    autocorrelation is not an unbiased estimtor.

    References
    ----------
    Parzen, E. (1963), "On spectral analysis with missing observations and amplitude
    modulation", Sankhya: The Indian Journal of Statistics, Series A.

    """
    nobs = len(x)  # should this shrink for missing='drop' and NaNs in x?
    avf = acovf(x, unbiased=unbiased, demean=True, fft=fft, missing=missing)
    acf = avf[: nlags + 1] / avf[0]
    if not (qstat or alpha):
        return acf
    if alpha is not None:
        varacf = np.ones(nlags + 1) / nobs
        varacf[0] = 0
        varacf[1] = 1.0 / nobs
        varacf[2:] *= 1 + 2 * np.cumsum(acf[1:-1] ** 2)
        interval = stats.norm.ppf(1 - alpha / 2.0) * np.sqrt(varacf)
        confint = np.array(list(zip(acf - interval, acf + interval)))
        if not qstat:
            return acf, confint
    if qstat:
        _qstat, pvalue = _acf_q_stat(acf[1:], nobs=nobs)
        if alpha is not None:
            return acf, confint, _qstat, pvalue
        else:
            return acf, _qstat, pvalue


def ccovf(
    x: np.ndarray, y: np.ndarray, unbiased: bool = True, demean: bool = True
) -> np.ndarray:
    """crosscovariance for 1D

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators is n-k, otherwise n

    Returns
    -------
    ccovf : array
        autocovariance function

    Notes
    -----
    This uses np.correlate which does full convolution. For very long time series it is
    recommended to use fft convolution instead.
    """
    n = len(x)
    if demean:
        xo = x - x.mean()
        yo = y - y.mean()
    else:
        xo = x
        yo = y
    if unbiased:
        xi = np.ones(n)
        d: np.ndarray | int = np.correlate(xi, xi, "full")
    else:
        d = n
    return (np.correlate(xo, yo, "full") / d)[n - 1 :]


def ccf(x: np.ndarray, y: np.ndarray, unbiased: bool = True) -> np.ndarray:
    """cross-correlation function for 1d

    Parameters
    ----------
    x, y : arrays
       time series data
    unbiased : boolean
       if True, then denominators for autocovariance is n-k, otherwise n

    Returns
    -------
    ccf : array
        cross-correlation function of x and y

    Notes
    -----
    This is based np.correlate which does full convolution. For very long time series it
    is recommended to use fft convolution instead.

    If unbiased is true, the denominator for the autocovariance is adjusted but the
    autocorrelation is not an unbiased estimtor.

    """
    cvf = ccovf(x, y, unbiased=unbiased, demean=True)
    return cvf / (np.std(x) * np.std(y))


def has_missing(data: np.ndarray) -> bool:
    """
    Returns True if 'data' contains missing entries, otherwise False
    """
    return np.isnan(np.sum(data))


def _next_regular(target: int) -> int:
    """
    Find the next regular number greater than or equal to target. Regular numbers are
    composites of the prime factors 2, 3, and 5. Also known as 5-smooth numbers or
    Hamming numbers, these are the optimal size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target - 1)):
        return target

    match: int = 2 * target  # will be replaced by any smaller valid candidate
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float (quotient =
            # ceil(target / p35))
            quotient = -(-target // p35)
            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2 ** ((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2 ** _bit_length_26(quotient - 1)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match
