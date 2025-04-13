"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains utility functions for data processing, unit conversions.

Code written by: Luc IJspeert
"""
import numpy as np
import numba as nb

from star_shine.config.helpers import get_config


# load configuration
config = get_config()


@nb.njit(cache=True)
def float_to_str(x, dec=2):
    """Convert float to string for Numba up to some decimal place
    
    Parameters
    ----------
    x: float
        Value to convert
    dec: int
        Number of decimals (be careful with large numbers here)
    
    Returns
    -------
    str
        String with the value x
    """
    x_round = np.round(x, dec)
    x_int = int(x_round)
    x_dec = int(np.abs(np.round(x_round - x_int, dec)) * 10**dec)
    s = str(x_int) + '.' + str(x_dec).zfill(dec)

    return s


@nb.njit(cache=True)
def weighted_mean(x, w):
    """Weighted mean since Numba doesn't support numpy.average
    
    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the mean over
    w: numpy.ndarray[Any, dtype[float]]
        Weights corresponding to each value
    
    Returns
    -------
    float
        Mean of x weighted by w
    """
    w_mean = np.sum(x * w) / np.sum(w)

    return w_mean


@nb.njit(cache=True)
def std_unb(x, n):
    """Unbiased standard deviation

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the std over
    n: int
        Number of degrees of freedom

    Returns
    -------
    float
        Unbiased standard deviation
    """
    residuals = x - np.mean(x)

    # tested to be faster in numba than np.sum(x**2)
    sum_r_2 = 0
    for r in residuals:
        sum_r_2 += r**2
    std = np.sqrt(sum_r_2 / n)  # unbiased standard deviation of the residuals

    return std


@nb.njit(cache=True)
def decimal_figures(x, n_sf):
    """Determine the number of decimal figures to print given a target
    number of significant figures
    
    Parameters
    ----------
    x: float
        Value to determine the number of decimals for
    n_sf: int
        Number of significant figures to compute
    
    Returns
    -------
    int
        Number of decimal places to round to
    """
    if x != 0:
        decimals = (n_sf - 1) - int(np.floor(np.log10(abs(x))))
    else:
        decimals = 1

    return decimals


def signal_to_noise_threshold(time):
    """Determine the signal-to-noise threshold for accepting frequencies
    based on the number of points

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.

    Returns
    -------
    float
        signal-to-noise threshold for this data set.

    Notes
    -----
    Based on Baran & Koen 2021, eq 6.
    (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)

    snr_thr = 1.201 * np.sqrt(1.05 * np.log(len(time)) + 7.184)

    Plus 0.25 in case of gaps > 27 days
    """
    # if user defined (unequal to -1), return the configured number
    if config.snr_thr != -1:
        return config.snr_thr

    # equation 6 from Baran & Koen 2021
    snr_thr = 1.201 * np.sqrt(1.05 * np.log(len(time)) + 7.184)

    # increase threshold by 0.25 if gaps longer than 27 days
    if np.any((time[1:] - time[:-1]) > 27):
        snr_thr += 0.25

    # round to two decimals
    snr_thr = np.round(snr_thr, 2)

    return snr_thr


def frequency_upper_threshold(time, func='min'):
    """Determines the maximum frequency for extraction and periodograms

    If set in configuration, the user defined value is used.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    func: str
        Function name for the calculation of the Nyquist frequency. Choose from 'min', 'median', or 'rigorous'.
    """
    # if user defined (unequal to -1), return the configured number
    if config.f_max != -1:
        return config.f_max

    # calculate the Nyquist frequency with the specified approach
    if func == 'min':
        f_max = 1 / (2 * np.min(time[1:] - time[:-1]))
    elif func == 'median':
        f_max = 1 / (2 * np.median(time[1:] - time[:-1]))
    elif func == 'rigorous':
        # implement the rigorous way of calculating Nyquist
        f_max = 1 / (2 * np.min(time[1:] - time[:-1]))
    else:
        # func name not recognised
        f_max = 0

    return f_max


def group_frequencies_for_fit(a_n, g_min=20, g_max=25):
    """Groups frequencies into sets of g_min to g_max for multi-sine fitting
    
    Parameters
    ----------
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    g_min: int
        Minimum group size
    g_max: int
        Maximum group size (g_max > g_min)
    
    Returns
    -------
    list[numpy.ndarray[Any, dtype[int]]]
        List of sets of indices indicating the groups
    
    Notes
    -----
    To make the task of fitting more manageable, the free parameters are binned into groups,
    in which the remaining parameters are kept fixed. Frequencies of similar amplitude are
    grouped together, and the group cut-off is determined by the biggest gaps in amplitude
    between frequencies, but group size is always kept between g_min and g_max. g_min < g_max.
    The idea of using amplitudes is that frequencies of similar amplitude have a similar
    amount of influence on each other.
    """
    # keep track of which freqs have been used with the sorted indices
    not_used = np.argsort(a_n)[::-1]
    groups = []
    while len(not_used) > 0:
        if len(not_used) > g_min + 1:
            a_diff = np.diff(a_n[not_used[g_min:g_max + 1]])
            i_max = np.argmin(a_diff)  # the diffs are negative so this is max absolute difference
            i_group = g_min + i_max + 1
            group_i = not_used[:i_group]
        else:
            group_i = np.copy(not_used)
            i_group = len(not_used)
        not_used = np.delete(not_used, np.arange(i_group))
        groups.append(group_i)

    return groups


@nb.njit(cache=True)
def correct_for_crowdsap(flux, crowdsap, i_chunks):
    """Correct the flux for flux contribution of a third source
    
    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series corrected for
        contaminating light
    
    Notes
    -----
    Uses the parameter CROWDSAP included with some TESS data.
    flux_corrected = (flux - (1 - crowdsap)) / crowdsap
    where all quantities are median-normalised, including the result.
    This corresponds to subtracting a fraction of (1 - crowdsap) of third light
    from the (non-median-normalised) flux measurements.
    """
    cor_flux = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        cor_flux[s[0]:s[1]] = (flux[s[0]:s[1]] - 1 + crowd) / crowd

    return cor_flux


@nb.njit(cache=True)
def model_crowdsap(flux, crowdsap, i_chunks):
    """Incorporate flux contribution of a third source into the flux

    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model of the flux incorporating light contamination

    Notes
    -----
    Does the opposite as correct_for_crowdsap, to be able to model the effect of
    third light to some degree (can only achieve an upper bound on CROWDSAP).
    """
    model = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        model[s[0]:s[1]] = flux[s[0]:s[1]] * crowd + 1 - crowd

    return model
