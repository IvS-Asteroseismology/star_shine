"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains functions that evaluate certain configuration attributes that depend on data properties.

Code written by: Luc IJspeert
"""
import numpy as np

from star_shine.core.utility import config


def signal_to_noise_threshold(time):
    """Determine the signal-to-noise threshold for accepting frequencies based on the number of points

    Based on Baran & Koen 2021, eq 6. (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)

    snr_thr = 1.201 * np.sqrt(1.05 * np.log(len(time)) + 7.184)
    Plus 0.25 in case of gaps > 27 days

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.

    Returns
    -------
    float
        signal-to-noise threshold for this data set.
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


def frequency_resolution(time, factor=1.5):
    """Calculate the frequency resolution of a time series

    Equation: factor / T, where T is the total time base of observations.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    factor: float, optional
        Number that multiplies the resolution (1/T). Common choices are 1, 1.5 (conservative), 2.5 (very conservative).

    Returns
    -------
    float
        Frequency resolution of the time series
    """

    f_res = factor / np.ptp(time)

    return f_res


def frequency_lower_threshold(time, factor=0.01):
    """Calculate the frequency resolution of a time series

    Equation: factor / T, where T is the total time base of observations.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    factor: float, optional
        Number that multiplies the resolution (1/T).

    Returns
    -------
    float
        Frequency resolution of the time series
    """

    f_min = factor / np.ptp(time)

    return f_min


def frequency_upper_threshold(time, func='min'):
    """Determines the maximum frequency for extraction and periodograms

    If set in configuration, the user defined value is used. Otherwise, the Nyquist frequency is calculated
    based on the time stamps, and using the desired built-in function.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    func: str
        Function name for the calculation of the Nyquist frequency. Choose from 'min', 'median', or 'rigorous'.


    Returns
    -------
    float
        Maximum frequency to be calculated
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
        f_max = 1 / (2 * np.min(time[1:] - time[:-1]))

    return f_max
