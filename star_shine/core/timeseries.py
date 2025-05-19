"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb

from star_shine.core import periodogram as pdg, frequency_sets as frs
from star_shine.core import utility as ut


@nb.njit(cache=True)
def mask_timestamps(time, stamps):
    """Mask out everything except the parts between the given timestamps

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    stamps: numpy.ndarray[Any, dtype[float]]
        Pairs of timestamps

    Returns
    -------
    numpy.ndarray[bool]
        Boolean mask that is True between the stamps
    """
    mask = np.zeros(len(time), dtype=np.bool_)
    for ts in stamps:
        mask = mask | ((time >= ts[0]) & (time <= ts[-1]))

    return mask


@nb.njit(cache=True)
def mark_gaps(time, min_gap=1.):
    """Mark gaps in a series of time points.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    min_gap: float, optional
        Minimum width for a gap (in time units).

    Returns
    -------
    gaps: numpy.ndarray[Any, dtype[float]]
        Gap timestamps in pairs.
    """
    # mark the gaps
    t_sorted = np.sort(time)
    t_diff = t_sorted[1:] - t_sorted[:-1]  # np.diff(a)
    gaps = (t_diff > min_gap)

    # get the timestamps
    t_left = t_sorted[:-1][gaps]
    t_right = t_sorted[1:][gaps]
    gaps = np.column_stack((t_left, t_right))

    return gaps


def refine_orbital_period(p_orb, time, f_n):
    """Find the most likely eclipse period from a sinusoid model

    Parameters
    ----------
    p_orb: float
        Orbital period of the eclipsing binary in days
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves

    Returns
    -------
    float
        Orbital period of the eclipsing binary in days

    Notes
    -----
    Uses the sum of distances between the harmonics and their
    theoretical positions to refine the orbital period.
    
    Period precision is 0.00001, but accuracy is a bit lower.
    
    Same refine algorithm as used in find_orbital_period
    """
    freq_res = 1.5 / np.ptp(time)  # Rayleigh criterion
    f_nyquist = 1 / (2 * np.min(np.diff(time)))  # nyquist frequency

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    p_orb = 1 / f_refine[mask_peak][i_min_dist]

    return p_orb


def find_orbital_period(time, flux, f_n):
    """Find the most likely eclipse period from a sinusoid model

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves

    Returns
    -------
    tuple
        A tuple containing the following elements:
        p_orb: float
            Orbital period of the eclipsing binary in days
        multiple: float
            Multiple of the initial period that was chosen

    Notes
    -----
    Uses a combination of phase dispersion minimisation and
    Lomb-Scargle periodogram (see Saha & Vivas 2017), and some
    refining steps to get the best period.
    
    Also tests various multiples of the period.
    Precision is 0.00001 (one part in one-hundred-thousand).
    (accuracy might be slightly lower)
    """
    freq_res = 1.5 / np.ptp(time)  # Rayleigh criterion
    f_nyquist = 1 / (2 * np.min(np.diff(time)))  # nyquist frequency

    # first to get a global minimum do combined PDM and LS, at select frequencies
    periods, phase_disp = pdg.phase_dispersion_minimisation(time, flux, f_n, local=False)
    ampls, _ = pdg.scargle_ampl_phase(time, flux, 1 / periods)
    psi_measure = ampls / phase_disp

    # also check the number of harmonics at each period and include into best f
    n_harm, completeness, distance = frs.harmonic_series_length(1 / periods, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness

    # select the best period, refine it and check double P
    p_orb = periods[np.argmax(psi_h_measure)]

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    p_orb = 1 / f_refine[mask_peak][i_min_dist]

    # reduce the search space by taking limits in the distance metric
    f_left = f_refine[mask_peak][:i_min_dist]
    f_right = f_refine[mask_peak][i_min_dist:]
    d_left = distance_r[mask_peak][:i_min_dist]
    d_right = distance_r[mask_peak][i_min_dist:]
    d_max = np.max(distance_r)
    if np.any(d_left > d_max / 2):
        f_l_bound = f_left[d_left > d_max / 2][-1]
    else:
        f_l_bound = f_refine[mask_peak][0]
    if np.any(d_right > d_max / 2):
        f_r_bound = f_right[d_right > d_max / 2][0]
    else:
        f_r_bound = f_refine[mask_peak][-1]
    bound_interval = f_r_bound - f_l_bound

    # decide on the multiple of the period
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    completeness_p = (len(harmonics) / (f_nyquist // (1 / p_orb)))
    completeness_p_l = (len(harmonics[harmonic_n <= 15]) / (f_nyquist // (1 / p_orb)))

    # check these (commonly missed) multiples
    n_multiply = np.array([1/2, 2, 3, 4, 5])
    p_multiples = p_orb * n_multiply
    n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(1 / p_multiples, f_n, freq_res, f_nyquist)
    h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # if there are very high numbers, add double that fraction for testing
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    if np.any(test_frac[2:] > 3):
        n_multiply = np.append(n_multiply, [2 * n_multiply[2:][test_frac[2:] > 3]])
        p_multiples = p_orb * n_multiply
        n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(1 / p_multiples, f_n, freq_res,
                                                                                f_nyquist)
        h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # compute diagnostic fractions that need to meet some threshold
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    compl_frac = completeness_r_m / completeness_p

    # doubling the period may be done if the harmonic filling factor below f_16 is very high
    f_cut = np.max(f_n[harmonics][harmonic_n <= 15])
    f_n_c = f_n[f_n <= f_cut]
    n_harm_r_2, completeness_r_2, distance_r_2 = frs.harmonic_series_length(1 / p_multiples, f_n_c, freq_res, f_nyquist)
    compl_frac_2 = completeness_r_2[1] / completeness_p_l

    # empirically determined thresholds for the various measures
    minimal_frac = 1.1
    minimal_compl_frac = 0.85
    minimal_frac_low = 0.95
    minimal_compl_frac_low = 0.95

    # test conditions
    test_condition = (test_frac > minimal_frac)
    compl_condition = (compl_frac > minimal_compl_frac)
    test_condition_2 = (test_frac[1] > minimal_frac_low)
    compl_condition_2 = (compl_frac_2 > minimal_compl_frac_low)
    if np.any(test_condition & compl_condition) | (test_condition_2 & compl_condition_2):
        if np.any(test_condition & compl_condition):
            i_best = np.argmax(test_frac[compl_condition])
            p_orb = p_multiples[compl_condition][i_best]
        else:
            p_orb = 2 * p_orb

        # make new bounds for refining
        f_left_b = 1 / p_orb - (bound_interval / 2)
        f_right_b = 1 / p_orb + (bound_interval / 2)

        # refine by using a dense sampling and the harmonic distances
        f_refine_2 = np.arange(f_left_b, f_right_b, 0.00001 / p_orb)
        n_harm_r2, completeness_r2, distance_r2 = frs.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
        h_measure_2 = n_harm_r2 * completeness_r2  # compute h_measure for constraining a domain
        mask_peak = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
        i_min_dist = np.argmin(distance_r2[mask_peak])
        p_orb = 1 / f_refine_2[mask_peak][i_min_dist]

    return p_orb


@nb.njit(cache=True)
def formal_uncertainties_linear(time, residuals, i_chunks):
    """Calculates the corrected uncorrelated (formal) uncertainties for the
    parameters constant and slope.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        sigma_const: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the constant for each sector
        sigma_slope: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the slope for each sector

    Notes
    -----
    Errors in const and slope:
    https://pages.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
    """
    n_param = 2

    # linear regression uncertainties
    sigma_const = np.zeros(len(i_chunks))
    sigma_slope = np.zeros(len(i_chunks))
    for i, s in enumerate(i_chunks):
        len_t = len(time[s[0]:s[1]])
        n_data = len(residuals[s[0]:s[1]])  # same as len_t, but just for the sake of clarity
        n_dof = n_data - n_param  # degrees of freedom

        # standard deviation of the residuals but per sector
        std = ut.std_unb(residuals[s[0]:s[1]], n_dof)

        # some sums for the uncertainty formulae
        sum_t = 0
        for t in time[s[0]:s[1]]:
            sum_t += t
        ss_xx = 0
        for t in time[s[0]:s[1]]:
            ss_xx += (t - sum_t / len_t)**2
        sigma_const[i] = std * np.sqrt(1 / n_data + (sum_t / len_t)**2 / ss_xx)
        sigma_slope[i] = std / np.sqrt(ss_xx)

    return sigma_const, sigma_slope


@nb.njit(cache=True)
def formal_uncertainties(time, residuals, flux_err, a_n, i_chunks):
    """Calculates the corrected uncorrelated (formal) uncertainties for the extracted
    parameters (constant, slope, frequencies, amplitudes and phases).

    Harmonics are not taken into account.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        sigma_const: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the constant for each sector
        sigma_slope: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the slope for each sector
        sigma_f: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the frequency for each sine wave
        sigma_a: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the amplitude for each sine wave (these are identical)
        sigma_ph: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the phase for each sine wave
    
    Notes
    -----
    As in Aerts 2021, https://ui.adsabs.harvard.edu/abs/2021RvMP...93a5001A/abstract
    The sigma value in the formulae is approximated by taking the maximum of the
    standard deviation of the residuals and the standard error of the minimum data error.
    Errors in const and slope:
    https://pages.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
    """
    n_data = len(residuals)
    n_param = ut.n_parameters(len(i_chunks), len(a_n), 0)  # number of parameters in the model
    n_dof = max(n_data - n_param, 1)  # degrees of freedom

    # calculate the standard deviation of the residuals
    std = ut.std_unb(residuals, n_dof)

    # calculate the standard error based on the smallest data error
    ste = np.median(flux_err) / np.sqrt(n_data)

    # take the maximum of the standard deviation and standard error as sigma N
    sigma_n = max(std, ste)

    # calculate the D factor (square root of the average number of consecutive data points of the same sign)
    positive = (residuals > 0).astype(np.int_)
    indices = np.arange(n_data)
    zero_crossings = indices[1:][np.abs(positive[1:] - positive[:-1]).astype(np.bool_)]
    sss_i = np.concatenate((np.array([0]), zero_crossings, np.array([n_data])))  # same-sign sequence indices
    d_factor = np.sqrt(np.mean(np.diff(sss_i)))

    # uncertainty formulae for sinusoids
    sigma_f = d_factor * sigma_n * np.sqrt(6 / n_data) / (np.pi * a_n * np.ptp(time))
    sigma_a = d_factor * sigma_n * np.sqrt(2 / n_data)
    sigma_ph = d_factor * sigma_n * np.sqrt(2 / n_data) / a_n  # times 2 pi w.r.t. the paper

    # make an array of sigma_a (these are the same)
    sigma_a = np.full(len(a_n), sigma_a)

    # linear regression uncertainties
    sigma_const, sigma_slope = formal_uncertainties_linear(time, residuals, i_chunks)

    return sigma_const, sigma_slope, sigma_f, sigma_a, sigma_ph


def linear_regression_uncertainty_ephem(time, p_orb, sigma_t=1):
    """Calculates the linear regression errors on period and t_zero

    Parameters
    ---------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    p_orb: float
        Orbital period in days.
    sigma_t: float
        Error in the individual time measurements.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        p_err: float
            Error in the period.
        t_err: float
            Error in t_zero.
        p_t_corr: float
            Covariance between the period and t_zero.

    Notes
    -----
    The number of eclipses, computed from the period and
    time base, is taken to be a contiguous set.
    var_matrix:
    [[std[0]**2          , std[0]*std[1]*corr],
     [std[0]*std[1]*corr,           std[1]**2]]
    """
    # number of observed eclipses (technically contiguous)
    n = int(abs(np.ptp(time) // p_orb)) + 1

    # the arrays
    x = np.arange(n, dtype=int)  # 'time' points
    y = np.ones(n, dtype=int)  # 'positive measurement'

    # remove points in gaps
    gaps = mark_gaps(time, min_gap=1.)
    mask = mask_timestamps(x * p_orb, gaps)  # convert x to time domain
    x = x[~mask] - n//2  # also centre the time for minimal correlation
    y = y[~mask]

    # M
    matrix = np.column_stack((x, y))

    # M^-1
    matrix_inv = np.linalg.pinv(matrix)  # inverse (of a general matrix)

    # M^-1 S M^-1^T, S unit matrix times some sigma (no covariance in the data)
    var_matrix = matrix_inv @ matrix_inv.T
    var_matrix = var_matrix * sigma_t ** 2

    # errors in the period and t_zero
    p_err = np.sqrt(var_matrix[0, 0])
    t_err = np.sqrt(var_matrix[1, 1])
    p_t_corr = var_matrix[0, 1] / (t_err * p_err)  # or [1, 0]

    return p_err, t_err, p_t_corr
