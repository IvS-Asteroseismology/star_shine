"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb

from star_shine.core import periodogram as pdg
from star_shine.core import goodness_of_fit as gof
from star_shine.core import fitting as fit
from star_shine.core import analysis as anf
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
    n_harm_r, completeness_r, distance_r = anf.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
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
    n_harm, completeness, distance = anf.harmonic_series_length(1 / periods, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness

    # select the best period, refine it and check double P
    p_orb = periods[np.argmax(psi_h_measure)]

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = anf.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
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
    harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    completeness_p = (len(harmonics) / (f_nyquist // (1 / p_orb)))
    completeness_p_l = (len(harmonics[harmonic_n <= 15]) / (f_nyquist // (1 / p_orb)))

    # check these (commonly missed) multiples
    n_multiply = np.array([1/2, 2, 3, 4, 5])
    p_multiples = p_orb * n_multiply
    n_harm_r_m, completeness_r_m, distance_r_m = anf.harmonic_series_length(1 / p_multiples, f_n, freq_res, f_nyquist)
    h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # if there are very high numbers, add double that fraction for testing
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    if np.any(test_frac[2:] > 3):
        n_multiply = np.append(n_multiply, [2 * n_multiply[2:][test_frac[2:] > 3]])
        p_multiples = p_orb * n_multiply
        n_harm_r_m, completeness_r_m, distance_r_m = anf.harmonic_series_length(1 / p_multiples, f_n, freq_res, f_nyquist)
        h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # compute diagnostic fractions that need to meet some threshold
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    compl_frac = completeness_r_m / completeness_p

    # doubling the period may be done if the harmonic filling factor below f_16 is very high
    f_cut = np.max(f_n[harmonics][harmonic_n <= 15])
    f_n_c = f_n[f_n <= f_cut]
    n_harm_r_2, completeness_r_2, distance_r_2 = anf.harmonic_series_length(1 / p_multiples, f_n_c, freq_res, f_nyquist)
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
        n_harm_r2, completeness_r2, distance_r2 = anf.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
        h_measure_2 = n_harm_r2 * completeness_r2  # compute h_measure for constraining a domain
        mask_peak = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
        i_min_dist = np.argmin(distance_r2[mask_peak])
        p_orb = 1 / f_refine_2[mask_peak][i_min_dist]

    return p_orb


@nb.njit(cache=True)
def linear_curve(time, const, slope, i_chunks, t_shift=True):
    """Returns a piece-wise linear curve for the given time points
    with slopes and y-intercepts.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The model time series of a (set of) straight line(s)
    
    Notes
    -----
    Assumes the constants and slopes are determined with respect
    to the sector mean time as zero point.
    """
    curve = np.zeros(len(time))
    for co, sl, s in zip(const, slope, i_chunks):
        if t_shift:
            t_sector_mean = np.mean(time[s[0]:s[1]])
        else:
            t_sector_mean = 0
        curve[s[0]:s[1]] = co + sl * (time[s[0]:s[1]] - t_sector_mean)

    return curve


@nb.njit(cache=True)
def linear_pars(time, flux, i_chunks):
    """Calculate the slopes and y-intercepts of a linear trend with the MLE.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        y_inter: numpy.ndarray[Any, dtype[float]]
            The y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            The slopes of a piece-wise linear curve
    
    Notes
    -----
    Source: https://towardsdatascience.com/linear-regression-91eeae7d6a2e
    Determines the constants and slopes with respect to the sector mean time
    as zero point to avoid correlations.
    """
    y_inter = np.zeros(len(i_chunks))
    slope = np.zeros(len(i_chunks))
    for i, s in enumerate(i_chunks):
        # mean and mean subtracted quantities
        x_m = np.mean(time[s[0]:s[1]])
        x_ms = (time[s[0]:s[1]] - x_m)
        y_m = np.mean(flux[s[0]:s[1]])
        y_ms = (flux[s[0]:s[1]] - y_m)

        # sums
        s_xx = np.sum(x_ms**2)
        s_xy = np.sum(x_ms * y_ms)

        # parameters
        slope[i] = s_xy / s_xx
        # y_inter[i] = y_m - slope[i] * x_m  # original non-mean-centered formula
        y_inter[i] = y_m  # mean-centered value

    return y_inter, slope


@nb.njit(cache=True)
def linear_pars_two_points(x1, y1, x2, y2):
    """Calculate the slope(s) and y-intercept(s) of a linear curve defined by two points.
    
    Parameters
    ----------
    x1: float, numpy.ndarray[Any, dtype[float]]
        The x-coordinate of the left point(s)
    y1: float, numpy.ndarray[Any, dtype[float]]
        The y-coordinate of the left point(s)
    x2: float, numpy.ndarray[Any, dtype[float]]
        The x-coordinate of the right point(s)
    y2: float, numpy.ndarray[Any, dtype[float]]
        The y-coordinate of the right point(s)
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        y_inter: float, numpy.ndarray[Any, dtype[float]]
            The y-intercept(s) of a piece-wise linear curve
        slope: float, numpy.ndarray[Any, dtype[float]]
            The slope(s) of a piece-wise linear curve
    """
    slope = (y2 - y1) / (x2 - x1)
    y_inter = y1 - (x1 * slope)

    return y_inter, slope


@nb.njit(cache=True)
def sum_sines(time, f_n, a_n, ph_n, t_shift=True):
    """A sum of sine waves at times t, given the frequencies, amplitudes and phases.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of a sum of sine waves. Varies around 0.
    
    Notes
    -----
    Assumes the phases are determined with respect
    to the mean time as zero point by default.
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    model_sines = np.zeros(len(time))
    for f, a, ph in zip(f_n, a_n, ph_n):
        # model_sines += a * np.sin((2 * np.pi * f * (time - mean_t)) + ph)
        # double loop runs a tad bit quicker when numba-JIT-ted
        for i, t in enumerate(time):
            model_sines[i] += a * np.sin((2 * np.pi * f * (t - mean_t)) + ph)

    return model_sines


@nb.njit(cache=True)
def sum_sines_deriv(time, f_n, a_n, ph_n, deriv=1, t_shift=True):
    """The derivative of a sum of sine waves at times t,
    given the frequencies, amplitudes and phases.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: list[float], numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: list[float], numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: list[float], numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    deriv: int
        Number of time derivatives taken (>= 1)
    t_shift: bool
        Mean center the time axis
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of a sum of sine wave derivatives. Varies around 0.
    
    Notes
    -----
    Assumes the phases are determined with respect
    to the mean time as zero point by default.
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    model_sines = np.zeros(len(time))
    mod_2 = deriv % 2
    mod_4 = deriv % 4
    ph_cos = (np.pi / 2) * mod_2  # alternate between cosine and sine
    sign = (-1)**((mod_4 - mod_2) // 2)  # (1, -1, -1, 1, 1, -1, -1... for deriv=1, 2, 3...)
    for f, a, ph in zip(f_n, a_n, ph_n):
        for i, t in enumerate(time):
            model_sines[i] += sign * (2 * np.pi * f)**deriv * a * np.sin((2 * np.pi * f * (t - mean_t)) + ph + ph_cos)

    return model_sines


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
    n_param = 2 + 3 * len(a_n)  # number of parameters in the model
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


def extract_single(time, flux, f0=-1, fn=-1, select='a'):
    """Extract a single sinusoid from a time series.

    The extracted frequency is based on the highest amplitude or signal-to-noise in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float, optional
        Lowest allowed frequency for extraction.
        If left -1, default is f0 = 1/(100*T)
    fn: float, optional
        Highest allowed frequency for extraction.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    select: str, optional
        Select the next frequency based on amplitude 'a' or signal-to-noise 'sn'
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid
    
    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # full LS periodogram
    freqs, ampls = pdg.scargle_parallel(time, flux, f0=f0, fn=fn, df=df)

    # selection step based on flux to noise (refine step keeps using ampl)
    if select == 'sn':
        noise_spectrum = pdg.scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0)
        ampls = ampls / noise_spectrum

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df/100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_local(time, flux, f0, fn, select='a'):
    """Extract a single sinusoid from a time series at a predefined frequency interval.

    The extracted frequency is based on the highest amplitude or signal-to-noise in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Lowest allowed frequency for extraction.
    fn: float
        Highest allowed frequency for extraction.
    select: str, optional
        Select the next frequency based on amplitude 'a' or signal-to-noise 'sn'

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid

    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # full LS periodogram (accurate version)
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # cut off the ends of the frequency range if they are rising
    i_f_min_edges = ut.uphill_local_max(freqs, -ampls, freqs[[0, -1]])
    freqs = freqs[i_f_min_edges[0]:i_f_min_edges[1] + 1]
    ampls = ampls[i_f_min_edges[0]:i_f_min_edges[1] + 1]

    # selection step based on flux to noise (refine step keeps using ampl)
    if select == 'sn':
        noise_spectrum = pdg.scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0)
        ampls = ampls / noise_spectrum

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_approx(time, flux, f_approx):
    """Extract a single sinusoid from a time series at an approximate location.

    Follows the periodogram upwards to the nearest peak. The periodogram is oversampled for a more precise result.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_approx: float
        Approximate location of the frequency of maximum amplitude.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # LS periodogram around the approximate location
    f0 = max(f_approx - 50 * df, df / 10)
    fn = f_approx + 50 * df
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # get the index of the frequency of the maximum amplitude
    i_f_max = ut.uphill_local_max(freqs, ampls, np.array([f_approx]))[0]

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def refine_subset(time, flux, close_f, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other,
    taking into account (and not changing the frequencies of) harmonics if present.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    close_f: list[int], numpy.ndarray[Any, dtype[int]]
        Indices of the subset of frequencies to be refined
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves
    
    See Also
    --------
    extract_all
    
    Notes
    -----
    Intended as a sub-loop within another extraction routine (extract_all),
    can work standalone too.
    """
    freq_res = 1.5 / np.ptp(time)  # frequency resolution
    n_sectors = len(i_chunks)
    n_f = len(f_n)
    n_g = len(close_f)  # number of frequencies being updated
    harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)

    # determine initial bic
    model_sinusoid_ncf = sum_sines(time, np.delete(f_n, close_f), np.delete(a_n, close_f), np.delete(ph_n, close_f))
    cur_resid = flux - (model_sinusoid_ncf + sum_sines(time, f_n[close_f], a_n[close_f], ph_n[close_f]))
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_f - n_harm)
    bic_prev = gof.calc_bic(resid, n_param)
    bic_init = bic_prev

    # stop the loop when the BIC increases
    accept = True
    while accept:
        accept = False
        # remove each frequency one at a time to then re-extract them
        for j in close_f:
            cur_resid += sum_sines(time, np.array([f_n_temp[j]]), np.array([a_n_temp[j]]), np.array([ph_n_temp[j]]))
            const, slope = linear_pars(time, cur_resid, i_chunks)
            resid = cur_resid - linear_curve(time, const, slope, i_chunks)

            # if f is a harmonic, don't shift the frequency
            if j in harmonics:
                f_j = f_n_temp[j]
                a_j, ph_j = pdg.scargle_ampl_phase_single(time, resid, f_j)
            else:
                f_j, a_j, ph_j = extract_approx(time, resid, f_n_temp[j])

            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
            cur_resid -= sum_sines(time, np.array([f_j]), np.array([a_j]), np.array([ph_j]))

        # as a last model-refining step, redetermine the constant and slope
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        # calculate BIC before moving to the next iteration
        bic = gof.calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        if np.round(d_bic, 2) > 0:
            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic
            accept = True

        if logger is not None:
            logger.extra(f'N_f= {n_f}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) '
                         f'- N_refine= {n_g}, f= {f_j:1.6f}, a= {a_j:1.6f}')

    if logger is not None:
        logger.extra(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (total= {bic_init - bic_prev:1.2f}) - end refinement')

    # redo the constant and slope without the last iteration of changes
    resid = flux - (model_sinusoid_ncf + sum_sines(time, f_n[close_f], a_n[close_f], ph_n[close_f]))
    const, slope = linear_pars(time, resid, i_chunks)

    return const, slope, f_n, a_n, ph_n


def extract_sinusoids(time, flux, i_chunks, p_orb=0, f_n=None, a_n=None, ph_n=None, bic_thr=2, snr_thr=0,
                      stop_crit='bic', select='hybrid', n_extract=0, f0=-1, fn=-1, fit_each_step=False, logger=None):
    """Extract all the frequencies from a periodic flux.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    p_orb: float, optional
        Orbital period of the eclipsing binary in days (can be 0)
    f_n: numpy.ndarray[Any, dtype[float]], optional
        The frequencies of a number of sine waves (can be empty or None)
    a_n: numpy.ndarray[Any, dtype[float]], optional
        The amplitudes of a number of sine waves (can be empty or None)
    ph_n: numpy.ndarray[Any, dtype[float]], optional
        The phases of a number of sine waves (can be empty or None)
    bic_thr: float, optional
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    snr_thr: float, optional
        Threshold for signal-to-noise ratio for a signal to be considered significant.
    stop_crit: str, optional
        Use the BIC as stopping criterion or the SNR, choose from 'bic', or 'snr'
    select: str, optional
        Select the next frequency based on amplitude ('a'),
        signal-to-noise ('sn'), or hybrid ('hybrid') (first a then sn).
    n_extract: int, optional
        Maximum number of frequencies to extract. The stop criterion is still leading. Zero means as many as possible.
    f0: float
        Lowest allowed frequency for extraction.
        If left -1, default is f0 = 1/(100*T)
    fn: float
        Highest allowed frequency for extraction.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    fit_each_step: bool
        If set to True, a non-linear least-squares fit of all extracted sinusoids in groups is performed at each
        iteration. While this increases the quality of the extracted signals, it drastically slows down the code.
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            The y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            The slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves

    Notes
    -----
    Spits out frequencies and amplitudes in the same units as the input,
    and phases that are measured with respect to the first time point.
    Also determines the flux average, so this does not have to be subtracted
    before input into this function.
    Note: does not perform a non-linear least-squares fit at the end,
    which is highly recommended! (In fact, no fitting is done at all).
    
    The function optionally takes a pre-existing frequency list to append
    additional frequencies to. Set these to np.array([]) to start from scratch.
    
    i_chunks is a 2D array with start and end indices of each (half) sector.
    This is used to model a piecewise-linear trend in the data.
    If you have no sectors like the TESS mission does, set
    i_chunks = np.array([[0, len(time)]])

    Exclusively uses the Lomb-Scargle periodogram (and an iterative parameter
    improvement scheme) to extract the frequencies.
    Uses a delta BIC > bic_thr stopping criterion.

    [Author's note] Although it is my belief that doing a non-linear
    multi-sinusoid fit at each iteration of the prewhitening is the
    ideal approach, it is also a very (very!) time-consuming one and this
    algorithm aims to be fast while approaching the optimal solution.

    [Another author's note] I added an option to do the non-linear multi-
    sinusoid fit at each iteration.
    """
    if f_n is None:
        f_n = np.array([])
    if a_n is None:
        a_n = np.array([])
    if ph_n is None:
        ph_n = np.array([])
    if n_extract == 0:
        n_extract = 10**6  # 'a lot'

    # setup
    freq_res = 1.5 / np.ptp(time)  # frequency resolution
    n_sectors = len(i_chunks)
    n_freq_init = len(f_n)
    if n_freq_init > 0:
        harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    else:
        harmonics = np.array([])
    n_harm = len(harmonics)

    # set up selection process
    if select == 'hybrid':
        switch = True  # when we would normally end, we switch strategy
        select = 'a'  # start with amplitude extraction
    else:
        switch = False

    # determine the initial bic
    cur_resid = flux - sum_sines(time, f_n, a_n, ph_n)
    const, slope = linear_pars(time, cur_resid, i_chunks)
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq_init - n_harm)
    bic_prev = gof.calc_bic(resid, n_param)  # initialise current BIC to the mean (and slope) subtracted flux
    bic_init = bic_prev

    # log a message
    if logger is not None:
        logger.extra(f'N_f= {len(f_n)}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')

    # stop the loop when the BIC decreases by less than 2 (or increases)
    n_freq_cur = -1
    while (len(f_n) > n_freq_cur) | switch:
        # switch selection method when extraction would normally stop
        if switch & (not (len(f_n) > n_freq_cur)):
            select = 'sn'
            switch = False

        # update number of current frequencies
        n_freq_cur = len(f_n)

        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(time, resid, f0=f0, fn=fn, select=select)

        # now improve frequencies - make a temporary array including the current one
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)
        if fit_each_step:
            # select all frequencies for the full fit
            close_f = np.arange((len(f_n)))  # all f
        else:
            # select only close frequencies for iteration
            close_f = anf.f_within_rayleigh(n_freq_cur, f_n_temp, freq_res)

        # make a model of only the close sinusoids and subtract the current sinusoid
        model_sinusoid_r = sum_sines(time, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        model_sinusoid_r -= sum_sines(time, np.array([f_i]), np.array([a_i]), np.array([ph_i]))

        if fit_each_step:
            # fit all frequencies for best improvement
            fit_out = fit.fit_multi_sinusoid_per_group(time, flux, const, slope, f_n_temp, a_n_temp, ph_n_temp,
                                                       i_chunks, logger=logger)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = fit_out
        elif len(close_f) > 1:
            # iterate over (re-extract) close frequencies (around f_i) a number of times to improve them
            refine_out = refine_subset(time, flux, close_f, p_orb, const, slope, f_n_temp, a_n_temp, ph_n_temp,
                                       i_chunks, logger=logger)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_out

        # make the model of the updated close sinusoids and determine new residuals
        model_sinusoid_n = sum_sines(time, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        cur_resid -= (model_sinusoid_n - model_sinusoid_r)  # add the changes to the sinusoid residuals

        # as a last model-refining step, redetermine the constant and slope
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        # calculate BIC
        n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq_cur + 1 - n_harm)
        bic = gof.calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        condition = np.round(d_bic, 2) > bic_thr
        # calculate SNR in a 1 c/d window around the extracted frequency
        if stop_crit == 'snr':
            noise = pdg.scargle_noise_at_freq(np.array([f_i]), time, resid, window_width=1.0)
            snr = a_i / noise
            condition = snr > snr_thr

        # check acceptance condition before moving to the next iteration
        if condition & (n_freq_cur - n_freq_init < n_extract):
            # accept the new frequency
            f_n, a_n, ph_n = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)

            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic

        if logger is not None:
            logger.extra(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) - '
                         f'f= {f_i:1.6f}, a= {a_i:1.6f}')

    if logger is not None:
        logger.extra(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')

    # lastly re-determine slope and const
    cur_resid += (model_sinusoid_n - model_sinusoid_r)  # undo last change
    const, slope = linear_pars(time, cur_resid, i_chunks)

    return const, slope, f_n, a_n, ph_n


def extract_harmonics(time, flux, p_orb, i_chunks, bic_thr, f_n=None, a_n=None, ph_n=None, logger=None):
    """Tries to extract more harmonics from the flux
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_n: numpy.ndarray[Any, dtype[float]], optional
        The frequencies of a number of sine waves (can be empty or None)
    a_n: numpy.ndarray[Any, dtype[float]], optional
        The amplitudes of a number of sine waves (can be empty or None)
    ph_n: numpy.ndarray[Any, dtype[float]], optional
        The phases of a number of sine waves (can be empty or None)
    i_chunks: numpy.ndarray[Any, dtype[int]], optional
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    bic_thr: float
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    logger: logging.Logger, optional
        Instance of the logging library.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (higher) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (higher) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (higher) number of sine waves
    
    See Also
    --------
    fix_harmonic_frequency
    
    Notes
    -----
    Looks for missing harmonics and checks whether adding them
    decreases the BIC sufficiently (by more than 2).
    Assumes the harmonics are already fixed multiples of 1/p_orb
    as can be achieved with fix_harmonic_frequency.
    """
    if f_n is None:
        f_n = np.array([])
    if a_n is None:
        a_n = np.array([])
    if ph_n is None:
        ph_n = np.array([])

    # setup
    f_max = 1 / (2 * np.min(time[1:] - time[:-1]))  # Nyquist freq
    n_sectors = len(i_chunks)
    n_freq = len(f_n)

    # extract the existing harmonics using the period
    if n_freq > 0:
        harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    else:
        harmonics, harmonic_n = np.array([], dtype=int), np.array([], dtype=int)
    n_harm = len(harmonics)

    # make a list of not-present possible harmonics
    h_candidate = np.arange(1, p_orb * f_max, dtype=int)
    h_candidate = np.delete(h_candidate, harmonic_n - 1)  # harmonic_n minus one is the position

    # initial residuals
    cur_resid = flux - sum_sines(time, f_n, a_n, ph_n)
    const, slope = linear_pars(time, cur_resid, i_chunks)
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_init = gof.calc_bic(resid, n_param)
    bic_prev = bic_init
    if logger is not None:
        logger.extra(f'N_f= {n_freq}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')

    # loop over candidates and try to extract (BIC decreases by 2 or more)
    n_h_acc = []
    for h_c in h_candidate:
        f_c = h_c / p_orb
        a_c, ph_c = pdg.scargle_ampl_phase_single(time, resid, f_c)

        # redetermine the constant and slope
        model_sinusoid_n = sum_sines(time, np.array([f_c]), np.array([a_c]), np.array([ph_c]))
        cur_resid -= model_sinusoid_n
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        # determine new BIC and whether it improved
        n_harm_cur = n_harm + len(n_h_acc) + 1
        n_param = 2 * n_sectors + 1 * (n_harm_cur > 0) + 2 * n_harm_cur + 3 * (n_freq - n_harm)
        bic = gof.calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        if np.round(d_bic, 2) > bic_thr:
            # h_c is accepted, add it to the final list and continue
            bic_prev = bic
            f_n, a_n, ph_n = np.append(f_n, f_c), np.append(a_n, a_c), np.append(ph_n, ph_c)
            n_h_acc.append(h_c)
        else:
            # h_c is rejected, revert to previous residual
            cur_resid += model_sinusoid_n
            const, slope = linear_pars(time, cur_resid, i_chunks)
            resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        if logger is not None:
            logger.extra(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f})'
                         f' - h= {h_c}')

    if logger is not None:
        logger.extra(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')
        if len(n_h_acc) > 0:
            logger.extra(f'Successfully extracted harmonics {n_h_acc}')

    return const, slope, f_n, a_n, ph_n


def fix_harmonic_frequency(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Fixes the frequency of harmonics to the theoretical value, then
    re-determines the amplitudes and phases.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of the same number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of the same number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of the same number of sine waves
    """
    # extract the harmonics using the period and determine some numbers
    freq_res = 1.5 / np.ptp(time)
    harmonics, harmonic_n = anf.find_harmonics_tolerance(f_n, p_orb, f_tol=freq_res / 2)
    if len(harmonics) == 0:
        raise ValueError('No harmonic frequencies found')
    
    n_sectors = len(i_chunks)
    n_freq = len(f_n)
    n_harm_init = len(harmonics)

    # indices of harmonic candidates to remove
    remove_harm_c = np.zeros(0, dtype=np.int_)
    f_new, a_new, ph_new = np.zeros((3, 0))

    # determine initial bic
    model_sinusoid = sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 + 2 * n_harm_init + 3 * (n_freq - n_harm_init)
    bic_init = gof.calc_bic(resid, n_param)

    # go through the harmonics by harmonic number and re-extract them (removing all duplicate n's in the process)
    for n in np.unique(harmonic_n):
        remove = np.arange(len(f_n))[harmonics][harmonic_n == n]
        # make a model of the removed sinusoids and subtract it from the full sinusoid residual
        model_sinusoid_r = sum_sines(time, f_n[remove], a_n[remove], ph_n[remove])
        cur_resid += model_sinusoid_r
        const, slope = linear_pars(time, resid, i_chunks)  # redetermine const and slope
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        # calculate the new harmonic
        f_i = n / p_orb  # fixed f
        a_i, ph_i = pdg.scargle_ampl_phase_single(time, resid, f_i)

        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = sum_sines(time, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        cur_resid -= model_sinusoid_n

        # add to freq list and removal list
        f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
        remove_harm_c = np.append(remove_harm_c, remove)
        if logger is not None:
            logger.extra(f'Harmonic number {n} re-extracted, replacing {len(remove)} candidates')

    # lastly re-determine slope and const (not needed here)
    # const, slope = linear_pars(time, cur_resid, i_chunks)
    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.append(np.delete(f_n, remove_harm_c), f_new)
    a_n = np.append(np.delete(a_n, remove_harm_c), a_new)
    ph_n = np.append(np.delete(ph_n, remove_harm_c), ph_new)

    # re-extract the non-harmonics
    n_freq = len(f_n)
    harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)
    remove_non_harm = np.zeros(0, dtype=np.int_)
    for i in non_harm:
        # make a model of the removed sinusoid and subtract it from the full sinusoid residual
        model_sinusoid_r = sum_sines(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid += model_sinusoid_r
        const, slope = linear_pars(time, cur_resid, i_chunks)  # redetermine const and slope
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)

        # extract the updated frequency
        fl, fr = f_n[i] - freq_res, f_n[i] + freq_res
        f_n[i], a_n[i], ph_n[i] = extract_approx(time, resid, f_n[i])
        if (f_n[i] <= fl) | (f_n[i] >= fr):
            remove_non_harm = np.append(remove_non_harm, [i])

        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = sum_sines(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
        cur_resid -= model_sinusoid_n

    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.delete(f_n, non_harm[remove_non_harm])
    a_n = np.delete(a_n, non_harm[remove_non_harm])
    ph_n = np.delete(ph_n, non_harm[remove_non_harm])

    # re-establish cur_resid
    model_sinusoid = sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    const, slope = linear_pars(time, cur_resid, i_chunks)  # lastly re-determine slope and const

    if logger is not None:
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (n_freq - n_harm)
        bic = gof.calc_bic(resid, n_param)
        logger.extra(f'Candidate harmonics replaced: {n_harm_init} ({n_harm} left). '
                     f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {bic_init - bic:1.2f})')

    return const, slope, f_n, a_n, ph_n


# @nb.njit(cache=True)
def remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt the removal of individual frequencies
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency.
    Harmonics are taken into account.
    """
    n_sectors = len(i_chunks)
    n_freq = len(f_n)
    harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)

    # indices of single frequencies to remove
    remove_single = np.zeros(0, dtype=np.int_)

    # determine initial bic
    model_sinusoid = sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = gof.calc_bic(resid, n_param)
    bic_init = bic_prev
    n_prev = -1

    # while frequencies are added to the remove list, continue loop
    while len(remove_single) > n_prev:
        n_prev = len(remove_single)
        for i in range(n_freq):
            if i in remove_single:
                continue
            
            # make a model of the removed sinusoids and subtract it from the full sinusoid model
            model_sinusoid_r = sum_sines(time, np.array([f_n[i]]), np.array([a_n[i]]), np.array([ph_n[i]]))
            resid = cur_resid + model_sinusoid_r
            const, slope = linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= linear_curve(time, const, slope, i_chunks)

            # number of parameters and bic
            n_harm_i = n_harm - len([h for h in remove_single if h in harmonics]) - 1 * (i in harmonics)
            n_freq_i = n_freq - len(remove_single) - 1 - n_harm_i
            n_param = 2 * n_sectors + 1 * (n_harm_i > 0) + 2 * n_harm_i + 3 * n_freq_i
            bic = gof.calc_bic(resid, n_param)

            # if improvement, add to list of removed freqs
            if np.round(bic_prev - bic, 2) > 0:
                remove_single = np.append(remove_single, i)
                cur_resid += model_sinusoid_r
                bic_prev = bic

    # lastly re-determine slope and const
    const, slope = linear_pars(time, cur_resid, i_chunks)

    # finally, remove all the designated sinusoids from the lists
    f_n = np.delete(f_n, remove_single)
    a_n = np.delete(a_n, remove_single)
    ph_n = np.delete(ph_n, remove_single)

    if logger is not None:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        logger.extra(f'Single frequencies removed: {n_freq - len(f_n)}, '
                     f'N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')

    return const, slope, f_n, a_n, ph_n


# @nb.njit(cache=True)
def replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt the replacement of groups of frequencies by a single one

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves

    Notes
    -----
    Checks whether the BIC can be improved by replacing a group of
    frequencies by only one. Harmonics are never removed.
    """
    freq_res = 1.5 / np.ptp(time)  # frequency resolution
    n_sectors = len(i_chunks)
    n_freq = len(f_n)
    harmonics, harmonic_n = anf.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)

    # make an array of sets of frequencies (non-harmonic) to be investigated for replacement
    close_f_groups = anf.chains_within_rayleigh(f_n[non_harm], freq_res)
    close_f_groups = [non_harm[group] for group in close_f_groups]  # convert to the right indices
    f_sets = [g[np.arange(p1, p2 + 1)]
              for g in close_f_groups
              for p1 in range(len(g) - 1)
              for p2 in range(p1 + 1, len(g))]

    # make an array of sets of frequencies (now with harmonics) to be investigated for replacement
    close_f_groups = anf.chains_within_rayleigh(f_n, freq_res)
    f_sets_h = [g[np.arange(p1, p2 + 1)]
                for g in close_f_groups
                for p1 in range(len(g) - 1)
                for p2 in range(p1 + 1, len(g))
                if np.any(np.array([g_f in harmonics for g_f in g[np.arange(p1, p2 + 1)]]))]

    # join the two lists, and remember which is which
    harm_sets = np.arange(len(f_sets), len(f_sets) + len(f_sets_h))
    f_sets.extend(f_sets_h)
    remove_sets = np.zeros(0, dtype=np.int_)  # sets of frequencies to replace (by 1 freq)
    used_sets = np.zeros(0, dtype=np.int_)  # sets that are not to be examined anymore
    f_new, a_new, ph_new = np.zeros((3, 0))

    # determine initial bic
    model_sinusoid = sum_sines(time, f_n, a_n, ph_n)
    best_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = best_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = gof.calc_bic(resid, n_param)
    bic_init = bic_prev
    n_prev = -1

    # while frequencies are added to the remove list, continue loop
    while len(remove_sets) > n_prev:
        n_prev = len(remove_sets)
        for i, set_i in enumerate(f_sets):
            if i in used_sets:
                continue
            
            # make a model of the removed and new sinusoids and subtract/add it from/to the full sinusoid model
            model_sinusoid_r = sum_sines(time, f_n[set_i], a_n[set_i], ph_n[set_i])
            resid = best_resid + model_sinusoid_r
            const, slope = linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= linear_curve(time, const, slope, i_chunks)

            # extract a single freq to try replacing the set
            if i in harm_sets:
                harm_i = np.array([h for h in set_i if h in harmonics])
                f_i = f_n[harm_i]  # fixed f
                a_i, ph_i = pdg.scargle_ampl_phase(time, resid, f_n[harm_i])
            else:
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                out = extract_local(time, resid, f0=edges[0], fn=edges[1])
                f_i, a_i, ph_i = np.array([out[0]]), np.array([out[1]]), np.array([out[2]])

            # make a model including the new freq
            model_sinusoid_n = sum_sines(time, f_i, a_i, ph_i)
            resid -= model_sinusoid_n
            const, slope = linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= linear_curve(time, const, slope, i_chunks)

            # number of parameters and bic
            n_freq_i = n_freq - sum([len(f_sets[j]) for j in remove_sets]) - len(set_i) + len(f_new) + len(f_i) - n_harm
            n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * n_freq_i
            bic = gof.calc_bic(resid, n_param)
            if np.round(bic_prev - bic, 2) > 0:
                # do not look at sets with the same freqs as the just removed set anymore
                overlap = [j for j, subset in enumerate(f_sets) if np.any(np.array([k in set_i for k in subset]))]
                used_sets = np.unique(np.append(used_sets, overlap))

                # add to list of removed sets
                remove_sets = np.append(remove_sets, i)

                # remember the new frequency (or the current one if it is a harmonic)
                f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
                best_resid += model_sinusoid_r - model_sinusoid_n
                bic_prev = bic

    # lastly re-determine slope and const
    const, slope = linear_pars(time, best_resid, i_chunks)

    # finally, remove all the designated sinusoids from the lists and add the new ones
    i_to_remove = [k for i in remove_sets for k in f_sets[i]]
    f_n = np.append(np.delete(f_n, i_to_remove), f_new)
    a_n = np.append(np.delete(a_n, i_to_remove), a_new)
    ph_n = np.append(np.delete(ph_n, i_to_remove), ph_new)

    if logger is not None:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        logger.extra(f'Frequency sets replaced by a single frequency: {len(remove_sets)} '
                     f'({len(i_to_remove)} frequencies). N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')

    return const, slope, f_n, a_n, ph_n


def reduce_sinusoids(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Attempt to reduce the number of frequencies taking into account any harmonics if present.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the eclipsing binary in days (can be 0)
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        const: numpy.ndarray[Any, dtype[float]]
            (Updated) y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            (Updated) slopes of a piece-wise linear curve
        f_n: numpy.ndarray[Any, dtype[float]]
            (Updated) frequencies of a (lower) number of sine waves
        a_n: numpy.ndarray[Any, dtype[float]]
            (Updated) amplitudes of a (lower) number of sine waves
        ph_n: numpy.ndarray[Any, dtype[float]]
            (Updated) phases of a (lower) number of sine waves
    
    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency. Special attention
    is given to frequencies that are within the Rayleigh criterion of each other.
    It is attempted to replace these by a single frequency.
    """
    # first check if any frequency can be left out (after the fit, this may be possible)
    out_a = remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=logger)
    const, slope, f_n, a_n, ph_n = out_a

    # Now go on to trying to replace sets of frequencies that are close together
    out_b = replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=logger)
    const, slope, f_n, a_n, ph_n = out_b

    return const, slope, f_n, a_n, ph_n


def select_sinusoids(time, flux, flux_err, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Selects the credible frequencies from the given set
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
        If the sinusoids exclude the eclipse model,
        this should be the residuals of the eclipse model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days.
        May be zero.
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        passed_sigma: numpy.ndarray[bool]
            Non-harmonic frequencies that passed the sigma check
        passed_snr: numpy.ndarray[bool]
            Non-harmonic frequencies that passed the signal-to-noise check
        passed_both: numpy.ndarray[bool]
            Non-harmonic frequencies that passed both checks
        passed_harmonic: numpy.ndarray[bool]
            Harmonic frequencies that passed
    
    Notes
    -----
    Harmonic frequencies that are said to be passing the criteria
    are in fact passing the criteria for individual frequencies,
    not those for a set of harmonics (which would be a looser constraint).
    """
    t_tot = np.ptp(time)
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    
    # obtain the errors on the sine waves (depends on residual and thus model)
    model_lin = linear_curve(time, const, slope, i_chunks)
    model_sin = sum_sines(time, f_n, a_n, ph_n)
    residuals = flux - (model_lin + model_sin)
    errors = formal_uncertainties(time, residuals, flux_err, a_n, i_chunks)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = errors
    
    # find the insignificant frequencies
    remove_sigma = anf.remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3, sigma_f=3)
    
    # apply the signal-to-noise threshold
    noise_at_f = pdg.scargle_noise_at_freq(f_n, time, residuals, window_width=1.0)
    remove_snr = anf.remove_insignificant_snr(time, a_n, noise_at_f)
    
    # frequencies that pass sigma criteria
    passed_sigma = np.ones(len(f_n), dtype=bool)
    passed_sigma[remove_sigma] = False
    
    # frequencies that pass S/N criteria
    passed_snr = np.ones(len(f_n), dtype=bool)
    passed_snr[remove_snr] = False
    
    # passing both
    passed_both = (passed_sigma & passed_snr)
    
    # candidate harmonic frequencies
    passed_harmonic = np.zeros(len(f_n), dtype=bool)
    if p_orb != 0:
        harmonics, harmonic_n = anf.select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol=freq_res / 2, sigma_f=3)
        passed_harmonic[harmonics] = True
    else:
        harmonics = np.array([], dtype=int)
    if logger is not None:
        logger.extra(f'Number of frequencies passed criteria: {np.sum(passed_both)} of {len(f_n)}. '
                     f'Candidate harmonics: {np.sum(passed_harmonic)}, '
                     f'of which {np.sum(passed_both[harmonics])} passed.')
        
    return passed_sigma, passed_snr, passed_both, passed_harmonic
