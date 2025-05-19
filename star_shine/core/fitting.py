"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and harmonic sinusoids.

Notes
-----
Minimize methods:
Nelder-Mead is extensively tested and found robust, while slow.
TNC is tested, and seems reliable while being fast, though slightly worse BIC results.
L-BFGS-B is tested, and seems reliable while being fast, though slightly worse BIC results.
See publication appendix for more information.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import scipy as sp
import scipy.optimize

import star_shine.core.frequency_sets
from star_shine.core import timeseries as ts, goodness_of_fit as gof, frequency_sets as frs


@nb.njit(cache=True)
def dsin_dx(two_pi_t, f, a, ph, d='f', p_orb=0):
    """The derivative of a sine wave at times t, where x is on of the parameters.

    Parameters
    ----------
    two_pi_t: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series times two pi
    f: float
        The frequency of a sine wave
    a: float
        The amplitude of a sine wave
    ph: float
        The phase of a sine wave
    d: string
        Which derivative to take
        Choose f, a, ph, p_orb
    p_orb: float
        Orbital period of the eclipsing binary in days

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of the derivative of a sine wave to f.

    Notes
    -----
    Make sure the phases correspond to the given
    time zero point.
    If d='p_orb', it is assumed that f is a harmonic
    """
    if d == 'f':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t
    elif d == 'a':
        model_deriv = np.sin(two_pi_t * f + ph)
    elif d == 'ph':
        model_deriv = a * np.cos(two_pi_t * f + ph)
    elif d == 'p_orb':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t * f / p_orb
    else:
        model_deriv = np.zeros(len(two_pi_t))

    return model_deriv


@nb.njit(cache=True)
def objective_sinusoids(params, time, flux, i_chunks):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_chunk) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    const = params[:n_chunk]
    slope = params[n_chunk:2 * n_chunk]
    f_n = params[2 * n_chunk:2 * n_chunk + n_sin]
    a_n = params[2 * n_chunk + n_sin:2 * n_chunk + 2 * n_sin]
    ph_n = params[2 * n_chunk + 2 * n_sin:2 * n_chunk + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = ts.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = ts.sum_sines(time, f_n, a_n, ph_n)

    # calculate the likelihood (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    ln_likelihood = gof.calc_iid_normal_likelihood(resid)

    return -ln_likelihood


@nb.njit(cache=True, parallel=True)
def jacobian_sinusoids(params, time, flux, i_chunks):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...]
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    objective_sinusoids
    """
    time_ms = time - np.mean(time)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_chunk) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    const = params[:n_chunk]
    slope = params[n_chunk:2 * n_chunk]
    f_n = params[2 * n_chunk:2 * n_chunk + n_sin]
    a_n = params[2 * n_chunk + n_sin:2 * n_chunk + 2 * n_sin]
    ph_n = params[2 * n_chunk + 2 * n_sin:2 * n_chunk + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = ts.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = ts.sum_sines(time, f_n, a_n, ph_n)

    # calculate the likelihood derivative (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    two_pi_t = 2 * np.pi * time_ms

    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_chunk)  # calculated per sector
    df_1b = -len(time) / np.sum(resid**2)

    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_chunk)
    for i in nb.prange(n_chunk):
        s = i_chunks[i]
        i_s = i + n_chunk
        df_1a[i] = -len(time[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (time[s[0]:s[1]] - np.mean(time[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a

    # calculate the rest of the jacobian for the sinusoid parameters, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin)
    for i in nb.prange(n_sin):
        i_a = i + n_sin  # index of df_2b for a_n
        i_ph = i + 2 * n_sin  # index of df_2b for ph_n
        df_2b[i] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='ph'))

    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)

    return jac


def fit_multi_sinusoid(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Perform the multi-sinusoid, non-linear least-squares fit.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
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
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_f_n: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_a_n: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_ph_n: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_f_n = np.copy(f_n)
    res_a_n = np.copy(a_n)
    res_ph_n = np.copy(ph_n)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = len(f_n)  # each sine has freq, ampl and phase

    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(time)
    f_low = 0.01 / t_tot

    # do the fit
    par_init = np.concatenate((res_const, res_slope, res_f_n, res_a_n, res_ph_n))
    par_bounds = [(None, None) for _ in range(2 * n_chunk)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    arguments = (time, flux, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids, jac=jacobian_sinusoids, x0=par_init, args=arguments,
                                  method='L-BFGS-B', bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_const = result.x[0:n_chunk]
    res_slope = result.x[n_chunk:2 * n_chunk]
    res_f_n = result.x[2 * n_chunk:2 * n_chunk + n_sin]
    res_a_n = result.x[2 * n_chunk + n_sin:2 * n_chunk + 2 * n_sin]
    res_ph_n = result.x[2 * n_chunk + 2 * n_sin:2 * n_chunk + 3 * n_sin]

    if logger is not None:
        model_linear = ts.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = ts.sum_sines(time, res_f_n, res_a_n, res_ph_n)
        resid = flux - model_linear - model_sinusoid
        bic = gof.calc_bic(resid, 2 * n_chunk + 3 * n_sin)
        logger.extra(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
                     f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    return res_const, res_slope, res_f_n, res_a_n, res_ph_n


def fit_multi_sinusoid_per_group(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Perform the multi-sinusoid, non-linear least-squares fit per frequency group

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
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
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_f_n: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_a_n: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_ph_n: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits per group of 15-20 frequencies, leaving the other frequencies as
    fixed parameters.
    """
    f_groups = star_shine.core.frequency_sets.group_frequencies_for_fit(a_n, g_min=45, g_max=50)
    n_groups = len(f_groups)
    n_chunk = len(i_chunks)
    n_sin = len(f_n)

    # make a copy of the initial parameters
    res_const = np.copy(const)
    res_slope = np.copy(slope)
    res_f_n = np.copy(f_n)
    res_a_n = np.copy(a_n)
    res_ph_n = np.copy(ph_n)

    # update the parameters for each group
    for k, group in enumerate(f_groups):
        # subtract all other sines from the data, they are fixed now
        resid = flux - ts.sum_sines(time, np.delete(res_f_n, group), np.delete(res_a_n, group),
                                     np.delete(res_ph_n, group))

        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(time, resid, res_const, res_slope, res_f_n[group],
                                    res_a_n[group], res_ph_n[group], i_chunks, logger=None)

        res_const, res_slope, out_f_n, out_a_n, out_ph_n = output
        res_f_n[group] = out_f_n
        res_a_n[group] = out_a_n
        res_ph_n[group] = out_ph_n

        if logger is not None:
            model_linear = ts.linear_curve(time, res_const, res_slope, i_chunks)
            model_sinusoid = ts.sum_sines(time, res_f_n, res_a_n, res_ph_n)
            resid = flux - model_linear - model_sinusoid
            bic = gof.calc_bic(resid, 2 * n_chunk + 3 * n_sin)
            logger.extra(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')

    return res_const, res_slope, res_f_n, res_a_n, res_ph_n


@nb.njit(cache=True)
def objective_sinusoids_harmonics(params, time, flux, harmonic_n, i_chunks):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [p_orb, constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...,
         ampl_h1, ampl_h2, ..., phase_h1, phase_h2, ...]
        where _hi indicates harmonics.
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    harmonic_n: numpy.ndarray[Any, dtype[int]]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_harm = len(harmonic_n)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_chunk - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm

    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_chunk]
    slope = params[1 + n_chunk:1 + 2 * n_chunk]
    f_n = np.zeros(n_f_tot)
    f_n[:n_sin] = params[1 + 2 * n_chunk:1 + 2 * n_chunk + n_sin]
    f_n[n_sin:] = harmonic_n / p_orb
    a_n = np.zeros(n_f_tot)
    a_n[:n_sin] = params[1 + 2 * n_chunk + n_sin:1 + 2 * n_chunk + 2 * n_sin]
    a_n[n_sin:] = params[1 + 2 * n_chunk + 3 * n_sin:1 + 2 * n_chunk + 3 * n_sin + n_harm]
    ph_n = np.zeros(n_f_tot)
    ph_n[:n_sin] = params[1 + 2 * n_chunk + 2 * n_sin:1 + 2 * n_chunk + 3 * n_sin]
    ph_n[n_sin:] = params[1 + 2 * n_chunk + 3 * n_sin + n_harm:1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm]

    # make the linear and sinusoid model
    model_linear = ts.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = ts.sum_sines(time, f_n, a_n, ph_n)

    # calculate the likelihood (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    ln_likelihood = gof.calc_iid_normal_likelihood(resid)

    return -ln_likelihood


@nb.njit(cache=True, parallel=True)
def jacobian_sinusoids_harmonics(params, time, flux, harmonic_n, i_chunks):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves
    plus a set of harmonic frequencies.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s).
        Has to be a flat array and are ordered in the following way:
        [p_orb, constant1, constant2, ..., slope1, slope2, ...,
         freq1, freg2, ..., ampl1, ampl2, ..., phase1, phase2, ...,
         ampl_h1, ampl_h2, ..., phase_h1, phase_h2, ...]
        where _hi indicates harmonics.
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    harmonic_n: numpy.ndarray[Any, dtype[int]]
        Integer indicating which harmonic each index in 'harmonics'
        points to. n=1 for the base frequency (=orbital frequency)
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    time_ms = time - np.mean(time)
    n_harm = len(harmonic_n)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_chunk - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm

    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_chunk]
    slope = params[1 + n_chunk:1 + 2 * n_chunk]
    f_n = np.zeros(n_f_tot)
    f_n[:n_sin] = params[1 + 2 * n_chunk:1 + 2 * n_chunk + n_sin]
    f_n[n_sin:] = harmonic_n / p_orb
    a_n = np.zeros(n_f_tot)
    a_n[:n_sin] = params[1 + 2 * n_chunk + n_sin:1 + 2 * n_chunk + 2 * n_sin]
    a_n[n_sin:] = params[1 + 2 * n_chunk + 3 * n_sin:1 + 2 * n_chunk + 3 * n_sin + n_harm]
    ph_n = np.zeros(n_f_tot)
    ph_n[:n_sin] = params[1 + 2 * n_chunk + 2 * n_sin:1 + 2 * n_chunk + 3 * n_sin]
    ph_n[n_sin:] = params[1 + 2 * n_chunk + 3 * n_sin + n_harm:1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm]

    # make the linear and sinusoid model and subtract from the flux
    model_linear = ts.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = ts.sum_sines(time, f_n, a_n, ph_n)
    resid = flux - model_linear - model_sinusoid

    # common factor
    two_pi_t = 2 * np.pi * time_ms

    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_chunk)  # calculated per sector
    df_1b = -len(time) / np.sum(resid**2)

    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_chunk)
    for i in nb.prange(n_chunk):
        s = i_chunks[i]
        i_s = i + n_chunk
        df_1a[i] = -len(time[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (time[s[0]:s[1]] - np.mean(time[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a

    # calculate the rest of the jacobian, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin + 2 * n_harm + 1)
    for i in nb.prange(n_sin):
        i_f = i + 1  # index of df_2b for f_n
        i_a = i + n_sin + 1  # index of df_2b for a_n
        i_ph = i + 2 * n_sin + 1  # index of df_2b for ph_n
        df_2b[i_f] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='ph'))

    for i in nb.prange(n_harm):
        i_h = n_sin + i  # index shifted to the harmonics in f_n, a_n, ph_n
        i_a = i + 3 * n_sin + 1  # index of df_2b for a_h
        i_ph = i + 3 * n_sin + n_harm + 1  # index of df_2b for ph_h
        df_2b[0] -= np.sum(resid * dsin_dx(two_pi_t, f_n[i_h], a_n[i_h], ph_n[i_h], d='p_orb', p_orb=p_orb))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f_n[i_h], a_n[i_h], ph_n[i_h], d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f_n[i_h], a_n[i_h], ph_n[i_h], d='ph'))

    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)

    return jac


def fit_multi_sinusoid_harmonics(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, logger=None):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies.

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
        res_p_orb: float
            Updated Orbital period in days
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_f_n: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_a_n: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_ph_n: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_f_tot = len(f_n)
    n_harm = len(harmonics)
    n_sin = n_f_tot - n_harm  # each independent sine has freq, ampl and phase
    non_harm = np.delete(np.arange(n_f_tot), harmonics)

    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(time)
    f_low = 0.01 / t_tot

    # do the fit
    par_init = np.concatenate(([p_orb], np.atleast_1d(const), np.atleast_1d(slope), np.delete(f_n, harmonics),
                               np.delete(a_n, harmonics), np.delete(ph_n, harmonics), a_n[harmonics], ph_n[harmonics]))
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_chunk)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (time, flux, harmonic_n, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init, args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_chunk]
    res_slope = result.x[1 + n_chunk:1 + 2 * n_chunk]
    res_f_n = np.zeros(n_f_tot)
    res_f_n[non_harm] = result.x[1 + 2 * n_chunk:1 + 2 * n_chunk + n_sin]
    res_f_n[harmonics] = harmonic_n / res_p_orb
    res_a_n = np.zeros(n_f_tot)
    res_a_n[non_harm] = result.x[1 + 2 * n_chunk + n_sin:1 + 2 * n_chunk + 2 * n_sin]
    res_a_n[harmonics] = result.x[1 + 2 * n_chunk + 3 * n_sin:1 + 2 * n_chunk + 3 * n_sin + n_harm]
    res_ph_n = np.zeros(n_f_tot)
    res_ph_n[non_harm] = result.x[1 + 2 * n_chunk + 2 * n_sin:1 + 2 * n_chunk + 3 * n_sin]
    res_ph_n[harmonics] = result.x[1 + 2 * n_chunk + 3 * n_sin + n_harm:1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm]

    if logger is not None:
        model_linear = ts.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = ts.sum_sines(time, res_f_n, res_a_n, res_ph_n)
        resid = flux - model_linear - model_sinusoid
        bic = gof.calc_bic(resid, 1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm)
        logger.extra(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
                     f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    return res_p_orb, res_const, res_slope, res_f_n, res_a_n, res_ph_n


def fit_multi_sinusoid_harmonics_per_group(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks,
                                           logger=None):
    """Perform the multi-sinusoid, non-linear least-squares fit with harmonic frequencies
    per frequency group

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
        res_p_orb: float
            Updated Orbital period in days
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_f_n: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_a_n: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_ph_n: numpy.ndarray[Any, dtype[float]]
            Updated ph_n of a number of sine waves

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve
    the fits per group of 15-20 frequencies, leaving the other frequencies
    as fixed parameters.
    Contrary to multi_sine_NL_LS_fit_per_group, the groups don't have to be provided
    as they are made with the default parameters of ut.group_frequencies_for_fit.
    The orbital harmonics are always the first group.
    """
    # get harmonics and group the remaining frequencies
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    indices = np.arange(len(f_n))
    i_non_harm = np.delete(indices, harmonics)
    f_groups = star_shine.core.frequency_sets.group_frequencies_for_fit(a_n[i_non_harm], g_min=45, g_max=50)
    f_groups = [i_non_harm[g] for g in f_groups]  # convert back to indices for full f_n list
    n_groups = len(f_groups)
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_harm = len(harmonics)
    n_sin = len(f_n) - n_harm

    # make a copy of the initial parameters
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_f_n, res_a_n, res_ph_n = np.copy(f_n), np.copy(a_n), np.copy(ph_n)

    # fit the harmonics (first group)
    # remove harmonic frequencies
    resid = flux - ts.sum_sines(time, np.delete(res_f_n, harmonics), np.delete(res_a_n, harmonics),
                                   np.delete(res_ph_n, harmonics))
    par_init = np.concatenate(([p_orb], res_const, res_slope, a_n[harmonics], ph_n[harmonics]))
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_chunk)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (time, resid, harmonic_n, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init,  args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_chunk]
    res_slope = result.x[1 + n_chunk:1 + 2 * n_chunk]
    res_f_n[harmonics] = harmonic_n / res_p_orb
    res_a_n[harmonics] = result.x[1 + 2 * n_chunk:1 + 2 * n_chunk + n_harm]
    res_ph_n[harmonics] = result.x[1 + 2 * n_chunk + n_harm:1 + 2 * n_chunk + 2 * n_harm]

    if logger is not None:
        model_linear = ts.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = ts.sum_sines(time, res_f_n, res_a_n, res_ph_n)
        resid = flux - model_linear - model_sinusoid
        bic = gof.calc_bic(resid, 1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm)
        logger.extra(f'Fit of harmonics - BIC: {bic:1.2f}. N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    # update the parameters for each group
    for k, group in enumerate(f_groups):
        # subtract all other sines from the data, they are fixed now
        resid = flux - ts.sum_sines(time, np.delete(res_f_n, group), np.delete(res_a_n, group),
                                       np.delete(res_ph_n, group))

        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(time, resid, res_const, res_slope, res_f_n[group],
                                    res_a_n[group], res_ph_n[group], i_chunks, logger=None)
        res_const, res_slope, out_f_n, out_a_n, out_ph_n = output
        res_f_n[group] = out_f_n
        res_a_n[group] = out_a_n
        res_ph_n[group] = out_ph_n

        if logger is not None:
            model_linear = ts.linear_curve(time, res_const, res_slope, i_chunks)
            model_sinusoid = ts.sum_sines(time, res_f_n, res_a_n, res_ph_n)
            resid_new = flux - (model_linear + model_sinusoid)
            bic = gof.calc_bic(resid_new, 1 + 2 * n_chunk + 3 * n_sin + 2 * n_harm)
            logger.extra(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')

    return res_p_orb, res_const, res_slope, res_f_n, res_a_n, res_ph_n
