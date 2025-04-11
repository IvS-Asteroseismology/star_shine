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

from star_shine.core import utility as ut, analysis_functions as af, timeseries_functions as tsf


@nb.njit(cache=True)
def dsin_dx(two_pi_t, f, a, ph, d='f', p_orb=0):
    """The derivative of a sine wave at times t,
    where x is on of the parameters.

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
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    const = params[:n_sect]
    slope = params[n_sect:2 * n_sect]
    freqs = params[2 * n_sect:2 * n_sect + n_sin]
    ampls = params[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    phases = params[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = tsf.sum_sines(time, freqs, ampls, phases)

    # calculate the likelihood (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    ln_likelihood = tsf.calc_likelihood(residual=resid, time=time, flux_err=None, func=tsf.calc_iid_normal_likelihood)

    return -ln_likelihood


@nb.njit(cache=True)
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
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    const = params[:n_sect]
    slope = params[n_sect:2 * n_sect]
    freqs = params[2 * n_sect:2 * n_sect + n_sin]
    ampls = params[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    phases = params[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = tsf.sum_sines(time, freqs, ampls, phases)

    # calculate the likelihood derivative (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    two_pi_t = 2 * np.pi * time_ms

    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_sect)  # calculated per sector
    df_1b = -len(time) / np.sum(resid**2)

    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_sect)
    for i, (co, sl, s) in enumerate(zip(const, slope, i_chunks)):
        i_s = i + n_sect
        df_1a[i] = -len(time[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (time[s[0]:s[1]] - np.mean(time[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a

    # calculate the rest of the jacobian for the sinusoid parameters, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin)
    for i, (f, a, ph) in enumerate(zip(freqs, ampls, phases)):
        i_a = i + n_sin
        i_ph = i + 2 * n_sin
        df_2b[i] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))

    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)

    return jac


def fit_multi_sinusoid(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, verbose=False):
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
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    tuple
        A tuple containing the following elements:
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_freqs: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_ampls: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_phases: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = len(f_n)  # each sine has freq, ampl and phase

    # we don't want the frequencies to go lower than about 1/T/100
    t_tot = np.ptp(time)
    f_low = 0.01 / t_tot

    # do the fit
    par_init = np.concatenate((res_const, res_slope, res_freqs, res_ampls, res_phases))
    par_bounds = [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    arguments = (time, flux, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids, jac=jacobian_sinusoids, x0=par_init, args=arguments,
                                  method='L-BFGS-B', bounds=par_bounds, options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_const = result.x[0:n_sect]
    res_slope = result.x[n_sect:2 * n_sect]
    res_freqs = result.x[2 * n_sect:2 * n_sect + n_sin]
    res_ampls = result.x[2 * n_sect + n_sin:2 * n_sect + 2 * n_sin]
    res_phases = result.x[2 * n_sect + 2 * n_sin:2 * n_sect + 3 * n_sin]

    if verbose:
        model_linear = tsf.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = tsf.sum_sines(time, res_freqs, res_ampls, res_phases)
        resid = flux - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 2 * n_sect + 3 * n_sin)
        print(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    return res_const, res_slope, res_freqs, res_ampls, res_phases


def fit_multi_sinusoid_per_group(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, verbose=False):
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
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    tuple
        A tuple containing the following elements:
        res_const: numpy.ndarray[Any, dtype[float]]
            Updated y-intercepts of a piece-wise linear curve
        res_slope: numpy.ndarray[Any, dtype[float]]
            Updated slopes of a piece-wise linear curve
        res_freqs: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_ampls: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_phases: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    In reducing the overall runtime of the NL-LS fit, this will improve the
    fits per group of 15-20 frequencies, leaving the other frequencies as
    fixed parameters.
    """
    f_groups = ut.group_frequencies_for_fit(a_n, g_min=20, g_max=25)
    n_groups = len(f_groups)
    n_sect = len(i_chunks)
    n_sin = len(f_n)

    # make a copy of the initial parameters
    res_const = np.copy(const)
    res_slope = np.copy(slope)
    res_freqs = np.copy(f_n)
    res_ampls = np.copy(a_n)
    res_phases = np.copy(ph_n)

    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')

        # subtract all other sines from the data, they are fixed now
        resid = flux - tsf.sum_sines(time, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))

        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(time, resid, res_const, res_slope, res_freqs[group],
                                    res_ampls[group], res_phases[group], i_chunks, verbose=False)

        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases

        if verbose:
            model_linear = tsf.linear_curve(time, res_const, res_slope, i_chunks)
            model_sinusoid = tsf.sum_sines(time, res_freqs, res_ampls, res_phases)
            resid = flux - model_linear - model_sinusoid
            bic = tsf.calc_bic(resid, 2 * n_sect + 3 * n_sin)
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')

    return res_const, res_slope, res_freqs, res_ampls, res_phases


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
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm

    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_sect]
    slope = params[1 + n_sect:1 + 2 * n_sect]
    freqs = np.zeros(n_f_tot)
    freqs[:n_sin] = params[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    freqs[n_sin:] = harmonic_n / p_orb
    ampls = np.zeros(n_f_tot)
    ampls[:n_sin] = params[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    ampls[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    phases = np.zeros(n_f_tot)
    phases[:n_sin] = params[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    phases[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]

    # make the linear and sinusoid model
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = tsf.sum_sines(time, freqs, ampls, phases)

    # calculate the likelihood (minus this for minimisation)
    resid = flux - model_linear - model_sinusoid
    ln_likelihood = tsf.calc_likelihood(residual=resid, time=time, flux_err=None, func=tsf.calc_iid_normal_likelihood)

    return -ln_likelihood


@nb.njit(cache=True)
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
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_sin = (len(params) - 2 * n_sect - 1 - 2 * n_harm) // 3  # each sine has freq, ampl and phase
    n_f_tot = n_sin + n_harm

    # separate the parameters
    p_orb = params[0]
    const = params[1:1 + n_sect]
    slope = params[1 + n_sect:1 + 2 * n_sect]
    freqs = np.zeros(n_f_tot)
    freqs[:n_sin] = params[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    freqs[n_sin:] = harmonic_n / p_orb
    ampls = np.zeros(n_f_tot)
    ampls[:n_sin] = params[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    ampls[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    phases = np.zeros(n_f_tot)
    phases[:n_sin] = params[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    phases[n_sin:] = params[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]

    # make the linear and sinusoid model and subtract from the flux
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = tsf.sum_sines(time, freqs, ampls, phases)
    resid = flux - model_linear - model_sinusoid

    # common factor
    two_pi_t = 2 * np.pi * time_ms

    # factor 1 of df/dx: -n / S
    df_1a = np.zeros(n_sect)  # calculated per sector
    df_1b = -len(time) / np.sum(resid**2)

    # calculate the rest of the jacobian for the linear parameters, factor 2 of df/dx:
    df_2a = np.zeros(2 * n_sect)
    for i, (co, sl, s) in enumerate(zip(const, slope, i_chunks)):
        i_s = i + n_sect
        df_1a[i] = -len(time[s[0]:s[1]]) / np.sum(resid[s[0]:s[1]]**2)
        df_2a[i] = np.sum(resid[s[0]:s[1]])
        df_2a[i_s] = np.sum(resid[s[0]:s[1]] * (time[s[0]:s[1]] - np.mean(time[s[0]:s[1]])))
    df_1a = np.append(df_1a, df_1a)  # copy to double length
    jac_lin = df_1a * df_2a

    # calculate the rest of the jacobian, factor 2 of df/dx:
    df_2b = np.zeros(3 * n_sin + 2 * n_harm + 1)
    for i, (f, a, ph) in enumerate(zip(freqs[:n_sin], ampls[:n_sin], phases[:n_sin])):
        i_f = i + 1
        i_a = i + n_sin + 1
        i_ph = i + 2 * n_sin + 1
        df_2b[i_f] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))
    for i, (f, a, ph) in enumerate(zip(freqs[n_sin:], ampls[n_sin:], phases[n_sin:])):
        i_a = i + 3 * n_sin + 1
        i_ph = i + 3 * n_sin + n_harm + 1
        df_2b[0] -= np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='p_orb', p_orb=p_orb))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f, a, ph, d='ph'))

    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac_sin = df_1b * df_2b
    jac = np.append(jac_lin, jac_sin)

    return jac


def fit_multi_sinusoid_harmonics(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=False):
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
    verbose: bool
        If set to True, this function will print some information

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
        res_freqs: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_ampls: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_phases: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

    Notes
    -----
    Strictly speaking it is doing a maximum log-likelihood fit, but that is
    in essence identical (and numerically more stable due to the logarithm).
    """
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
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
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(f_low, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_sin)] + [(None, None) for _ in range(n_sin)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (time, flux, harmonic_n, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init, args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs = np.zeros(n_f_tot)
    res_freqs[non_harm] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_sin]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls = np.zeros(n_f_tot)
    res_ampls[non_harm] = result.x[1 + 2 * n_sect + n_sin:1 + 2 * n_sect + 2 * n_sin]
    res_ampls[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin:1 + 2 * n_sect + 3 * n_sin + n_harm]
    res_phases = np.zeros(n_f_tot)
    res_phases[non_harm] = result.x[1 + 2 * n_sect + 2 * n_sin:1 + 2 * n_sect + 3 * n_sin]
    res_phases[harmonics] = result.x[1 + 2 * n_sect + 3 * n_sin + n_harm:1 + 2 * n_sect + 3 * n_sin + 2 * n_harm]

    if verbose:
        model_linear = tsf.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = tsf.sum_sines(time, res_freqs, res_ampls, res_phases)
        resid = flux - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit convergence: {result.success} - BIC: {bic:1.2f}. '
              f'N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases


def fit_multi_sinusoid_harmonics_per_group(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks,
                                           verbose=False):
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
    verbose: bool
        If set to True, this function will print some information

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
        res_freqs: numpy.ndarray[Any, dtype[float]]
            Updated frequencies of a number of sine waves
        res_ampls: numpy.ndarray[Any, dtype[float]]
            Updated amplitudes of a number of sine waves
        res_phases: numpy.ndarray[Any, dtype[float]]
            Updated phases of a number of sine waves

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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    indices = np.arange(len(f_n))
    i_non_harm = np.delete(indices, harmonics)
    f_groups = ut.group_frequencies_for_fit(a_n[i_non_harm], g_min=20, g_max=25)
    f_groups = [i_non_harm[g] for g in f_groups]  # convert back to indices for full f_n list
    n_groups = len(f_groups)
    n_sect = len(i_chunks)  # each sector has its own slope (or two)
    n_harm = len(harmonics)
    n_sin = len(f_n) - n_harm

    # make a copy of the initial parameters
    res_const = np.copy(np.atleast_1d(const))
    res_slope = np.copy(np.atleast_1d(slope))
    res_freqs, res_ampls, res_phases = np.copy(f_n), np.copy(a_n), np.copy(ph_n)

    # fit the harmonics (first group)
    if verbose:
        print(f'Fit of harmonics', end='\r')

    # remove harmonic frequencies
    resid = flux - tsf.sum_sines(time, np.delete(res_freqs, harmonics), np.delete(res_ampls, harmonics),
                                   np.delete(res_phases, harmonics))
    par_init = np.concatenate(([p_orb], res_const, res_slope, a_n[harmonics], ph_n[harmonics]))
    par_bounds = [(0, None)] + [(None, None) for _ in range(2 * n_sect)]
    par_bounds = par_bounds + [(0, None) for _ in range(n_harm)] + [(None, None) for _ in range(n_harm)]
    arguments = (time, resid, harmonic_n, i_chunks)
    result = sp.optimize.minimize(objective_sinusoids_harmonics, jac=jacobian_sinusoids_harmonics,
                                  x0=par_init,  args=arguments, method='L-BFGS-B', bounds=par_bounds,
                                  options={'maxiter': 10**4 * len(par_init)})

    # separate results
    res_p_orb = result.x[0]
    res_const = result.x[1:1 + n_sect]
    res_slope = result.x[1 + n_sect:1 + 2 * n_sect]
    res_freqs[harmonics] = harmonic_n / res_p_orb
    res_ampls[harmonics] = result.x[1 + 2 * n_sect:1 + 2 * n_sect + n_harm]
    res_phases[harmonics] = result.x[1 + 2 * n_sect + n_harm:1 + 2 * n_sect + 2 * n_harm]

    if verbose:
        model_linear = tsf.linear_curve(time, res_const, res_slope, i_chunks)
        model_sinusoid = tsf.sum_sines(time, res_freqs, res_ampls, res_phases)
        resid = flux - model_linear - model_sinusoid
        bic = tsf.calc_bic(resid, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
        print(f'Fit of harmonics - BIC: {bic:1.2f}. N_iter: {int(result.nit)}, N_fev: {int(result.nfev)}.')

    # update the parameters for each group
    for k, group in enumerate(f_groups):
        if verbose:
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)}', end='\r')

        # subtract all other sines from the data, they are fixed now
        resid = flux - tsf.sum_sines(time, np.delete(res_freqs, group), np.delete(res_ampls, group),
                                       np.delete(res_phases, group))

        # fit only the frequencies in this group (constant and slope are also fitted still)
        output = fit_multi_sinusoid(time, resid, res_const, res_slope, res_freqs[group],
                                    res_ampls[group], res_phases[group], i_chunks, verbose=False)
        res_const, res_slope, out_freqs, out_ampls, out_phases = output
        res_freqs[group] = out_freqs
        res_ampls[group] = out_ampls
        res_phases[group] = out_phases

        if verbose:
            model_linear = tsf.linear_curve(time, res_const, res_slope, i_chunks)
            model_sinusoid = tsf.sum_sines(time, res_freqs, res_ampls, res_phases)
            resid_new = flux - (model_linear + model_sinusoid)
            bic = tsf.calc_bic(resid_new, 1 + 2 * n_sect + 3 * n_sin + 2 * n_harm)
            print(f'Fit of group {k + 1} of {n_groups} - N_f(group)= {len(group)} - BIC: {bic:1.2f}')

    return res_p_orb, res_const, res_slope, res_freqs, res_ampls, res_phases
