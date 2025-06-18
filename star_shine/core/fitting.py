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
from scipy.optimize import minimize

from star_shine.core import time_series as tms, model as mdl, goodness_of_fit as gof
from star_shine.core import frequency_sets as frs, periodogram as pdg


@nb.njit(cache=True)
def dsin_dx(two_pi_t, f, a, ph, d='f', f_base=0):
    """The derivative of a sine wave at times t, where x is on of the parameters.

    Parameters
    ----------
    two_pi_t: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series times two pi.
    f: float
        The frequency of a sine wave.
    a: float
        The amplitude of a sine wave.
    ph: float
        The phase of a sine wave.
    d: string
        Which derivative to take. Choose f, a, ph, h (for harmonics).
    f_base: float
        Base frequency of the harmonic series.

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of the derivative of a sine wave to f.

    Notes
    -----
    Make sure the phases correspond to the given time zero point.
    If d='h', it is assumed that f is a harmonic and the base frequency f_base is provided
    """
    if d == 'f':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t
    elif d == 'a':
        model_deriv = np.sin(two_pi_t * f + ph)
    elif d == 'ph':
        model_deriv = a * np.cos(two_pi_t * f + ph)
    elif d == 'h':
        model_deriv = a * np.cos(two_pi_t * f + ph) * two_pi_t * f / f_base  # (f / f_b = n_h)
    else:
        model_deriv = np.zeros(len(two_pi_t))

    return model_deriv


@nb.njit(cache=True)
def objective_sinusoids(params, time, flux, i_chunks, h_base, h_mult):
    """The objective function to give to scipy.optimize.minimize for a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [f_base_1, f_base_2, ..., const_1, const_2, ..., slope_1, slope_2, ...,
        f_1, f_2, ..., a_1, a_2, ..., p_1, p_2, ...]
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    h_base: numpy.ndarray[Any, dtype[int]]
         Indices of the base frequencies in params.
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each harmonic.

    Returns
    -------
    float
        Minus the (natural)log-likelihood of the residuals

    See Also
    --------
    linear_curve and sum_sines for the definition of the parameters.
    """
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_base = len(h_base[h_mult == 1])
    n_sin = (len(params) - n_base - 2 * n_chunk) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    f_base = params[:n_base]
    const = params[n_base:n_base + n_chunk]
    slope = params[n_base + n_chunk:n_base + 2 * n_chunk]
    f_n = params[n_base + 2 * n_chunk:n_base + 2 * n_chunk + n_sin]
    a_n = params[n_base + 2 * n_chunk + n_sin:n_base + 2 * n_chunk + 2 * n_sin]
    ph_n = params[n_base + 2 * n_chunk + 2 * n_sin:n_base + 2 * n_chunk + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = mdl.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)

    # interim residual
    resid = flux - model_linear - model_sinusoid

    # do the harmonics with linear regression (constant term should have already been taken care of)
    harmonic_betas, design_matrix, _ = solve_harmonic_sinusoids(time, resid, f_base[h_base], h_mult)

    # reconstruct the predicted flux based on our linear model and subtract
    resid -= design_matrix @ harmonic_betas

    # calculate the likelihood (minus this for minimisation)
    ln_likelihood = gof.calc_iid_normal_likelihood(resid)

    return -ln_likelihood


@nb.njit(cache=True, parallel=True)
def jacobian_sinusoids(params, time, flux, i_chunks, h_base, h_mult):
    """The jacobian function to give to scipy.optimize.minimize for a sum of sine waves.

    Parameters
    ----------
    params: numpy.ndarray[Any, dtype[float]]
        The parameters of a set of sine waves and linear curve(s)
        Has to be a flat array and are ordered in the following way:
        [f_base_1, f_base_2, ..., const_1, const_2, ..., slope_1, slope_2, ...,
        f_1, f_2, ..., a_1, a_2, ..., p_1, p_2, ...]
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    h_base: numpy.ndarray[Any, dtype[int]]
         Indices of the base frequencies in params.
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each harmonic.

    Returns
    -------
    float
        The derivative of minus the (natural)log-likelihood of the residuals

    See Also
    --------
    objective_sinusoids
    """
    n_chunk = len(i_chunks)  # each sector has its own slope (or two)
    n_base = len(h_base[h_mult == 1])
    n_sin = (len(params) - n_base - 2 * n_chunk) // 3  # each sine has freq, ampl and phase

    # separate the parameters
    f_base = params[:n_base]
    const = params[n_base:n_base + n_chunk]
    slope = params[n_base + n_chunk:n_base + 2 * n_chunk]
    f_n = params[n_base + 2 * n_chunk:n_base + 2 * n_chunk + n_sin]
    a_n = params[n_base + 2 * n_chunk + n_sin:n_base + 2 * n_chunk + 2 * n_sin]
    ph_n = params[n_base + 2 * n_chunk + 2 * n_sin:n_base + 2 * n_chunk + 3 * n_sin]

    # make the linear and sinusoid model
    model_linear = mdl.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)

    # interim residual
    resid = flux - model_linear - model_sinusoid

    # do the harmonics with linear regression
    harmonic_betas, design_matrix, _ = solve_harmonic_sinusoids(time, resid, f_base[h_base], h_mult)

    # reconstruct the predicted flux based on our linear model and subtract
    resid -= design_matrix @ harmonic_betas

    two_pi_t = 2 * np.pi * (time - np.mean(time))

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
        i_f = n_base + i
        i_a = n_base + i + n_sin  # index of df_2b for a_n
        i_ph = n_base + i + 2 * n_sin  # index of df_2b for ph_n
        df_2b[i_f] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='f'))
        df_2b[i_a] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='a'))
        df_2b[i_ph] = np.sum(resid * dsin_dx(two_pi_t, f_n[i], a_n[i], ph_n[i], d='ph'))

    jac_sin = df_1b * df_2b

    # calculate the rest of the jacobian for the harmonic base frequencies, factor 2 of df/dx:
    f_h = f_base[h_base] * h_mult
    a_h = np.sqrt(harmonic_betas[1::2] ** 2 + harmonic_betas[2::2] ** 2)
    ph_h = np.arctan2(harmonic_betas[2::2], harmonic_betas[1::2])
    df_2c = np.zeros(n_base)
    for i in nb.prange(n_base):
        h_j = np.arange(len(h_base))[h_base == i]
        for j in h_j:
            df_2c[i] += np.sum(resid * dsin_dx(two_pi_t, f_h[j], a_h[j], ph_h[j], d='h', f_base=f_base[i]))

    jac_harm = df_1b * df_2c

    # jacobian = df/dx = df/dy * dy/dx (f is objective function, y is model)
    jac = np.concatenate((jac_harm, jac_lin, jac_sin))

    return jac


@nb.njit(cache=True)
def make_design_matrix_harmonic(time, f_base, h_mult, t_shift=True):
    """Generate a design matrix for a series of harmonics and a constant term.

    Also returns weights for each model feature.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    f_base: numpy.ndarray[Any, dtype[float]]
        Base frequencies of the harmonics (of multiple series).
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each harmonic.
    t_shift: bool
        Mean center the time axis.

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The design matrix. N by M where N = len(time) and M = 1 + 2 len(h_mult).

    Notes
    -----
    Based on code from Puffins, by C. Johnston, D.W. Hogg, and N.L. Eisner;
    https://github.com/colej/puffins/
    Which is itself based on D. W. Hogg & S. Villar (2021, https://arxiv.org/pdf/2101.07256)
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    # sinusoid angle term
    two_pi_f_t = 2. * np.pi * (f_base * h_mult)[None,:] * (time - mean_t)[:, None]

    # make empty matrix
    n_param = 1 + 2 * len(h_mult)
    design_matrix = np.ones((len(time), n_param))

    # constant term
    design_matrix[:, 0] = np.ones_like(time)
    # cosine terms
    design_matrix[:, 1::2] = np.cos(two_pi_f_t)
    # sine terms
    design_matrix[:, 2::2] = np.sin(two_pi_f_t)

    return design_matrix


@nb.njit(cache=True)
def make_feature_weights_harmonic(h_mult, s=0.1):
    """Generate the weights for the features of a harmonic series.

    Parameters
    ----------
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each harmonic (of multiple series).
    s: float
        The kernel width in Fourier space.

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Harmonic feature weights.

    Notes
    -----
    Based on code from Puffins, by C. Johnston, D.W. Hogg, and N.L. Eisner;
    https://github.com/colej/puffins/
    Which is itself based on D. W. Hogg & S. Villar (2021, https://arxiv.org/pdf/2101.07256)
    """
    # make empty array
    n_param = 1 + 2 * len(h_mult)
    feature_weights = np.zeros(n_param)

    # feature weights
    feature_weights[1::2] = (1. + s ** 2 * h_mult ** 2) ** 2
    feature_weights[2::2] = (1. + s ** 2 * h_mult ** 2) ** 2

    return feature_weights


@nb.njit(cache=True)
def solve_fw(x, y, w=None, l=None):
    """Feature weighted least-squares solver.

    Parameters
    ----------
    x: numpy.ndarray[(n, p), dtype[float]]
        Design matrix
    y: numpy.ndarray[(n,), dtype[float]]
        Target values
    l: numpy.ndarray[(p,), dtype[float]]
        Feature weights
    w: numpy.ndarray[(n,), dtype[float]]
        Weights for the targets
        Assumes matrix with diagonal elements as 1/inverse variance on Y

    Returns
    -------
    numpy.ndarray[(p,), dtype[float]]
        Solved coefficients (betas)

    Notes
    -----
    Implements equations 23 and 24 from D. W. Hogg & S. Villar (2021, https://arxiv.org/pdf/2101.07256)
    If weights is not provided, it assumes OLS
    W = 1/C^2, where C is the uncertainty on the target values

    Based on code from Puffins, by C. Johnston, D.W. Hogg, and N.L. Eisner;
    https://github.com/colej/puffins/
    """

    n, p = x.shape

    if w is None:
        w = np.ones(n)
    if l is None:
        l = np.ones(p)

    # Assumes the approximation of a periodic version of the Matern 3/2 kernal
    if n > p:  # Overdetermined
        # Use eq. 23 from https://arxiv.org/pdf/2101.07256
        # beta_hat = (X.T * W * X + L)^-1 * X.T * W * y

        xt_w = x.T * w
        xt_w_x = xt_w @ x
        # xt_w_x[np.diag_indices(p)] += l
        # Add l to the diagonal of xt_w_x
        np.fill_diagonal(xt_w_x, np.diag(xt_w_x) + l)

        # solve the linalg eq.
        betas = np.linalg.lstsq(xt_w_x, xt_w @ y, rcond=1e-14)[0]

    else:  # underdetermined
        # Use eq. 24 from https://arxiv.org/pdf/2101.07256
        # beta_hat = L^-1 @ X.T (X @ L^-1 @ X.T + 1/W)^-1 y

        linv_xt = (x / l).T
        x_linv_xt = x @ linv_xt
        # x_linv_xt[np.diag_indices(n)] += 1. / w
        # Add 1/w to the diagonal of x_linv_xt
        np.fill_diagonal(x_linv_xt, np.diag(x_linv_xt) + 1. / w)

        # solve the linalg eq.
        betas = linv_xt @ np.linalg.lstsq(x_linv_xt, y, rcond=1e-14)[0]

    return betas


@nb.njit(cache=True)
def solve_harmonic_sinusoids(time, flux, f_base, h_mult):
    """Solve for the amplitudes and phases of one or more harmonic series.

    Also returns the design matrix and feature weights.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    f_base: numpy.ndarray[Any, dtype[float]]
        Base frequencies of the harmonics (of multiple series).
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each harmonic.

    Returns
    -------
    tuple
        numpy.ndarray[Any, dtype[float]]
            Harmonic amplitude solutions (cosine and sine amplitudes).
        numpy.ndarray[Any, dtype[float]]
            The design matrix. N by M where N = len(time) and M = 1 + 2 len(h_mult).
        numpy.ndarray[Any, dtype[float]]
            Harmonic feature weights.
    """
    # construct the design matrix associated with each datapoint-harmonic pair
    design_matrix = make_design_matrix_harmonic(time, f_base=f_base, h_mult=h_mult)

    # higher frequencies are penalised by weights
    feature_weights = make_feature_weights_harmonic(h_mult, s=0.01)  # use low s for weights that don't do much

    # solve the linalg for the model parameters
    harmonic_betas = solve_fw(design_matrix, flux, l=feature_weights, w=None)

    return harmonic_betas, design_matrix, feature_weights


def fit_multi_sinusoid_grouped(ts_model, g_min=45, g_max=50, logger=None):
    """Perform the multi-sinusoid, non-linear least-squares fit per sinusoid group.

    This will fit per groups of 45-50 sinusoids (by default), leaving the other sinusoid parameters fixed,
    reducing the overall runtime of the NL-LS fit.
    If present, harmonics are always fit simultaneously with each group.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    g_min: int
        Minimum group size
    g_max: int
        Maximum group size (g_max > g_min)
    logger: logging.Logger, optional
        Instance of the logging library.

    Notes
    -----
    ts_model must have been cleaned up before use here.
    """
    # get the harmonic base frequencies and harmonic parameters
    h_base_unique, h_base_map = ts_model.sinusoid.get_h_base_map()
    harmonics = ts_model.sinusoid.harmonics
    h_base = ts_model.sinusoid.h_base[harmonics]  # only include the harmonics
    h_mult = ts_model.sinusoid.h_mult[harmonics]  # only include the harmonics

    # indices of harmonic and non-harmonic sinusoids
    i_h = np.arange(len(harmonics))[harmonics]
    i_non_h = np.arange(len(harmonics))[~harmonics]

    # remove harmonics from the sinusoids (to be fitted)
    f_n = ts_model.sinusoid.f_n
    a_n = ts_model.sinusoid.a_n
    ph_n = ts_model.sinusoid.ph_n

    # prepare the group lists and some numbers
    f_groups = frs.group_frequencies_for_fit(a_n[i_non_h], g_min=g_min, g_max=g_max, indices=i_non_h)
    n_groups = len(f_groups)
    n_chunk = len(ts_model.i_chunks)
    n_sin = len(f_n)
    n_base = len(h_base_unique)

    # we don't want the frequencies to go lower than about 1/100/T
    f_low = 0.01 / ts_model.t_tot

    # prepare fixed fit input
    par_bounds = [(f_low, None) for _ in range(n_base)]  # harmonic series base frequencies
    par_bounds += [(None, None) for _ in range(2 * n_chunk)]  # constants and slopes

    # update the parameters for each group
    for k, group in enumerate(f_groups):
        # temporarily exclude group and harmonic sinusoids
        n_sin_g = len(group)
        ts_model.exclude_sinusoids(group)
        ts_model.exclude_sinusoids(i_h)

        # get current residual (excluding group, harmonics, and linear curve)
        residual = ts_model.flux - ts_model.sinusoid.sinusoid_model

        # get the periodogram amplitudes for amplitude limits
        pd_ampls, _ = pdg.scargle_ampl_phase(ts_model.time, residual, f_n[group])

        # prepare fit input
        par_init_i = np.concatenate((ts_model.sinusoid.f_base, ts_model.linear.const, ts_model.linear.slope,
                                     f_n[group], a_n[group], ph_n[group]))
        par_bounds_i = par_bounds + [(f_low, None) for _ in range(n_sin_g)]  # frequencies of free sinusoids
        par_bounds_i += [(0, 2 * pd_ampls[i]) for i in range(n_sin_g)]  # amplitudes of free sinusoids
        par_bounds_i += [(None, None) for _ in range(n_sin_g)]  # phases of free sinusoids
        arguments_i = (ts_model.time, residual, ts_model.i_chunks, h_base_map, h_mult)

        # do the fit
        result = minimize(objective_sinusoids, jac=jacobian_sinusoids, x0=par_init_i, args=arguments_i,
                          method='L-BFGS-B', bounds=par_bounds_i, options={'maxiter': 10 ** 4 * len(par_init_i)})

        # separate results
        f_base = result.x[:n_base]
        const = result.x[n_base:n_base + n_chunk]
        slope = result.x[n_base + n_chunk:n_base + 2 * n_chunk]
        f_g = result.x[n_base + 2 * n_chunk:n_base + 2 * n_chunk + n_sin_g]
        a_g = result.x[n_base + 2 * n_chunk + n_sin_g:n_base + 2 * n_chunk + 2 * n_sin_g]
        ph_g = result.x[n_base + 2 * n_chunk + 2 * n_sin_g:n_base + 2 * n_chunk + 3 * n_sin_g]

        # update model parameters (this re-includes the sinusoids)
        ts_model.set_linear_model(const, slope)
        ts_model.update_sinusoids(f_g, a_g, ph_g, group)

        # get the harmonic parameters with linear regression (using current residual, including group and linear curve)
        harmonic_betas, _, _ = solve_harmonic_sinusoids(ts_model.time, ts_model.residual(), f_base[h_base_map], h_mult)

        # get the harmonic amplitudes and phases from the betas
        cosines = harmonic_betas[1::2]
        sines = harmonic_betas[2::2]
        a_h = np.sqrt(cosines ** 2 + sines ** 2)
        ph_h = np.pi / 2 - np.arctan2(sines, cosines)  # this stupid formula ensures the phases are compatible

        # update the model parameters (this re-includes the harmonic sinusoids)
        f_h = f_base[h_base_map] * h_mult
        ts_model.update_sinusoids(f_h, a_h, ph_h, i_h, h_base_new=h_base, h_mult_new=h_mult)

    if logger is not None:
        logger.extra(f"N_f= {n_sin}, BIC= {ts_model.bic():1.2f} - Optimised model, N_group= {n_groups}",
                     extra={'update': True})

    return None
