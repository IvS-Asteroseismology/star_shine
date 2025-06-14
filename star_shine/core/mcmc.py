"""STAR SHADOW
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains functions for the use in and to perform
Markov Chain Monte Carlo (MCMC) with PyMC3

Code written by: Luc IJspeert
"""

import logging
import numpy as np
import scipy as sp
import scipy.stats

import star_shine.core.frequency_sets

try:
    # optional functionality
    import pymc3 as pm
    import theano.tensor as tt
    import arviz as az
    from fastprogress import fastprogress
except (ImportError, AttributeError) as e:
    if e == AttributeError:
        print('PyMC3 functionality unavailable, likely incompatible numpy version')
    pass

from star_shine.core import analysis as af


def sample_sinusoid(time, flux, const, slope, f_n, a_n, ph_n, c_err, sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                    i_chunks, logger=None):
    """NUTS sampling of a linear + sinusoid + eclipse model
    
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
    c_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the y-intercepts of a number of sine waves
    sl_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slopes of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequencies of a number of sine waves
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitudes of a number of sine waves
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        inf_data: object
            Arviz inference data object
        par_means: list[float]
            Parameter mean values in the following order:
            const, slope, f_n, a_n, ph_n
        par_hdi: list[float]
            Parameter HDI error values, same order as par_means
    """
    # setup
    time_t = time.reshape(-1, 1)  # transposed time
    t_mean = tt.as_tensor_variable(np.mean(time))
    t_mean_s = tt.as_tensor_variable(np.array([np.mean(time[s[0]:s[1]]) for s in i_chunks]))
    lin_shape = (len(const),)
    sin_shape = (len(f_n),)
    # progress bar
    if logger is not None:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)
    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=c_err, shape=lin_shape, testval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=sl_err, shape=lin_shape, testval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (time[s[0]:s[1]] - t_mean_s[k]) for k, s in enumerate(i_chunks)]
        model_linear = tt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n, sigma=f_n_err, lower=0, shape=sin_shape, testval=f_n)
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n, sigma=a_n_err, lower=0, shape=sin_shape, testval=a_n)
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n, kappa=1 / ph_n_err**2, shape=sin_shape, testval=ph_n)
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (time_t - t_mean)) + ph_n_pm), axis=1)
        # full light curve model
        model = model_linear + model_sinusoid
        # observed distribution
        pm.Normal('obs', mu=model, sigma=noise_level, observed=flux)
    
    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=1, progressbar=(logger is not None),
                             return_inferencedata=True)
    
    if logger is not None:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])
    # stacked parameter chains
    const_ch = inf_data.posterior.const.stack(dim=['chain', 'draw']).to_numpy()
    slope_ch = inf_data.posterior.slope.stack(dim=['chain', 'draw']).to_numpy()
    f_n_ch = inf_data.posterior.f_n.stack(dim=['chain', 'draw']).to_numpy()
    a_n_ch = inf_data.posterior.a_n.stack(dim=['chain', 'draw']).to_numpy()
    ph_n_ch = inf_data.posterior.ph_n.stack(dim=['chain', 'draw']).to_numpy()
    # parameter means
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    par_means = [const_m, slope_m, f_n_m, a_n_m, ph_n_m]
    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    # convert interval to error bars
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [const_e, slope_e, f_n_e, a_n_e, ph_n_e]
    return inf_data, par_means, par_hdi


def sample_sinusoid_h(time, flux, p_orb, const, slope, f_n, a_n, ph_n, p_err, c_err, sl_err, f_n_err, a_n_err,
                      ph_n_err, noise_level, i_chunks, logger=None):
    """NUTS sampling of a linear + sinusoid + eclipse model
    
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
    p_err: float
        Uncertainty in the orbital period
    c_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the y-intercepts of a number of sine waves
    sl_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slopes of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequencies of a number of sine waves
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitudes of a number of sine waves
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phases of a number of sine waves
    noise_level: float
        The noise level (standard deviation of the residuals)
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    logger: logging.Logger, optional
        Instance of the logging library.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        inf_data: object
            Arviz inference data object
        par_means: list[float]
            Parameter mean values in the following order:
            p_orb, const, slope, f_n, a_n, ph_n
        par_hdi: list[float]
            Parameter HDI error values, same order as par_means
    """
    # setup
    time_t = time.reshape(-1, 1)  # transposed time
    t_mean = tt.as_tensor_variable(np.mean(time))
    t_mean_s = tt.as_tensor_variable(np.array([np.mean(time[s[0]:s[1]]) for s in i_chunks]))
    harmonics, harmonic_n = star_shine.core.frequency_sets.find_harmonics_from_pattern(f_n, 1/p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    lin_shape = (len(const),)
    sin_shape = (len(f_n[non_harm]),)
    harm_shape = (len(f_n[harmonics]),)
    # progress bar
    if logger is not None:
        fastprogress.printing = lambda: True
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.INFO)
    else:
        fastprogress.printing = lambda: False
        mc_logger = logging.getLogger('pymc3')
        mc_logger.setLevel(logging.ERROR)
    # make pymc3 model
    with pm.Model() as lc_model:
        # piece-wise linear curve parameter models
        const_pm = pm.Normal('const', mu=const, sigma=c_err, shape=lin_shape, testval=const)
        slope_pm = pm.Normal('slope', mu=slope, sigma=sl_err, shape=lin_shape, testval=slope)
        # piece-wise linear curve
        linear_curves = [const_pm[k] + slope_pm[k] * (time[s[0]:s[1]] - t_mean_s[k]) for k, s in enumerate(i_chunks)]
        model_linear = tt.concatenate(linear_curves)
        # sinusoid parameter models
        f_n_pm = pm.TruncatedNormal('f_n', mu=f_n[non_harm], sigma=f_n_err[non_harm], lower=0, shape=sin_shape,
                                    testval=f_n[non_harm])
        a_n_pm = pm.TruncatedNormal('a_n', mu=a_n[non_harm], sigma=a_n_err[non_harm], lower=0, shape=sin_shape,
                                    testval=a_n[non_harm])
        ph_n_pm = pm.VonMises('ph_n', mu=ph_n[non_harm], kappa=1 / ph_n_err[non_harm]**2, shape=sin_shape,
                              testval=ph_n[non_harm])
        # sum of sinusoids
        model_sinusoid = pm.math.sum(a_n_pm * pm.math.sin((2 * np.pi * f_n_pm * (time_t - t_mean)) + ph_n_pm), axis=1)
        # harmonic parameter models
        p_orb_pm = pm.TruncatedNormal('p_orb', mu=p_orb, sigma=p_err, lower=0, testval=p_orb)
        f_h_pm = pm.Deterministic('f_h', harmonic_n / p_orb_pm)
        a_h_pm = pm.TruncatedNormal('a_h', mu=a_n[harmonics], sigma=a_n_err[harmonics], lower=0, shape=harm_shape,
                                    testval=a_n[harmonics])
        ph_h_pm = pm.VonMises('ph_h', mu=ph_n[harmonics], kappa=1 / ph_n_err[harmonics]**2, shape=harm_shape,
                              testval=ph_n[harmonics])
        # sum of harmonic sinusoids
        model_harmonic = pm.math.sum(a_h_pm * pm.math.sin((2 * np.pi * f_h_pm * (time_t - t_mean)) + ph_h_pm), axis=1)
        # full light curve model
        model = model_linear + model_sinusoid + model_harmonic
        # observed distribution
        pm.Normal('obs', mu=model, sigma=noise_level, observed=flux)
    
    # do the sampling
    with lc_model:
        inf_data = pm.sample(draws=1000, tune=1000, init='adapt_diag', cores=1, progressbar=(logger is not None),
                             return_inferencedata=True)
    
    if logger is not None:
        az.summary(inf_data, round_to=2, circ_var_names=['ph_n'])
    # stacked parameter chains
    p_orb_ch = inf_data.posterior.p_orb.stack(dim=['chain', 'draw']).to_numpy()
    const_ch = inf_data.posterior.const.stack(dim=['chain', 'draw']).to_numpy()
    slope_ch = inf_data.posterior.slope.stack(dim=['chain', 'draw']).to_numpy()
    f_n_ch = inf_data.posterior.f_n.stack(dim=['chain', 'draw']).to_numpy()
    a_n_ch = inf_data.posterior.a_n.stack(dim=['chain', 'draw']).to_numpy()
    ph_n_ch = inf_data.posterior.ph_n.stack(dim=['chain', 'draw']).to_numpy()
    f_h_ch = inf_data.posterior.f_h.stack(dim=['chain', 'draw']).to_numpy()
    a_h_ch = inf_data.posterior.a_h.stack(dim=['chain', 'draw']).to_numpy()
    ph_h_ch = inf_data.posterior.ph_h.stack(dim=['chain', 'draw']).to_numpy()
    # parameter means
    p_orb_m = np.mean(p_orb_ch)
    const_m = np.mean(const_ch, axis=1).flatten()
    slope_m = np.mean(slope_ch, axis=1).flatten()
    f_n_m = np.mean(f_n_ch, axis=1).flatten()
    a_n_m = np.mean(a_n_ch, axis=1).flatten()
    ph_n_m = sp.stats.circmean(ph_n_ch, axis=1).flatten()
    f_h_m = harmonic_n / p_orb_m  # taking the mean from the chain leads to slightly different results and is wrong
    a_h_m = np.mean(a_h_ch, axis=1).flatten()
    ph_h_m = sp.stats.circmean(ph_h_ch, axis=1).flatten()
    f_n_m = np.append(f_n_m, f_h_m)
    a_n_m = np.append(a_n_m, a_h_m)
    ph_n_m = np.append(ph_n_m, ph_h_m)
    par_means = [p_orb_m, const_m, slope_m, f_n_m, a_n_m, ph_n_m]
    # parameter errors (from hdi) [hdi expects (chain, draw) as first two axes... annoying warnings...]
    p_orb_e = az.hdi(p_orb_ch, hdi_prob=0.683)
    const_e = az.hdi(np.moveaxis(const_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    slope_e = az.hdi(np.moveaxis(slope_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    f_n_e = az.hdi(np.moveaxis(f_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_n_e = az.hdi(np.moveaxis(a_n_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_n_e = az.hdi(np.moveaxis(ph_n_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_h_e = az.hdi(np.moveaxis(f_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    a_h_e = az.hdi(np.moveaxis(a_h_ch[np.newaxis], 1, 2), hdi_prob=0.683)
    ph_h_e = az.hdi(np.moveaxis(ph_h_ch[np.newaxis], 1, 2), hdi_prob=0.683, circular=True)
    f_n_e = np.append(f_n_e, f_h_e, axis=0)
    a_n_e = np.append(a_n_e, a_h_e, axis=0)
    ph_n_e = np.append(ph_n_e, ph_h_e, axis=0)
    # convert interval to error bars
    p_orb_e = np.array([p_orb_m - p_orb_e[0], p_orb_e[1] - p_orb_m])
    const_e = np.column_stack([const_m - const_e[:, 0], const_e[:, 1] - const_m])
    slope_e = np.column_stack([slope_m - slope_e[:, 0], slope_e[:, 1] - slope_m])
    f_n_e = np.column_stack([f_n_m - f_n_e[:, 0], f_n_e[:, 1] - f_n_m])
    a_n_e = np.column_stack([a_n_m - a_n_e[:, 0], a_n_e[:, 1] - a_n_m])
    ph_n_e = np.column_stack([ph_n_m - ph_n_e[:, 0], ph_n_e[:, 1] - ph_n_m])
    par_hdi = [p_orb_e, const_e, slope_e, f_n_e, a_n_e, ph_n_e]
    return inf_data, par_means, par_hdi
