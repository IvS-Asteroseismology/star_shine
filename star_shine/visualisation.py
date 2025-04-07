"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for visualisation;
specifically for visualising the analysis of stellar variability and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import os

import numpy as np
import matplotlib.pyplot as plt
from astropy.constants.codata2018 import alpha

try:
    import arviz as az  # optional functionality
except ImportError:
    pass

from . import timeseries_functions as tsf
from . import analysis_functions as af

# mpl style sheet
script_dir = os.path.dirname(os.path.abspath(__file__))  # absolute dir the script is in
plt.style.use(os.path.join(script_dir, 'data', 'mpl_stylesheet.dat'))


def plot_pd_single_output(time, flux, flux_err, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_chunks,
                          annotate=True, save_file=None, show=True):
    """Plot the periodogram with one output of the analysis recipe.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    p_orb: float
        Orbital period
    p_err: float
        Error associated with the orbital period
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
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(time)]]).
    annotate: bool, optional
        If True, annotate the plot with the frequencies.
    save_file: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # separate harmonics
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    # make model
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = tsf.sum_sines(time, f_n, a_n, ph_n)
    model = model_linear + model_sinusoid
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(time, flux)
    freq_range = np.ptp(freqs)
    freqs_r, ampls_r = tsf.astropy_scargle(time, flux - model)
    # get error values
    errors = tsf.formal_uncertainties(time, flux - model, flux_err, a_n, i_chunks)
    # max plot value
    y_max = max(np.max(ampls), np.max(a_n))
    # plot
    fig, ax = plt.subplots()
    if (len(harmonics) > 0):
        ax.errorbar([1 / p_orb, 1 / p_orb], [0, y_max], xerr=[0, p_err / p_orb**2],
                    linestyle='-', capsize=2, c='tab:grey', label=f'orbital frequency (p={p_orb:1.4f}d +-{p_err:1.4f})')
        for i in range(2, np.max(harmonic_n) + 1):
            ax.plot([i / p_orb, i / p_orb], [0, y_max], linestyle='-', c='tab:grey', alpha=0.3)
        ax.errorbar([], [], xerr=[], yerr=[], linestyle='-', capsize=2, c='tab:red', label='extracted harmonics')
    ax.plot(freqs, ampls, c='tab:blue', label='flux')
    ax.plot(freqs_r, ampls_r, c='tab:orange', label='residual')
    for i in range(len(f_n)):
        if i in harmonics:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:red')
        else:
            ax.errorbar([f_n[i], f_n[i]], [0, a_n[i]], xerr=[0, errors[2][i]], yerr=[0, errors[3][i]],
                        linestyle='-', capsize=2, c='tab:pink')
        if annotate:
            ax.annotate(f'{i + 1}', (f_n[i], a_n[i]))
    ax.errorbar([], [], xerr=[], yerr=[], linestyle='-', capsize=2, c='tab:pink', label='extracted frequencies')
    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_pd_full_output(time, flux, flux_err, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_chunks, save_file=None,
                        show=True):
    """Plot the periodogram with the full output of the analysis recipe.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    models: list[numpy.ndarray[Any, dtype[float]]]
        List of model fluxs for different stages of the analysis
    p_orb_i: list[float]
        Orbital periods for different stages of the analysis
    p_err_i: list[float]
        Errors associated with the orbital periods
        for different stages of the analysis
    f_n_i: list[numpy.ndarray[Any, dtype[float]]]
        List of extracted frequencies for different stages of the analysis
    a_n_i: list[numpy.ndarray[Any, dtype[float]]]
        List of amplitudes corresponding to the extracted frequencies
        for different stages of the analysis
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(time)]]).
    save_file: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # make periodograms
    freqs, ampls = tsf.astropy_scargle(time, flux - np.mean(flux))
    freq_range = np.ptp(freqs)
    freqs_1, ampls_1 = tsf.astropy_scargle(time, flux - models[0] - np.all(models[0] == 0) * np.mean(flux))
    freqs_2, ampls_2 = tsf.astropy_scargle(time, flux - models[1] - np.all(models[1] == 0) * np.mean(flux))
    freqs_3, ampls_3 = tsf.astropy_scargle(time, flux - models[2] - np.all(models[2] == 0) * np.mean(flux))
    freqs_4, ampls_4 = tsf.astropy_scargle(time, flux - models[3] - np.all(models[3] == 0) * np.mean(flux))
    freqs_5, ampls_5 = tsf.astropy_scargle(time, flux - models[4] - np.all(models[4] == 0) * np.mean(flux))
    # get error values
    err_1 = tsf.formal_uncertainties(time, flux - models[0], flux_err, a_n_i[0], i_chunks)
    err_2 = tsf.formal_uncertainties(time, flux - models[1], flux_err, a_n_i[1], i_chunks)
    err_3 = tsf.formal_uncertainties(time, flux - models[2], flux_err, a_n_i[2], i_chunks)
    err_4 = tsf.formal_uncertainties(time, flux - models[3], flux_err, a_n_i[3], i_chunks)
    err_5 = tsf.formal_uncertainties(time, flux - models[4], flux_err, a_n_i[4], i_chunks)
    # max plot value
    if (len(f_n_i[4]) > 0):
        y_max = max(np.max(ampls), np.max(a_n_i[4]))
    else:
        y_max = np.max(ampls)
    # plot
    fig, ax = plt.subplots()
    ax.plot(freqs, ampls, label='flux')
    if (len(f_n_i[0]) > 0):
        ax.plot(freqs_1, ampls_1, label='extraction residual')
    if (len(f_n_i[1]) > 0):
        ax.plot(freqs_2, ampls_2, label='NL-LS optimisation residual')
    if (len(f_n_i[2]) > 0):
        ax.plot(freqs_3, ampls_3, label='coupled harmonics residual')
    if (len(f_n_i[3]) > 0):
        ax.plot(freqs_4, ampls_4, label='additional frequencies residual')
    if (len(f_n_i[4]) > 0):
        ax.plot(freqs_5, ampls_5, label='NL-LS fit residual with harmonics residual')
    # period
    if (p_orb_i[4] > 0):
        ax.errorbar([1 / p_orb_i[4], 1 / p_orb_i[4]], [0, y_max], xerr=[0, p_err_i[4] / p_orb_i[4]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[4]:1.4f}d)')
    elif (p_orb_i[2] > 0):
        ax.errorbar([1 / p_orb_i[2], 1 / p_orb_i[2]], [0, y_max], xerr=[0, p_err_i[2] / p_orb_i[2]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[2]:1.4f}d)')
    # frequencies
    for i in range(len(f_n_i[0])):
        ax.errorbar([f_n_i[0][i], f_n_i[0][i]], [0, a_n_i[0][i]], xerr=[0, err_1[2][i]], yerr=[0, err_1[3][i]],
                    linestyle=':', capsize=2, c='tab:orange')
    for i in range(len(f_n_i[1])):
        ax.errorbar([f_n_i[1][i], f_n_i[1][i]], [0, a_n_i[1][i]], xerr=[0, err_2[2][i]], yerr=[0, err_2[3][i]],
                    linestyle=':', capsize=2, c='tab:green')
    for i in range(len(f_n_i[2])):
        ax.errorbar([f_n_i[2][i], f_n_i[2][i]], [0, a_n_i[2][i]], xerr=[0, err_3[2][i]], yerr=[0, err_3[3][i]],
                    linestyle=':', capsize=2, c='tab:red')
    for i in range(len(f_n_i[3])):
        ax.errorbar([f_n_i[3][i], f_n_i[3][i]], [0, a_n_i[3][i]], xerr=[0, err_4[2][i]], yerr=[0, err_4[3][i]],
                    linestyle=':', capsize=2, c='tab:purple')
    for i in range(len(f_n_i[4])):
        ax.errorbar([f_n_i[4][i], f_n_i[4][i]], [0, a_n_i[4][i]], xerr=[0, err_5[2][i]], yerr=[0, err_5[3][i]],
                    linestyle=':', capsize=2, c='tab:brown')
        ax.annotate(f'{i + 1}', (f_n_i[4][i], a_n_i[4][i]))
    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc(time, flux, flux_err, i_chunks, file_name=None, show=True):
    """Shows the light curve data

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    file_name: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # plot the light curve data with different colours for each chunk
    fig, ax = plt.subplots(figsize=(16, 9))
    for ch in i_chunks:
        t_mean = np.mean(time[ch[0], ch[1]])
        f_min = np.min(flux[ch[0], ch[1]])
        f_max = np.max(flux[ch[0], ch[1]])
        ax.plot([t_mean, t_mean], [f_min, f_max], alpha=0.3)
        ax.errorbar(time[ch[0], ch[1]], flux[ch[0], ch[1]], yerr=flux_err[ch[0], ch[1]], color='grey', alpha=0.3)
        ax.scatter(time[ch[0], ch[1]], flux[ch[0], ch[1]], marker='.', label='dataset')
    ax.set_xlabel('time')
    ax.set_ylabel('flux')
    ax.legend()
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return None


def plot_lc_sinusoids(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, save_file=None, show=True):
    """Shows the separated harmonics in several ways

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
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(time)]]).
    save_file: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    t_mean = np.mean(time)
    # make models
    model_linear = tsf.linear_curve(time, const, slope, i_chunks)
    model_sines = tsf.sum_sines(time, f_n, a_n, ph_n)
    resid = flux - (model_linear + model_sines)
    # plot the full model light curve
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot([t_mean, t_mean], [np.min(flux), np.max(flux)], c='grey', alpha=0.3)
    ax[0].scatter(time, flux, marker='.', label='flux')
    ax[0].plot(time, model_linear + model_sines, c='tab:orange', label='full model (linear + sinusoidal)')
    ax[1].plot([t_mean, t_mean], [np.min(resid), np.max(resid)], c='grey', alpha=0.3)
    ax[1].scatter(time, resid, marker='.')
    ax[0].set_ylabel('flux/model')
    ax[0].legend()
    ax[1].set_ylabel('residual')
    ax[1].set_xlabel('time (d)')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_lc_harmonics(time, flux, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_chunks, save_file=None,
                      show=True):
    """Shows the separated harmonics in several ways

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    p_orb: float
        Orbital period of the system
    p_err: float
        Error in the orbital period
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
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(time)]]).
    save_file: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    t_mean = np.mean(time)
    # make models
    model_line = tsf.linear_curve(time, const, slope, i_chunks)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    model_h = tsf.sum_sines(time, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = tsf.sum_sines(time, np.delete(f_n, harmonics), np.delete(a_n, harmonics),
                             np.delete(ph_n, harmonics))
    resid_nh = flux - model_nh
    resid_h = flux - model_h
    # plot the harmonic model and non-harmonic model
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot([t_mean, t_mean], [np.min(resid_nh), np.max(resid_nh)], c='grey', alpha=0.3)
    ax[0].scatter(time, resid_nh, marker='.', c='tab:blue', label='flux - non-harmonics')
    ax[0].plot(time, model_line + model_h, c='tab:orange', label='linear + harmonic model, '
                                                                  f'p={p_orb:1.4f}d (+-{p_err:1.4f})')
    ax[1].plot([t_mean, t_mean], [np.min(resid_h), np.max(resid_h)], c='grey', alpha=0.3)
    ax[1].scatter(time, resid_h, marker='.', c='tab:blue', label='flux - harmonics')
    ax[1].plot(time, model_line + model_nh, c='tab:orange', label='linear + non-harmonic model')
    ax[0].set_ylabel('residual/model')
    ax[0].legend()
    ax[1].set_ylabel('residual/model')
    ax[1].set_xlabel('time (d)')
    ax[1].legend()
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    if save_file is not None:
        plt.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()
    return


def plot_trace_sinusoids(inf_data, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot

    Parameters
    ----------
    inf_data: object
        Arviz inference data object
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

    Returns
    -------
    None
    """
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    par_lines = [('const', {}, const), ('slope', {}, slope), ('f_n', {}, f_n), ('a_n', {}, a_n), ('ph_n', {}, ph_n)]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return


def plot_pair_harmonics(inf_data, p_orb, const, slope, f_n, a_n, ph_n, save_file=None, show=True):
    """Show the pymc3 sampling results in several pair plots

    Parameters
    ----------
    inf_data: object
        Arviz inference data object
    p_orb: float
        Orbital period
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
    save_file: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    ref_values = {'p_orb': p_orb, 'const': const, 'slope': slope,
                  'f_n': f_n[non_harm], 'a_n': a_n[non_harm], 'ph_n': ph_n[non_harm],
                  'f_h': f_n[harmonics], 'a_h': a_n[harmonics], 'ph_h': ph_n[harmonics]}
    kwargs = {'marginals': True, 'textsize': 14, 'kind': ['scatter', 'kde'],
              'marginal_kwargs': {'quantiles': [0.158, 0.5, 0.842]}, 'point_estimate': 'mean',
              'reference_values': ref_values, 'show': show}
    az.plot_pair(inf_data, var_names=['f_n', 'a_n', 'ph_n'],
                 coords={'f_n_dim_0': [0, 1, 2], 'a_n_dim_0': [0, 1, 2], 'ph_n_dim_0': [0, 1, 2]}, **kwargs)
    az.plot_pair(inf_data, var_names=['p_orb', 'f_n'], coords={'f_n_dim_0': np.arange(9)}, **kwargs)
    ax = az.plot_pair(inf_data, var_names=['p_orb', 'const', 'slope', 'f_n', 'a_n', 'ph_n', 'a_h', 'ph_h'],
                      coords={'const_dim_0': [0], 'slope_dim_0': [0], 'f_n_dim_0': [0], 'a_n_dim_0': [0],
                              'ph_n_dim_0': [0], 'a_h_dim_0': [0], 'ph_h_dim_0': [0]}, **kwargs)
    # save if wanted (only last plot - most interesting one)
    if save_file is not None:
        fig = ax.ravel()[0].figure
        fig.savefig(save_file, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    return


def plot_trace_harmonics(inf_data, p_orb, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot

    Parameters
    ----------
    inf_data: object
        Arviz inference data object
    p_orb: float
        Orbital period
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

    Returns
    -------
    None
    """
    # convert phases to interval [-pi, pi] from [0, 2pi]
    above_pi = (ph_n >= np.pi)
    ph_n[above_pi] = ph_n[above_pi] - 2 * np.pi
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    par_lines = [('p_orb', {}, p_orb), ('const', {}, const), ('slope', {}, slope),
                 ('f_n', {}, f_n[non_harm]), ('a_n', {}, a_n[non_harm]), ('ph_n', {}, ph_n[non_harm]),
                 ('f_h', {}, f_n[harmonics]), ('a_h', {}, a_n[harmonics]), ('ph_h', {}, ph_n[harmonics])]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)
    return
