"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for visualisation;
specifically for visualising the analysis of stellar variability and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import arviz as az  # optional functionality
except ImportError:
    az = None
    pass

from star_shine.core import time_series as tms, model as mdl, frequency_sets as frs
from star_shine.core import periodogram as pdg, utility as ut, io
from star_shine.config.helpers import get_mpl_stylesheet_path


# mpl style sheet
plt.style.use(get_mpl_stylesheet_path())


def plot_pd(time, flux, i_chunks, plot_per_chunk=False, file_name=None, show=True):
    """Plot the periodogram of the data.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    plot_per_chunk: bool
        If True, plot the periodogram of all time chunks in one plot.
    file_name: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # make the periodograms
    freqs, ampls = pdg.scargle_parallel(time, flux)
    freqs_list, ampls_list = [], []
    if plot_per_chunk:
        for ch in i_chunks:
            f, a = pdg.scargle_parallel(time[ch[0]:ch[1]], flux[ch[0]:ch[1]])
            freqs_list.append(f)
            ampls_list.append(a)

    # plot ranges
    freq_range = np.ptp(freqs)

    # plot
    fig, ax = plt.subplots()

    if plot_per_chunk:
        for i in range(len(i_chunks)):
            ax.plot(freqs_list[i], ampls_list[i])
    else:
        ax.plot(freqs, ampls, c='tab:blue', label='flux')

    ax.set_xlim(freqs[0] - freq_range * 0.05, freqs[-1] + freq_range * 0.05)
    plt.xlabel('frequency (1/d)')
    plt.ylabel('amplitude')
    plt.legend()
    plt.tight_layout()

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return None

def plot_pd_single_output(time, flux, flux_err, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_chunks,
                          annotate=True, file_name=None, show=True):
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
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    annotate: bool, optional
        If True, annotate the plot with the frequencies.
    file_name: str, optional
        File path to save the plot
    show: bool, optional
        If True, display the plot

    Returns
    -------
    None
    """
    # separate harmonics
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, 1/p_orb, f_tol=1e-9)

    # make model
    model_linear = mdl.linear_curve(time, const, slope, i_chunks)
    model_sinusoid = mdl.sum_sines(time, f_n, a_n, ph_n)
    model = model_linear + model_sinusoid

    # make periodograms
    freqs, ampls = pdg.scargle_parallel(time, flux)
    freq_range = np.ptp(freqs)
    freqs_r, ampls_r = pdg.scargle_parallel(time, flux - model)

    # get error values
    errors = ut.formal_uncertainties(time, flux - model, flux_err, a_n, i_chunks)

    # max plot value
    y_max = max(np.max(ampls), np.max(a_n))

    # plot
    fig, ax = plt.subplots()
    if len(harmonics) > 0:
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

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return None


def plot_pd_full_output(time, flux, flux_err, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_chunks, file_name=None,
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
    # make periodograms
    freqs, ampls = pdg.scargle_parallel(time, flux - np.mean(flux))
    freq_range = np.ptp(freqs)
    freqs_1, ampls_1 = pdg.scargle_parallel(time, flux - models[0] - np.all(models[0] == 0) * np.mean(flux))
    freqs_2, ampls_2 = pdg.scargle_parallel(time, flux - models[1] - np.all(models[1] == 0) * np.mean(flux))
    freqs_3, ampls_3 = pdg.scargle_parallel(time, flux - models[2] - np.all(models[2] == 0) * np.mean(flux))
    freqs_4, ampls_4 = pdg.scargle_parallel(time, flux - models[3] - np.all(models[3] == 0) * np.mean(flux))
    freqs_5, ampls_5 = pdg.scargle_parallel(time, flux - models[4] - np.all(models[4] == 0) * np.mean(flux))

    # get error values
    err_1 = ut.formal_uncertainties(time, flux - models[0], flux_err, a_n_i[0], i_chunks)
    err_2 = ut.formal_uncertainties(time, flux - models[1], flux_err, a_n_i[1], i_chunks)
    err_3 = ut.formal_uncertainties(time, flux - models[2], flux_err, a_n_i[2], i_chunks)
    err_4 = ut.formal_uncertainties(time, flux - models[3], flux_err, a_n_i[3], i_chunks)
    err_5 = ut.formal_uncertainties(time, flux - models[4], flux_err, a_n_i[4], i_chunks)

    # max plot value
    if len(f_n_i[4]) > 0:
        y_max = max(np.max(ampls), np.max(a_n_i[4]))
    else:
        y_max = np.max(ampls)

    # plot
    fig, ax = plt.subplots()
    ax.plot(freqs, ampls, label='flux')
    if len(f_n_i[0]) > 0:
        ax.plot(freqs_1, ampls_1, label='extraction residual')
    if len(f_n_i[1]) > 0:
        ax.plot(freqs_2, ampls_2, label='NL-LS optimisation residual')
    if len(f_n_i[2]) > 0:
        ax.plot(freqs_3, ampls_3, label='coupled harmonics residual')
    if len(f_n_i[3]) > 0:
        ax.plot(freqs_4, ampls_4, label='additional frequencies residual')
    if len(f_n_i[4]) > 0:
        ax.plot(freqs_5, ampls_5, label='NL-LS fit residual with harmonics residual')

    # period
    if p_orb_i[4] > 0:
        ax.errorbar([1 / p_orb_i[4], 1 / p_orb_i[4]], [0, y_max], xerr=[0, p_err_i[4] / p_orb_i[4]**2],
                    linestyle='--', capsize=2, c='k', label=f'orbital frequency (p={p_orb_i[4]:1.4f}d)')
    elif p_orb_i[2] > 0:
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

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return None


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

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return None


def plot_lc_sinusoids(time, flux, const, slope, f_n, a_n, ph_n, i_chunks, file_name=None, show=True):
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
    t_mean = np.mean(time)

    # make models
    model_linear = mdl.linear_curve(time, const, slope, i_chunks)
    model_sines = mdl.sum_sines(time, f_n, a_n, ph_n)
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

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return None


def plot_lc_harmonics(time, flux, p_orb, p_err, const, slope, f_n, a_n, ph_n, i_chunks, file_name=None,
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
    t_mean = np.mean(time)

    # make models
    model_line = mdl.linear_curve(time, const, slope, i_chunks)
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, 1/p_orb, f_tol=1e-9)
    model_h = mdl.sum_sines(time, f_n[harmonics], a_n[harmonics], ph_n[harmonics])
    model_nh = mdl.sum_sines(time, np.delete(f_n, harmonics), np.delete(a_n, harmonics),
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

    # save/show/close
    if file_name is not None:
        plt.savefig(file_name, dpi=120, format='png')  # 16 by 9 at 120 dpi is 1080p
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_trace_sinusoids(inf_data, const, slope, f_n, a_n, ph_n):
    """Show the pymc3 sampling results in a trace plot

    Parameters
    ----------
    inf_data: InferenceData object
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
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, 1/p_orb, f_tol=1e-9)
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
    inf_data: InferenceData object
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
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, 1/p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(len(f_n)), harmonics)
    par_lines = [('p_orb', {}, p_orb), ('const', {}, const), ('slope', {}, slope),
                 ('f_n', {}, f_n[non_harm]), ('a_n', {}, a_n[non_harm]), ('ph_n', {}, ph_n[non_harm]),
                 ('f_h', {}, f_n[harmonics]), ('a_h', {}, a_n[harmonics]), ('ph_h', {}, ph_n[harmonics])]
    az.plot_trace(inf_data, combined=False, compact=True, rug=True, divergences='top', lines=par_lines)

    return


def sequential_plotting(time, flux, flux_err, i_chunks, target_id, load_dir, save_dir=None, show=True):
    """Due to plotting not working under multiprocessing this function is
    made to make plots after running the analysis in parallel.

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
    target_id: int, str
        In case of using analyse_from_tic:
        The TESS Input Catalog number
        In case of user-defined light curve files (analyse_from_file):
        Should be the same as the name of the light curve file.
    load_dir: str
        Path to a directory for loading analysis results.
        Will append <target_id> + _analysis automatically
    save_dir: str, None
        Path to a directory for saving the plots.
        Will append <target_id> + _analysis automatically
        Directory is created if it doesn't exist yet
    show: bool
        Whether to show the plots or not.

    Returns
    -------
    None
    """
    load_dir = os.path.join(load_dir, f'{target_id}_analysis')  # add subdir
    if save_dir is not None:
        save_dir = os.path.join(save_dir, f'{target_id}_analysis')  # add subdir

        # for saving, make a folder if not there yet
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)  # create the subdir

    # open all the data
    file_name = os.path.join(load_dir, f'{target_id}_analysis_1.hdf5')
    if os.path.isfile(file_name):
        results = io.load_result_hdf5(file_name)
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = results['sin_mean']
        model_linear = mdl.linear_curve(time, const_1, slope_1, i_chunks)
        model_sinusoid = mdl.sum_sines(time, f_n_1, a_n_1, ph_n_1)
        model_1 = model_linear + model_sinusoid
    else:
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = np.array([[], [], [], [], []])
        model_1 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_2.hdf5')
    if os.path.isfile(file_name):
        results = io.load_result_hdf5(file_name)
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = results['sin_mean']
        model_linear = mdl.linear_curve(time, const_2, slope_2, i_chunks)
        model_sinusoid = mdl.sum_sines(time, f_n_2, a_n_2, ph_n_2)
        model_2 = model_linear + model_sinusoid
    else:
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = np.array([[], [], [], [], []])
        model_2 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_3.hdf5')
    if os.path.isfile(file_name):
        results = io.load_result_hdf5(file_name)
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = results['sin_mean']
        p_orb_3, _ = results['ephem']
        p_err_3, _ = results['ephem_err']
        model_linear = mdl.linear_curve(time, const_3, slope_3, i_chunks)
        model_sinusoid = mdl.sum_sines(time, f_n_3, a_n_3, ph_n_3)
        model_3 = model_linear + model_sinusoid
    else:
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = np.array([[], [], [], [], []])
        p_orb_3, p_err_3 = 0, 0
        model_3 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_4.hdf5')
    if os.path.isfile(file_name):
        results = io.load_result_hdf5(file_name)
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = results['sin_mean']
        model_linear = mdl.linear_curve(time, const_4, slope_4, i_chunks)
        model_sinusoid = mdl.sum_sines(time, f_n_4, a_n_4, ph_n_4)
        model_4 = model_linear + model_sinusoid
    else:
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = np.array([[], [], [], [], []])
        model_4 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name):
        results = io.load_result_hdf5(file_name)
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = results['sin_mean']
        p_orb_5, _ = results['ephem']
        p_err_5, _ = results['ephem_err']
        t_tot, t_mean, t_mean_s, t_int, n_param_5, bic_5, noise_level_5 = results['stats']
        model_linear = mdl.linear_curve(time, const_5, slope_5, i_chunks)
        model_sinusoid = mdl.sum_sines(time, f_n_5, a_n_5, ph_n_5)
        model_5 = model_linear + model_sinusoid
        harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n_5, 1/p_orb_5, f_tol=1e-9)
        f_h_5, a_h_5, ph_h_5 = f_n_5[harmonics], a_n_5[harmonics], ph_n_5[harmonics]
    else:
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = np.array([[], [], [], [], []])
        p_orb_5, p_err_5 = 0, 0
        n_param_5, bic_5, noise_level_5 = 0, 0, 0
        model_5 = np.zeros(len(time))
        f_h_5, a_h_5, ph_h_5 = np.array([[], [], []])

    # stick together for sending to plot function
    models = [model_1, model_2, model_3, model_4, model_5]
    p_orb_i = [0, 0, p_orb_3, p_orb_3, p_orb_5]
    p_err_i = [0, 0, p_err_3, p_err_3, p_err_5]
    f_n_i = [f_n_1, f_n_2, f_n_3, f_n_4, f_n_5]
    a_n_i = [a_n_1, a_n_2, a_n_3, a_n_4, a_n_5]

    # plot frequency_analysis
    try:
        if save_dir is not None:
            file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_full.png')
        else:
            file_name = None
        plot_pd_full_output(time, flux, flux_err, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_chunks,
                                file_name=file_name, show=show)
        if np.any([len(fs) != 0 for fs in f_n_i]):
            plot_nr = np.arange(1, len(f_n_i) + 1)[[len(fs) != 0 for fs in f_n_i]][-1]
            plot_data = [eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_sinusoids_{plot_nr}.png')
            else:
                file_name = None
            plot_lc_sinusoids(time, flux, *plot_data, i_chunks, file_name=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_output_{plot_nr}.png')
            else:
                file_name = None
            plot_data = [p_orb_i[plot_nr - 1], p_err_i[plot_nr - 1]] + plot_data
            plot_pd_single_output(time, flux, flux_err, *plot_data, i_chunks, annotate=False,
                                      file_name=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_harmonics_{plot_nr}.png')
            else:
                file_name = None
            plot_lc_harmonics(time, flux, *plot_data, i_chunks, file_name=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    except ValueError:
        pass  # no frequencies?

    return None

def plot_all_from_file(file_name, i_chunks=None, load_dir=None, save_dir=None, show=True):
    """Plot all diagnostic plots of the results for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    load_dir: str
        Path to a directory for loading analysis results.
        Will append <target_id> + _analysis automatically.
        Assumes the same directory as file_name if None.
    save_dir: str, None
        Path to a directory for saving the plots.
        Will append <target_id> + _analysis automatically.
        Directory is created if it doesn't exist yet.
    show: bool
        Whether to show the plots or not.

    Returns
    -------
    None
    """
    target_id = os.path.splitext(os.path.basename(file_name))[0]  # file name is used as target identifier
    if load_dir is None:
        load_dir = os.path.dirname(file_name)

    # load the data
    time, flux, flux_err = np.loadtxt(file_name, usecols=(0, 1, 2), unpack=True)

    # if sectors not given, take full length
    if i_chunks is None:
        i_chunks = np.array([[0, len(time)]])  # no sector information
    # i_half_s = i_chunks  # in this case no differentiation between half or full sectors

    # do the plotting
    sequential_plotting(time, flux, flux_err, i_chunks, target_id, load_dir, save_dir=save_dir, show=show)

    return None

def plot_all_from_tic(tic, all_files, load_dir=None, save_dir=None, show=True):
    """Plot all diagnostic plots of the results for a given light curve file

    Parameters
    ----------
    tic: int
        The TESS Input Catalog (TIC) number for loading/saving the data
        and later reference.
    all_files: list[str]
        List of all the TESS data product '.fits' files. The files
        with the corresponding TIC number are selected.
    load_dir: str
        Path to a directory for loading analysis results.
        Will append <tic> + _analysis automatically.
        Assumes the same directory as all_files if None.
    save_dir: str, None
        Path to a directory for saving the plots.
        Will append <tic> + _analysis automatically.
        Directory is created if it doesn't exist yet.
    show: bool
        Whether to show the plots or not.

    Returns
    -------
    None
    """
    if load_dir is None:
        load_dir = os.path.dirname(all_files[0])

    # load the data
    time, flux, flux_err, i_chunks, medians = io.load_light_curve(all_files, apply_flags=True)

    # do the plotting
    sequential_plotting(time, flux, flux_err, i_chunks, tic, load_dir, save_dir=save_dir, show=show)

    return None
