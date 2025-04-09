"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for time series analysis;
specifically for the fitting of stellar oscillations and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import inspect
import numpy as np
import scipy as sp
import numba as nb
import astropy.timeseries as apy

from . import analysis_functions as af
from . import utility as ut


@nb.njit(cache=True)
def fold_time_series_phase(time, p_orb, zero=None):
    """Fold the given time series over the orbital period to transform to phase space.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    zero: float, None
        Reference zero point in time when the phase equals zero

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Phase array for all timestamps. Phases are between -0.5 and 0.5
    """
    mean_t = np.mean(time)
    if zero is None:
        zero = -mean_t
    phases = ((time - mean_t - zero) / p_orb + 0.5) % 1 - 0.5
    return phases


@nb.njit(cache=True)
def fold_time_series(time, p_orb, t_zero, t_ext_1=0, t_ext_2=0):
    """Fold the given time series over the orbital period
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    p_orb: float
        The orbital period with which the time series is folded
    t_zero: float, None
        Reference zero point in time (with respect to the time series mean time)
        when the phase equals zero
    t_ext_1: float
        Negative time interval to extend the folded time series to the left.
    t_ext_2: float
        Positive time interval to extend the folded time series to the right.
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        t_extended: numpy.ndarray[Any, dtype[float]]
            Folded time series array for all timestamps (and possible extensions).
        ext_left: numpy.ndarray[bool]
            Mask of points to extend time series to the left (for if t_ext_1!=0)
        ext_right: numpy.ndarray[bool]
            Mask of points to extend time series to the right (for if t_ext_2!=0)
    """
    # reference time is the mean of the time array
    mean_t = np.mean(time)
    t_folded = (time - mean_t - t_zero) % p_orb
    # extend to both sides
    ext_left = (t_folded > p_orb + t_ext_1)
    ext_right = (t_folded < t_ext_2)
    t_extended = np.concatenate((t_folded[ext_left] - p_orb, t_folded, t_folded[ext_right] + p_orb))
    return t_extended, ext_left, ext_right


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
def phase_dispersion(phases, flux, n_bins):
    """Phase dispersion, as in PDM, without overlapping bins.
    
    Parameters
    ----------
    phases: numpy.ndarray[Any, dtype[float]]
        The phase-folded timestamps of the time series, between -0.5 and 0.5.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    n_bins: int
        The number of bins over the orbital phase
    
    Returns
    -------
    float
        Phase dispersion, or summed variance over the bins divided by
        the variance of the flux
    
    Notes
    -----
    Intentionally does not make use of scipy to enable JIT-ting, which makes this considerably faster.
    """
    def var_no_avg(a):
        return np.sum(np.abs(a - np.mean(a))**2)  # if mean instead of sum, this is variance
    
    edges = np.linspace(-0.5, 0.5, n_bins + 1)
    # binned, edges, indices = sp.stats.binned_statistic(phases, flux, statistic=statistic, bins=bins)
    binned_var = np.zeros(n_bins)
    for i, (b1, b2) in enumerate(zip(edges[:-1], edges[1:])):
        bin_mask = (phases >= b1) & (phases < b2)
        if np.any(bin_mask):
            binned_var[i] = var_no_avg(flux[bin_mask])
        else:
            binned_var[i] = 0
    total_var = np.sum(binned_var) / len(flux)
    overall_var = np.var(flux)
    return total_var / overall_var


@nb.njit(cache=True)
def phase_dispersion_minimisation(time, flux, f_n, local=False):
    """Determine the phase dispersion over a set of periods to find the minimum
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    local: bool
        If set True, only searches the given frequencies,
        else also fractions of the frequencies are searched
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        periods: numpy.ndarray[Any, dtype[float]]
            Periods at which the phase dispersion is calculated
        pd_all: numpy.ndarray[Any, dtype[float]]
            Phase dispersion at the given periods
    """
    # number of bins for dispersion calculation
    n_points = len(time)
    if n_points / 10 > 1000:
        n_bins = 1000
    else:
        n_bins = n_points // 10  # at least 10 data points per bin on average
    # determine where to look based on the frequencies, including fractions of the frequencies
    if local:
        periods = 1 / f_n
    else:
        periods = np.zeros(7 * len(f_n))
        for i, f in enumerate(f_n):
            periods[7*i:7*i+7] = np.arange(1, 8) / f
    # stay below the maximum
    periods = periods[periods < np.ptp(time)]
    # and above the minimum
    periods = periods[periods > (2 * np.min(time[1:] - time[:-1]))]
    # compute the dispersion measures
    pd_all = np.zeros(len(periods))
    for i, p in enumerate(periods):
        fold = fold_time_series_phase(time, p, 0)
        pd_all[i] = phase_dispersion(fold, flux, n_bins)
    return periods, pd_all


def scargle_noise_spectrum(time, resid, window_width=1.0):
    """Calculate the Lomb-Scargle noise spectrum by a convolution with a flat window of a certain width.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    resid: numpy.ndarray[Any, dtype[float]]
        Residual measurement values of the time series
    window_width: float
        The width of the window used to compute the noise spectrum,
        in inverse unit of the time array (i.e. 1/d if time is in d).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The noise spectrum calculated as the mean in a frequency window
        in the residual periodogram
    
    Notes
    -----
    The values calculated here capture the amount of noise on fitting a
    sinusoid of a certain frequency to all data points.
    Not to be confused with the noise on the individual data points of the
    time series.
    """
    # calculate the periodogram
    freqs, ampls = astropy_scargle(time, resid)  # use defaults to get full amplitude spectrum
    # determine the number of points to extend the spectrum with for convolution
    n_points = int(np.ceil(window_width / np.abs(freqs[1] - freqs[0])))  # .astype(int)
    window = np.full(n_points, 1 / n_points)
    # extend the array with mirrors for convolution
    ext_ampls = np.concatenate((ampls[(n_points - 1)::-1], ampls, ampls[:-(n_points + 1):-1]))
    ext_noise = np.convolve(ext_ampls, window, 'same')
    # cut back to original interval
    noise = ext_noise[n_points:-n_points]
    # extra correction to account for convolve mode='full' instead of 'same' (needed for JIT-ting)
    # noise = noise[n_points//2 - 1:-n_points//2]
    return noise


def scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0):
    """Calculate the Lomb-Scargle noise spectrum by a convolution with a flat window of a certain width,
    given an amplitude spectrum.

    Parameters
    ----------
    freqs: numpy.ndarray[Any, dtype[float]]
        Frequencies at which the periodogram was calculated
    ampls: numpy.ndarray[Any, dtype[float]]
        The periodogram spectrum in the chosen units
    window_width: float
        The width of the window used to compute the noise spectrum,
        in inverse unit of the time array (i.e. 1/d if time is in d).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The noise spectrum calculated as the mean in a frequency window
        in the residual periodogram

    Notes
    -----
    The values calculated here capture the amount of noise on fitting a
    sinusoid of a certain frequency to all data points.
    Not to be confused with the noise on the individual data points of the
    time series.
    """
    # determine the number of points to extend the spectrum with for convolution
    n_points = int(np.ceil(window_width / np.abs(freqs[1] - freqs[0])))  # .astype(int)
    window = np.full(n_points, 1 / n_points)
    # extend the array with mirrors for convolution
    ext_ampls = np.concatenate((ampls[(n_points - 1)::-1], ampls, ampls[:-(n_points + 1):-1]))
    ext_noise = np.convolve(ext_ampls, window, 'same')
    # cut back to original interval
    noise = ext_noise[n_points:-n_points]
    # extra correction to account for convolve mode='full' instead of 'same' (needed for JIT-ting)
    # noise = noise[n_points//2 - 1:-n_points//2]
    return noise


def scargle_noise_at_freq(fs, time, resid, window_width=1.0):
    """Calculate the Lomb-Scargle noise at a given set of frequencies

    Parameters
    ----------
    fs: numpy.ndarray[Any, dtype[float]]
        The frequencies at which to calculate the noise
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    resid: numpy.ndarray[Any, dtype[float]]
        Residual measurement values of the time series
    window_width: float
        The width of the window used to compute the noise spectrum,
        in inverse unit of the time array (i.e. 1/d if time is in d).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The noise level calculated as the mean in a window around the
        frequency in the residual periodogram
    
    Notes
    -----
    The values calculated here capture the amount of noise on fitting a
    sinusoid of a certain frequency to all data points.
    Not to be confused with the noise on the individual data points of the
    time series.
    """
    freqs, ampls = astropy_scargle(time, resid)  # use defaults to get full amplitude spectrum
    margin = window_width / 2
    # mask the frequency ranges and compute the noise
    f_masks = [(freqs > f - margin) & (freqs <= f + margin) for f in fs]
    noise = np.array([np.mean(ampls[mask]) for mask in f_masks])
    return noise


def spectral_window(time, freqs):
    """Computes the modulus square of the spectral window W_N(f) of a set of
    time points at the given frequencies.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    freqs: numpy.ndarray[Any, dtype[float]]
        Frequency points to calculate the window. Inverse unit of `time`
        
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The spectral window at the given frequencies, |W(freqs)|^2
    
    Notes
    -----
    The spectral window is the Fourier transform of the window function
    w_N(t) = 1/N sum(Dirac(t - t_i))
    The time points do not need to be equidistant.
    The normalisation is such that 1.0 is returned at frequency 0.
    """
    n_time = len(time)
    cos_term = np.sum(np.cos(2.0 * np.pi * freqs * time.reshape(n_time, 1)), axis=0)
    sin_term = np.sum(np.sin(2.0 * np.pi * freqs * time.reshape(n_time, 1)), axis=0)
    win_kernel = cos_term**2 + sin_term**2
    # Normalise such that win_kernel(nu = 0.0) = 1.0
    spec_win = win_kernel / n_time**2
    return spec_win


@nb.njit(cache=True)
def scargle(time, flux, f0=0, fn=0, df=0, norm='amplitude'):
    """Scargle periodogram with no weights.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    df: float
        Frequency sampling space of the periodogram
        If left zero, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str
        Normalisation of the periodogram. Choose from:
        'amplitude', 'density' or 'distribution'
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        f1: numpy.ndarray[Any, dtype[float]]
            Frequencies at which the periodogram was calculated
        s1: numpy.ndarray[Any, dtype[float]]
            The periodogram spectrum in the chosen units
    
    Notes
    -----
    Translated from Fortran (and just as fast when JIT-ted with Numba!)
        Computation of Scargles periodogram without explicit tau
        calculation, with iteration (Method Cuypers)
    
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    
    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # setup
    n = len(flux_ms)
    t_tot = np.ptp(time_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if df == 0:
        df = 0.1 / t_tot
    if fn == 0:
        fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    # pre-assign some memory
    ss = np.zeros(nf)
    sc = np.zeros(nf)
    ss2 = np.zeros(nf)
    sc2 = np.zeros(nf)
    # here is the actual calculation:
    two_pi = 2 * np.pi
    for i in range(n):
        t_f0 = (time_ms[i] * two_pi * f0) % two_pi
        sin_f0 = np.sin(t_f0)
        cos_f0 = np.cos(t_f0)
        mc_1_a = 2 * sin_f0 * cos_f0
        mc_1_b = cos_f0 * cos_f0 - sin_f0 * sin_f0

        t_df = (time_ms[i] * two_pi * df) % two_pi
        sin_df = np.sin(t_df)
        cos_df = np.cos(t_df)
        mc_2_a = 2 * sin_df * cos_df
        mc_2_b = cos_df * cos_df - sin_df * sin_df
        
        sin_f0_s = sin_f0 * flux_ms[i]
        cos_f0_s = cos_f0 * flux_ms[i]
        for j in range(nf):
            ss[j] = ss[j] + sin_f0_s
            sc[j] = sc[j] + cos_f0_s
            temp_cos_f0_s = cos_f0_s
            cos_f0_s = temp_cos_f0_s * cos_df - sin_f0_s * sin_df
            sin_f0_s = sin_f0_s * cos_df + temp_cos_f0_s * sin_df
            ss2[j] = ss2[j] + mc_1_a
            sc2[j] = sc2[j] + mc_1_b
            temp_mc_1_b = mc_1_b
            mc_1_b = temp_mc_1_b * mc_2_b - mc_1_a * mc_2_a
            mc_1_a = mc_1_a * mc_2_b + temp_mc_1_b * mc_2_a
    
    f1 = f0 + np.arange(nf) * df
    s1 = ((sc**2 * (n - sc2) + ss**2 * (n + sc2) - 2 * ss * sc * ss2) / (n**2 - sc2**2 - ss2**2))
    # conversion to amplitude spectrum (or power density or statistical distribution)
    if not np.isfinite(s1[0]):
        s1[0] = 0  # sometimes there can be a nan value
    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(flux_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
    else:  # unnormalised (PSD?)
        s1 = s1
    return f1, s1


@nb.njit(cache=True)
def scargle_simple_psd(time, flux):
    """Scargle periodogram with no weights and PSD normalisation.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f1: numpy.ndarray[Any, dtype[float]]
            Frequencies at which the periodogram was calculated
        s1: numpy.ndarray[Any, dtype[float]]
            The periodogram spectrum in the chosen units

    Notes
    -----
    Translated from Fortran (and just as fast when JIT-ted with Numba!)
        Computation of Scargles periodogram without explicit tau
        calculation, with iteration (Method Cuypers)

    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # setup
    n = len(flux_ms)
    t_tot = np.ptp(time_ms)
    f0 = 0
    df = 0.1 / t_tot
    fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    # pre-assign some memory
    ss = np.zeros(nf)
    sc = np.zeros(nf)
    ss2 = np.zeros(nf)
    sc2 = np.zeros(nf)
    # here is the actual calculation:
    two_pi = 2 * np.pi
    for i in range(n):
        t_f0 = (time_ms[i] * two_pi * f0) % two_pi
        sin_f0 = np.sin(t_f0)
        cos_f0 = np.cos(t_f0)
        mc_1_a = 2 * sin_f0 * cos_f0
        mc_1_b = cos_f0 * cos_f0 - sin_f0 * sin_f0

        t_df = (time_ms[i] * two_pi * df) % two_pi
        sin_df = np.sin(t_df)
        cos_df = np.cos(t_df)
        mc_2_a = 2 * sin_df * cos_df
        mc_2_b = cos_df * cos_df - sin_df * sin_df

        sin_f0_s = sin_f0 * flux_ms[i]
        cos_f0_s = cos_f0 * flux_ms[i]
        for j in range(nf):
            ss[j] = ss[j] + sin_f0_s
            sc[j] = sc[j] + cos_f0_s
            temp_cos_f0_s = cos_f0_s
            cos_f0_s = temp_cos_f0_s * cos_df - sin_f0_s * sin_df
            sin_f0_s = sin_f0_s * cos_df + temp_cos_f0_s * sin_df
            ss2[j] = ss2[j] + mc_1_a
            sc2[j] = sc2[j] + mc_1_b
            temp_mc_1_b = mc_1_b
            mc_1_b = temp_mc_1_b * mc_2_b - mc_1_a * mc_2_a
            mc_1_a = mc_1_a * mc_2_b + temp_mc_1_b * mc_2_a

    f1 = f0 + np.arange(nf) * df
    s1 = ((sc**2 * (n - sc2) + ss**2 * (n + sc2) - 2 * ss * sc * ss2) / (n**2 - sc2**2 - ss2**2))
    # conversion to amplitude spectrum (or power density or statistical distribution)
    if not np.isfinite(s1[0]):
        s1[0] = 0  # sometimes there can be a nan value
    return f1, s1


@nb.njit(cache=True)
def scargle_ampl_single(time, flux, f):
    """Amplitude at one frequency from the Scargle periodogram

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f: float
        A single frequency
    
    Returns
    -------
    float
        Amplitude at the given frequency
    
    See Also
    --------
    scargle_ampl, scargle_phase, scargle_phase_single
    
    Notes
    -----
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(time_ms)):
        cos_tau += np.cos(four_pi * f * time_ms[j])
        sin_tau += np.sin(four_pi * f * time_ms[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(time_ms)):
        cos = np.cos(two_pi * f * (time_ms[j] - tau))
        sin = np.sin(two_pi * f * (time_ms[j] - tau))
        s_cos += flux_ms[j] * cos
        cos_2 += cos**2
        s_sin += flux_ms[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos_2 = s_cos**2 / cos_2
    b_sin_2 = s_sin**2 / sin_2
    # amplitude
    ampl = (a_cos_2 + b_sin_2) / 2
    ampl = np.sqrt(4 / len(time_ms)) * np.sqrt(ampl)  # conversion to amplitude
    return ampl


@nb.njit(cache=True)
def scargle_ampl(time, flux, fs):
    """Amplitude at one or a set of frequencies from the Scargle periodogram
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    fs: numpy.ndarray[Any, dtype[float]]
        A set of frequencies
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Amplitude at the given frequencies
    
    See Also
    --------
    scargle_phase
    
    Notes
    -----
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)
    ampl = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(time_ms)):
            cos_tau += np.cos(four_pi * fs[i] * time_ms[j])
            sin_tau += np.sin(four_pi * fs[i] * time_ms[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(time_ms)):
            cos = np.cos(two_pi * fs[i] * (time_ms[j] - tau))
            sin = np.sin(two_pi * fs[i] * (time_ms[j] - tau))
            s_cos += flux_ms[j] * cos
            cos_2 += cos**2
            s_sin += flux_ms[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos_2 = s_cos**2 / cos_2
        b_sin_2 = s_sin**2 / sin_2
        # amplitude
        ampl[i] = (a_cos_2 + b_sin_2) / 2
        ampl[i] = np.sqrt(4 / len(time_ms)) * np.sqrt(ampl[i])  # conversion to amplitude
    return ampl


@nb.njit(cache=True)
def scargle_phase_single(time, flux, f):
    """Phase at one frequency from the Scargle periodogram
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f: float
        A single frequency
    
    Returns
    -------
    float
        Phase at the given frequency
    
    See Also
    --------
    scargle_phase, scargle_ampl_single
    
    Notes
    -----
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(len(time_ms)):
        cos_tau += np.cos(four_pi * f * time_ms[j])
        sin_tau += np.sin(four_pi * f * time_ms[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)
    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(len(time_ms)):
        cos = np.cos(two_pi * f * (time_ms[j] - tau))
        sin = np.sin(two_pi * f * (time_ms[j] - tau))
        s_cos += flux_ms[j] * cos
        cos_2 += cos**2
        s_sin += flux_ms[j] * sin
        sin_2 += sin**2
    # final calculations
    a_cos = s_cos / cos_2**(1/2)
    b_sin = s_sin / sin_2**(1/2)
    # sine phase (radians)
    phi = np.pi/2 - np.arctan2(b_sin, a_cos) - two_pi * f * tau
    return phi


@nb.njit(cache=True)
def scargle_phase(time, flux, fs):
    """Phase at one or a set of frequencies from the Scargle periodogram
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    fs: numpy.ndarray[Any, dtype[float]]
        A set of frequencies
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Phase at the given frequencies
    
    Notes
    -----
    Uses a slightly modified version of the function in Hocke 1997
    ("Phase estimation with the Lomb-Scargle periodogram method")
    https://www.researchgate.net/publication/283359043_Phase_estimation_with_the_Lomb-Scargle_periodogram_method
    (only difference is an extra pi/2 for changing cos phase to sin phase)
    
    Notes
    -----
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # multiples of pi
    two_pi = 2 * np.pi
    four_pi = 4 * np.pi
    fs = np.atleast_1d(fs)
    phi = np.zeros(len(fs))
    for i in range(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(len(time_ms)):
            cos_tau += np.cos(four_pi * fs[i] * time_ms[j])
            sin_tau += np.sin(four_pi * fs[i] * time_ms[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)
        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(len(time_ms)):
            cos = np.cos(two_pi * fs[i] * (time_ms[j] - tau))
            sin = np.sin(two_pi * fs[i] * (time_ms[j] - tau))
            s_cos += flux_ms[j] * cos
            cos_2 += cos**2
            s_sin += flux_ms[j] * sin
            sin_2 += sin**2
        # final calculations
        a_cos = s_cos / cos_2**(1/2)
        b_sin = s_sin / sin_2**(1/2)
        # sine phase (radians)
        phi[i] = np.pi / 2 - np.arctan2(b_sin, a_cos) - two_pi * fs[i] * tau
    return phi


def astropy_scargle(time, flux, f0=0, fn=0, df=0, norm='amplitude'):
    """Wrapper for the astropy Scargle periodogram.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    df: float
        Frequency sampling space of the periodogram
        If left zero, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str
        Normalisation of the periodogram. Choose from:
        'amplitude', 'density' or 'distribution'

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f1: numpy.ndarray[Any, dtype[float]]
            Frequencies at which the periodogram was calculated
        s1: numpy.ndarray[Any, dtype[float]]
            The periodogram spectrum in the chosen units

    Notes
    -----
    Approximation using fft, much faster than the other scargle in mode='fast'.
    Beware of computing narrower frequency windows, as there is inconsistency
    when doing this.
    
    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    
    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.
    
    Note that the astropy implementation uses functions under the hood
    that use the blas package for multithreading by default.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # setup
    n = len(flux)
    t_tot = np.ptp(time_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if df == 0:
        df = 0.1 / t_tot
    if fn == 0:
        fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    f1 = f0 + np.arange(nf) * df
    # use the astropy fast algorithm and normalise afterward
    ls = apy.LombScargle(time_ms, flux_ms, fit_mean=False, center_data=False)
    s1 = ls.power(f1, normalization='psd', method='fast', assume_regular_frequency=True)
    # replace negative by zero (just in case - have seen it happen)
    s1[s1 < 0] = 0
    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(flux_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / n) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / n) * s1 * t_tot
    else:  # unnormalised (PSD?)
        s1 = s1
    return f1, s1


def astropy_scargle_simple_psd(time, flux):
    """Wrapper for the astropy Scargle periodogram and PSD normalisation.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f1: numpy.ndarray[Any, dtype[float]]
            Frequencies at which the periodogram was calculated
        s1: numpy.ndarray[Any, dtype[float]]
            The periodogram spectrum in the chosen units

    Notes
    -----
    Approximation using fft, much faster than the other scargle in mode='fast'.
    Beware of computing narrower frequency windows, as there is inconsistency
    when doing this.

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract

    The time array is mean subtracted to reduce correlation between
    frequencies and phases. The flux array is mean subtracted to avoid
    a large peak at frequency equal to zero.

    Note that the astropy implementation uses functions under the hood
    that use the blas package for multithreading by default.
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s
    # setup
    t_tot = np.ptp(time_ms)
    f0 = 0
    df = 0.1 / t_tot
    fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1
    f1 = np.arange(nf) * df
    # use the astropy fast algorithm and normalise afterward
    ls = apy.LombScargle(time_ms, flux_ms, fit_mean=False, center_data=False)
    s1 = ls.power(f1, normalization='psd', method='fast', assume_regular_frequency=True)
    # replace negative or nan by zero
    s1[0] = 0
    return f1, s1


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
    n_harm_r, completeness_r, distance_r = af.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
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
    periods, phase_disp = phase_dispersion_minimisation(time, flux, f_n, local=False)
    ampls = scargle_ampl(time, flux, 1 / periods)
    psi_measure = ampls / phase_disp
    # also check the number of harmonics at each period and include into best f
    n_harm, completeness, distance = af.harmonic_series_length(1 / periods, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness
    # select the best period, refine it and check double P
    p_orb = periods[np.argmax(psi_h_measure)]
    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 / p_orb, 1.01 / p_orb, 0.00001 / p_orb)
    n_harm_r, completeness_r, distance_r = af.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    completeness_p = (len(harmonics) / (f_nyquist // (1 / p_orb)))
    completeness_p_l = (len(harmonics[harmonic_n <= 15]) / (f_nyquist // (1 / p_orb)))
    # check these (commonly missed) multiples
    n_multiply = np.array([1/2, 2, 3, 4, 5])
    p_multiples = p_orb * n_multiply
    n_harm_r_m, completeness_r_m, distance_r_m = af.harmonic_series_length(1/p_multiples, f_n, freq_res, f_nyquist)
    h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain
    # if there are very high numbers, add double that fraction for testing
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    if np.any(test_frac[2:] > 3):
        n_multiply = np.append(n_multiply, [2 * n_multiply[2:][test_frac[2:] > 3]])
        p_multiples = p_orb * n_multiply
        n_harm_r_m, completeness_r_m, distance_r_m = af.harmonic_series_length(1/p_multiples, f_n, freq_res, f_nyquist)
        h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain
    # compute diagnostic fractions that need to meet some threshold
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    compl_frac = completeness_r_m / completeness_p
    # doubling the period may be done if the harmonic filling factor below f_16 is very high
    f_cut = np.max(f_n[harmonics][harmonic_n <= 15])
    f_n_c = f_n[f_n <= f_cut]
    n_harm_r_2, completeness_r_2, distance_r_2 = af.harmonic_series_length(1/p_multiples, f_n_c, freq_res, f_nyquist)
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
        n_harm_r2, completeness_r2, distance_r2 = af.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
        h_measure_2 = n_harm_r2 * completeness_r2  # compute h_measure for constraining a domain
        mask_peak = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
        i_min_dist = np.argmin(distance_r2[mask_peak])
        p_orb = 1 / f_refine_2[mask_peak][i_min_dist]
    return p_orb


@nb.njit(cache=True)
def calc_iid_normal_likelihood(residuals):
    """Natural logarithm of the independent and identically distributed likelihood function.

    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is estimated as σ^2 = sum((residuals)^2)/n

    Parameters
    ----------
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    
    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)
    # like = -n / 2 * (np.log(2 * np.pi * np.sum(residuals**2) / n) + 1)
    # originally un-JIT-ted function, but for loop is quicker with numba
    sum_r_2 = 0
    for i, r in enumerate(residuals):
        sum_r_2 += r**2
    like = -n / 2 * (np.log(2 * np.pi * sum_r_2 / n) + 1)
    return like


def calc_approx_did_likelihood(time, residuals):
    """Approximation for the likelihood using periodograms.

    This function approximates the dependent likelihood for correlated data.
    ln(L(θ)) =  -n ln(2 pi) - sum(ln(PSD(residuals)))

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model

    Returns
    -------
    float
        Log-likelihood approximation
    """
    n = len(time)
    # Compute the Lomb-Scargle periodogram of the data
    freqs, psd = astropy_scargle_simple_psd(time, residuals)  # automatically mean subtracted
    # Compute the Whittle likelihood
    like = -n * np.log(2 * np.pi) - np.sum(np.log(psd))
    return like


def calc_whittle_likelihood(time, flux, model):
    """Whittle likelihood approximation using periodograms.

    This function approximates the dependent likelihood for correlated data.
    It assumes the data is identically distributed according to a normal
    distribution.
    ln(L(θ)) =  -n ln(2 pi) - sum(ln(PSD_model) + (PSD_data / PSD_model))

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    model: numpy.ndarray[Any, dtype[float]]
        Model values of the time series

    Returns
    -------
    float
        Log-likelihood approximation
    """
    n = len(time)
    # Compute the Lomb-Scargle periodogram of the data
    freqs, psd_d = astropy_scargle_simple_psd(time, flux)  # automatically mean subtracted
    # Compute the Lomb-Scargle periodogram of the model
    freqs_m, psd_m = astropy_scargle_simple_psd(time, model)  # automatically mean subtracted
    # Avoid division by zero in likelihood calculation
    psd_m = np.maximum(psd_m, 1e-15)  # Ensure numerical stability
    # Compute the Whittle likelihood
    like = -n * np.log(2 * np.pi) - np.sum(np.log(psd_m) + (psd_d / psd_m))
    return like


def calc_did_normal_likelihood(time, residuals):
    """Natural logarithm of the dependent and identically distributed likelihood function.

    Correlation in the data is taken into account. The data is still assumed to be
    identically distributed according to a normal distribution.

    ln(L(θ)) = -n ln(2 pi) / 2 - ln(det(∑)) / 2 - residuals @ ∑^-1 @ residuals^T / 2
    ∑ is the covariance matrix

    The covariance matrix is calculated using the power spectral density, following
    the Wiener–Khinchin theorem.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model

    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)
    # calculate the PSD, fast
    freqs, psd = astropy_scargle_simple_psd(time, residuals)
    # calculate the autocorrelation function
    psd_ext = np.append(psd, psd[-1:0:-1])  # double the PSD domain for ifft
    acf = np.fft.ifft(psd_ext)
    # unbias the variance measure and put the array the right way around
    acf = np.real(np.append(acf[len(freqs):], acf[:len(freqs)])) * n / (n - 1)
    # calculate the acf lags
    lags = np.fft.fftfreq(len(psd_ext), d=(freqs[1] - freqs[0]))
    lags = np.append(lags[len(psd):], lags[:len(psd)])  # put them the right way around
    # make the lags matrix, but re-use the same matrix
    matrix = time - time[:, np.newaxis]  # lags_matrix, same as np.outer
    # interpolate - I need the lags at specific times
    matrix = np.interp(matrix, lags, acf)  # cov_matrix, already mean-subtracted in PSD
    # Compute the Cholesky decomposition of cov_matrix (by definition positive definite)
    matrix = sp.linalg.cho_factor(matrix, lower=False, overwrite_a=True, check_finite=False)  # cho_decomp
    # Solve M @ x = v^T using the Cholesky factorization (x = M^-1 v^T)
    x = sp.linalg.cho_solve(matrix, residuals[:, np.newaxis], check_finite=False)
    # log of the exponent - analogous to the matrix multiplication
    ln_exp = (residuals @ x)[0]  # v @ x = v @ M^-1 @ v^T
    # log of the determinant (avoids too small eigenvalues that would result in 0)
    ln_det = 2 * np.sum(np.log(np.diag(matrix[0])))
    # likelihood for multivariate normal distribution
    like = -n * np.log(2 * np.pi) / 2 - ln_det / 2 - ln_exp / 2
    return like


def calc_ddd_normal_likelihood(time, residuals, flux_err):
    """Natural logarithm of the dependent and differently distributed likelihood function.

    Only assumes that the data is distributed according to a normal distribution.
    Correlation in the data is taken into account. The measurement errors take
    precedence over the measured variance in the data. This means the distributions
    need not be identical, either.

    ln(L(θ)) = -n ln(2 pi) / 2 - ln(det(∑)) / 2 - residuals @ ∑^-1 @ residuals^T / 2
    ∑ is the covariance matrix

    The covariance matrix is calculated using the power spectral density, following
    the Wiener–Khinchin theorem.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values

    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)
    # calculate the PSD, fast
    freqs, psd = astropy_scargle_simple_psd(time, residuals)
    # calculate the autocorrelation function
    psd_ext = np.append(psd, psd[-1:0:-1])  # double the PSD domain for ifft
    acf = np.fft.ifft(psd_ext)
    # unbias the variance measure and put the array the right way around
    acf = np.real(np.append(acf[len(freqs):], acf[:len(freqs)])) * n / (n - 1)
    # calculate the acf lags
    lags = np.fft.fftfreq(len(psd_ext), d=(freqs[1] - freqs[0]))
    lags = np.append(lags[len(psd):], lags[:len(psd)])  # put them the right way around
    # make the lags matrix, but re-use the same matrix
    matrix = time - time[:, np.newaxis]  # lags_matrix, same as np.outer
    # interpolate - I need the lags at specific times
    matrix = np.interp(matrix, lags, acf)  # cov_matrix, already mean-subtracted in PSD
    # substitute individual data errors if given
    var = matrix[0, 0]  # diag elements are the same by construction
    corr_matrix = matrix / var  # divide out the variance to get correlation matrix
    err_matrix = flux_err * flux_err[:, np.newaxis]  # make matrix of measurement errors (same as np.outer)
    matrix = err_matrix * corr_matrix  # multiply to get back to covariance
    # Compute the Cholesky decomposition of cov_matrix (by definition positive definite)
    matrix = sp.linalg.cho_factor(matrix, lower=False, overwrite_a=True, check_finite=False)  # cho_decomp
    # Solve M @ x = v^T using the Cholesky factorization (x = M^-1 v^T)
    x = sp.linalg.cho_solve(matrix, residuals[:, np.newaxis], check_finite=False)
    # log of the exponent - analogous to the matrix multiplication
    ln_exp = (residuals @ x)[0]  # v @ x = v @ M^-1 @ v^T
    # log of the determinant (avoids too small eigenvalues that would result in 0)
    ln_det = 2 * np.sum(np.log(np.diag(matrix[0])))
    # likelihood for multivariate normal distribution
    like = -n * np.log(2 * np.pi) / 2 - ln_det / 2 - ln_exp / 2
    return like


def calc_likelihood(time=None, flux=None, residuals=None, flux_err=None, func=calc_iid_normal_likelihood):
    """Natural logarithm of the likelihood function.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: None, numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    func: function
        The likelihood function to use for the calculation
        Choose from: calc_iid_normal_likelihood, calc_approx_did_likelihood,
        calc_whittle_likelihood, calc_did_normal_likelihood,
        calc_ddd_normal_likelihood

    Returns
    -------
    float
        Natural logarithm of the likelihood

    Notes
    -----
    Choose between a conventional iid simplification of the likelihood,
    a full matrix implementation that costs a lot of memory for large datasets,
    or some approximations in-between.
    """
    # make a dict of the given arguments
    kwargs = {'time': time, 'flux': flux, 'residuals': residuals, 'flux_err': flux_err}
    # check what the chosen function needs
    func_args = list(inspect.signature(func).parameters)
    # make a dict of those
    args_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in func_args}
    # feed to the function
    like = func(**args_dict)
    return like


@nb.njit(cache=True)
def calc_bic(residuals, n_param):
    """Bayesian Information Criterion.
    
    Parameters
    ----------
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    n_param: int
        Number of free parameters in the model
    
    Returns
    -------
    float
        Bayesian Information Criterion
    
    Notes
    -----
    BIC = −2 ln(L(θ)) + k ln(n)
    where L is the likelihood as function of the parameters θ, n the number of data points
    and k the number of free parameters.
    
    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is the error variance estimated as σ^2 = sum((residuals)^2)/n
    (residuals being data - model).
    
    Combining this gives:
    BIC = n ln(2 pi σ^2) + n + k ln(n)
    """
    n = len(residuals)
    # bic = n * np.log(2 * np.pi * np.sum(residuals**2) / n) + n + n_param * np.log(n)
    # originally JIT-ted function, but with for loop is slightly quicker
    sum_r_2 = ut.std_unb(residuals, n)
    bic = n * np.log(2 * np.pi * sum_r_2 / n) + n + n_param * np.log(n)
    return bic


def calc_bic_2(residuals, n_param, flux_err=None):
    """Bayesian Information Criterion with correlated likelihood function.

    BIC = k ln(n) − 2 ln(L(θ))
    where L is the likelihood as function of the parameters θ, n the number of data points
    and k the number of free parameters.

    Parameters
    ----------
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    n_param: int
        Number of free parameters in the model
    flux_err: None, numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values

    Returns
    -------
    float
        Bayesian Information Criterion
    """
    n = len(residuals)
    bic = n_param * np.log(n) - 2 * calc_likelihood(residuals, flux_err=flux_err)
    return bic


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


def extract_single(time, flux, f0=0, fn=0, select='a', verbose=True):
    """Extract a single frequency from a time series using oversampling
    of the periodogram.
    
    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    select: str
        Select the next frequency based on amplitude 'a' or signal-to-noise 'sn'
    verbose: bool
        If set to True, this function will print some information
    
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
    
    Notes
    -----
    The extracted frequency is based on the highest amplitude or signal-to-noise
    in the periodogram (over the interval where it is calculated). The highest
    peak is oversampled by a factor 100 to get a precise measurement.
    
    If and only if the full periodogram is calculated using the defaults
    for f0=0 and fn=0, the fast implementation of astropy scargle is used.
    It is accurate to a very high degree when used like this and gives
    a significant speed increase. It cannot be used on smaller intervals.
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution
    # full LS periodogram
    if (f0 == 0) & (fn == 0):
        # inconsistency with astropy_scargle for small freq intervals, so only do the full pd
        freqs, ampls = astropy_scargle(time, flux, f0=f0, fn=fn, df=df)
    else:
        freqs, ampls = scargle(time, flux, f0=f0, fn=fn, df=df)
    # selection step based on flux to noise (refine step keeps using ampl)
    if select == 'sn':
        noise_spectrum = scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0)
        ampls = ampls / noise_spectrum
    # select highest value
    p1 = np.argmax(ampls)
    # check if we pick the boundary frequency
    if p1 in [0, len(freqs) - 1]:
        if verbose:
            print(f'Edge of frequency range {freqs[p1]} at position {p1} during extraction phase 1.')
    # now refine once by increasing the frequency resolution x100
    f_left = max(freqs[p1] - df, df / 10)  # may not get too low
    f_right = freqs[p1] + df
    f_refine, a_refine = scargle(time, flux, f0=f_left, fn=f_right, df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if p2 in [0, len(f_refine) - 1]:
        if verbose:
            print(f'Edge of frequency range {f_refine[p2]} at position {p2} during extraction phase 2.')
    f_final = f_refine[p2]
    a_final = a_refine[p2]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(time, flux, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


@nb.njit(cache=True)
def extract_single_narrow(time, flux, f0=0, fn=0, verbose=True):
    """Extract a single frequency from a time series using oversampling
    of the periodogram.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Starting frequency of the periodogram.
        If left zero, default is f0 = 1/(100*T)
    fn: float
        Last frequency of the periodogram.
        If left zero, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    verbose: bool
        If set to True, this function will print some information

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

    Notes
    -----
    The extracted frequency is based on the highest amplitude in the
    periodogram (over the interval where it is calculated). The highest
    peak is oversampled by a factor 100 to get a precise measurement.

    If and only if the full periodogram is calculated using the defaults
    for f0 and fn, the fast implementation of astropy scargle is used.
    It is accurate to a very high degree when used like this and gives
    a significant speed increase.
    
    Same as extract_single, but meant for narrow frequency ranges. Much
    slower on the full frequency range, even though JIT-ted.
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution
    # full LS periodogram (over a narrow range)
    freqs, ampls = scargle(time, flux, f0=f0, fn=fn, df=df)
    p1 = np.argmax(ampls)
    # check if we pick the boundary frequency
    if p1 in [0, len(freqs) - 1]:
        if verbose:
            print(f'Edge of frequency range {ut.float_to_str(freqs[p1], dec=2)} at position {p1} '
                  f'during extraction phase 1.')
    # now refine once by increasing the frequency resolution x100
    f_left = max(freqs[p1] - df, df / 10)  # may not get too low
    f_right = freqs[p1] + df
    f_refine, a_refine = scargle(time, flux, f0=f_left, fn=f_right, df=df/100)
    p2 = np.argmax(a_refine)
    # check if we pick the boundary frequency
    if p2 in [0, len(f_refine) - 1]:
        if verbose:
            print(f'Edge of frequency range {ut.float_to_str(f_refine[p2], dec=2)} at position {p2} '
                  f'during extraction phase 2.')
    f_final = f_refine[p2]
    a_final = a_refine[p2]
    # finally, compute the phase (and make sure it stays within + and - pi)
    ph_final = scargle_phase_single(time, flux, f_final)
    ph_final = (ph_final + np.pi) % (2 * np.pi) - np.pi
    return f_final, a_final, ph_final


def refine_subset(time, flux, close_f, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=True):
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
    verbose: bool
        If set to True, this function will print some information
    
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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)
    # determine initial bic
    model_sinusoid_ncf = sum_sines(time, np.delete(f_n, close_f), np.delete(a_n, close_f), np.delete(ph_n, close_f))
    cur_resid = flux - (model_sinusoid_ncf + sum_sines(time, f_n[close_f], a_n[close_f], ph_n[close_f]))
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    f_n_temp, a_n_temp, ph_n_temp = np.copy(f_n), np.copy(a_n), np.copy(ph_n)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_f - n_harm)
    bic_prev = calc_bic(resid, n_param)
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
                a_j = scargle_ampl_single(time, resid, f_j)
                ph_j = scargle_phase_single(time, resid, f_j)
            else:
                f0 = f_n_temp[j] - freq_res
                fn = f_n_temp[j] + freq_res
                f_j, a_j, ph_j = extract_single(time, resid, f0=f0, fn=fn, select='a', verbose=verbose)
            f_n_temp[j], a_n_temp[j], ph_n_temp[j] = f_j, a_j, ph_j
            cur_resid -= sum_sines(time, np.array([f_j]), np.array([a_j]), np.array([ph_j]))
        # as a last model-refining step, redetermine the constant and slope
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        # calculate BIC before moving to the next iteration
        bic = calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        if np.round(d_bic, 2) > 0:
            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic
            accept = True
        if verbose:
            print(f'N_f= {n_f}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) '
                  f'- N_refine= {n_g}, f= {f_j:1.6f}, a= {a_j:1.6f}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (total= {bic_init - bic_prev:1.2f}) - end refinement', end='\r')
    # redo the constant and slope without the last iteration of changes
    resid = flux - (model_sinusoid_ncf + sum_sines(time, f_n[close_f], a_n[close_f], ph_n[close_f]))
    const, slope = linear_pars(time, resid, i_chunks)
    return const, slope, f_n, a_n, ph_n


def extract_sinusoids(time, flux, i_chunks, p_orb=0, f_n=None, a_n=None, ph_n=None, select='hybrid',
                      verbose=True):
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
    select: str, optional
        Select the next frequency based on amplitude ('a'),
        signal-to-noise ('sn'), or hybrid ('hybrid') (first a then sn).
    verbose: bool, optional
        If set to True, this function will print some information

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
    Uses a delta BIC > 2 stopping criterion.

    [Author's note] Although it is my belief that doing a non-linear
    multi-sinusoid fit at each iteration of the prewhitening is the
    ideal approach, it is also a very (very!) time-consuming one and this
    algorithm aims to be fast while approaching the optimal solution.
    """
    if f_n is None:
        f_n = np.array([])
    if a_n is None:
        a_n = np.array([])
    if ph_n is None:
        ph_n = np.array([])
    # setup
    freq_res = 1.5 / np.ptp(time)  # frequency resolution
    n_sectors = len(i_chunks)
    n_freq = len(f_n)
    if n_freq > 0:
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
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
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = calc_bic(resid, n_param)  # initialise current BIC to the mean (and slope) subtracted flux
    bic_init = bic_prev
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')
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
        f_i, a_i, ph_i = extract_single(time, resid, f0=0, fn=0, select=select, verbose=verbose)
        # now iterate over close frequencies (around f_i) a number of times to improve them
        f_n_temp, a_n_temp, ph_n_temp = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)
        close_f = af.f_within_rayleigh(n_freq_cur, f_n_temp, freq_res)
        model_sinusoid_r = sum_sines(time, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        model_sinusoid_r -= sum_sines(time, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        if len(close_f) > 1:
            refine_out = refine_subset(time, flux, close_f, p_orb, const, slope, f_n_temp, a_n_temp, ph_n_temp,
                                       i_chunks, verbose=verbose)
            const, slope, f_n_temp, a_n_temp, ph_n_temp = refine_out
        # as a last model-refining step, redetermine the constant and slope
        model_sinusoid_n = sum_sines(time, f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f])
        cur_resid -= (model_sinusoid_n - model_sinusoid_r)  # add the changes to the sinusoid residuals
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        # calculate BIC before moving to the next iteration
        n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq_cur + 1 - n_harm)
        bic = calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        if np.round(d_bic, 2) > 2:
            # accept the new frequency
            f_n, a_n, ph_n = np.append(f_n, f_i), np.append(a_n, a_i), np.append(ph_n, ph_i)
            # adjust the shifted frequencies
            f_n[close_f], a_n[close_f], ph_n[close_f] = f_n_temp[close_f], a_n_temp[close_f], ph_n_temp[close_f]
            bic_prev = bic
        if verbose:
            print(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) - '
                  f'f= {f_i:1.6f}, a= {a_i:1.6f}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')
    # lastly re-determine slope and const
    cur_resid += (model_sinusoid_n - model_sinusoid_r)  # undo last change
    const, slope = linear_pars(time, cur_resid, i_chunks)
    return const, slope, f_n, a_n, ph_n


def extract_harmonics(time, flux, p_orb, i_chunks, f_n=None, a_n=None, ph_n=None, verbose=True):
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
    verbose: bool, optional
        If set to True, this function will print some information
    
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
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
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
    bic_init = calc_bic(resid, n_param)
    bic_prev = bic_init
    if verbose:
        print(f'N_f= {n_freq}, BIC= {bic_init:1.2f} (delta= N/A) - start extraction')
    # loop over candidates and try to extract (BIC decreases by 2 or more)
    n_h_acc = []
    for h_c in h_candidate:
        f_c = h_c / p_orb
        a_c = scargle_ampl_single(time, resid, f_c)
        ph_c = scargle_phase_single(time, resid, f_c)
        ph_c = np.mod(ph_c + np.pi, 2 * np.pi) - np.pi  # make sure the phase stays within + and - pi
        # redetermine the constant and slope
        model_sinusoid_n = sum_sines(time, np.array([f_c]), np.array([a_c]), np.array([ph_c]))
        cur_resid -= model_sinusoid_n
        const, slope = linear_pars(time, cur_resid, i_chunks)
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        # determine new BIC and whether it improved
        n_harm_cur = n_harm + len(n_h_acc) + 1
        n_param = 2 * n_sectors + 1 * (n_harm_cur > 0) + 2 * n_harm_cur + 3 * (n_freq - n_harm)
        bic = calc_bic(resid, n_param)
        d_bic = bic_prev - bic
        if np.round(d_bic, 2) > 2:
            # h_c is accepted, add it to the final list and continue
            bic_prev = bic
            f_n, a_n, ph_n = np.append(f_n, f_c), np.append(a_n, a_c), np.append(ph_n, ph_c)
            n_h_acc.append(h_c)
        else:
            # h_c is rejected, revert to previous residual
            cur_resid += model_sinusoid_n
            const, slope = linear_pars(time, cur_resid, i_chunks)
            resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        if verbose:
            print(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {d_bic:1.2f}, total= {bic_init - bic:1.2f}) - '
                  f'h= {h_c}', end='\r')
    if verbose:
        print(f'N_f= {len(f_n)}, BIC= {bic_prev:1.2f} (delta= {bic_init - bic_prev:1.2f}) - end extraction')
        print(f'Successfully extracted harmonics {n_h_acc}')
    return const, slope, f_n, a_n, ph_n


def fix_harmonic_frequency(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=True):
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
    verbose: bool
        If set to True, this function will print some information

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
    harmonics, harmonic_n = af.find_harmonics_tolerance(f_n, p_orb, f_tol=freq_res / 2)
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
    bic_init = calc_bic(resid, n_param)
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
        a_i = scargle_ampl_single(time, resid, f_i)
        ph_i = scargle_phase_single(time, resid, f_i)
        ph_i = np.mod(ph_i + np.pi, 2 * np.pi) - np.pi  # make sure the phase stays within + and - pi
        # make a model of the new sinusoid and add it to the full sinusoid residual
        model_sinusoid_n = sum_sines(time, np.array([f_i]), np.array([a_i]), np.array([ph_i]))
        cur_resid -= model_sinusoid_n
        # add to freq list and removal list
        f_new, a_new, ph_new = np.append(f_new, f_i), np.append(a_new, a_i), np.append(ph_new, ph_i)
        remove_harm_c = np.append(remove_harm_c, remove)
        if verbose:
            print(f'Harmonic number {n} re-extracted, replacing {len(remove)} candidates', end='\r')
    # lastly re-determine slope and const (not needed here)
    # const, slope = linear_pars(time, cur_resid, i_chunks)
    # finally, remove all the designated sinusoids from the lists and add the new ones
    f_n = np.append(np.delete(f_n, remove_harm_c), f_new)
    a_n = np.append(np.delete(a_n, remove_harm_c), a_new)
    ph_n = np.append(np.delete(ph_n, remove_harm_c), ph_new)
    # re-extract the non-harmonics
    n_freq = len(f_n)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
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
        f_n[i], a_n[i], ph_n[i] = extract_single(time, resid, f0=fl, fn=fr, select='a', verbose=verbose)
        ph_n[i] = np.mod(ph_n[i] + np.pi, 2 * np.pi) - np.pi  # make sure the phase stays within + and - pi
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
    if verbose:
        resid = cur_resid - linear_curve(time, const, slope, i_chunks)
        n_param = 2 * n_sectors + 1 + 2 * n_harm + 3 * (n_freq - n_harm)
        bic = calc_bic(resid, n_param)
        print(f'Candidate harmonics replaced: {n_harm_init} ({n_harm} left). ')
        print(f'N_f= {len(f_n)}, BIC= {bic:1.2f} (delta= {bic_init - bic:1.2f})')
    return const, slope, f_n, a_n, ph_n


@nb.njit(cache=True)
def remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=True):
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
    verbose: bool
        If set to True, this function will print some information
    
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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_harm = len(harmonics)
    # indices of single frequencies to remove
    remove_single = np.zeros(0, dtype=np.int_)
    # determine initial bic
    model_sinusoid = sum_sines(time, f_n, a_n, ph_n)
    cur_resid = flux - model_sinusoid  # the residual after subtracting the model of sinusoids
    resid = cur_resid - linear_curve(time, const, slope, i_chunks)
    n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * (n_freq - n_harm)
    bic_prev = calc_bic(resid, n_param)
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
            bic = calc_bic(resid, n_param)
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
    if verbose:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        print(f'Single frequencies removed: {n_freq - len(f_n)}')
        print(f'N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')
    return const, slope, f_n, a_n, ph_n


@nb.njit(cache=True)
def replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=True):
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
    verbose: bool
        If set to True, this function will print some information

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
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    non_harm = np.delete(np.arange(n_freq), harmonics)
    n_harm = len(harmonics)
    # make an array of sets of frequencies (non-harmonic) to be investigated for replacement
    close_f_groups = af.chains_within_rayleigh(f_n[non_harm], freq_res)
    close_f_groups = [non_harm[group] for group in close_f_groups]  # convert to the right indices
    f_sets = [g[np.arange(p1, p2 + 1)]
              for g in close_f_groups
              for p1 in range(len(g) - 1)
              for p2 in range(p1 + 1, len(g))]
    # make an array of sets of frequencies (now with harmonics) to be investigated for replacement
    close_f_groups = af.chains_within_rayleigh(f_n, freq_res)
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
    bic_prev = calc_bic(resid, n_param)
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
                a_i = scargle_ampl(time, resid, f_n[harm_i])
                ph_i = scargle_phase(time, resid, f_n[harm_i])
            else:
                edges = [min(f_n[set_i]) - freq_res, max(f_n[set_i]) + freq_res]
                out = extract_single_narrow(time, resid, f0=edges[0], fn=edges[1], verbose=verbose)
                f_i, a_i, ph_i = np.array([out[0]]), np.array([out[1]]), np.array([out[2]])
            # make a model including the new freq
            model_sinusoid_n = sum_sines(time, f_i, a_i, ph_i)
            resid -= model_sinusoid_n
            const, slope = linear_pars(time, resid, i_chunks)  # redetermine const and slope
            resid -= linear_curve(time, const, slope, i_chunks)
            # number of parameters and bic
            n_freq_i = n_freq - sum([len(f_sets[j]) for j in remove_sets]) - len(set_i) + len(f_new) + len(f_i) - n_harm
            n_param = 2 * n_sectors + 1 * (n_harm > 0) + 2 * n_harm + 3 * n_freq_i
            bic = calc_bic(resid, n_param)
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
    if verbose:
        str_bic = ut.float_to_str(bic_prev, dec=2)
        str_delta = ut.float_to_str(bic_init - bic_prev, dec=2)
        print(f'Frequency sets replaced by a single frequency: {len(remove_sets)} ({len(i_to_remove)} frequencies). ')
        print(f'N_f= {len(f_n)}, BIC= {str_bic} (delta= {str_delta})')
    return const, slope, f_n, a_n, ph_n


def reduce_sinusoids(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=True):
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
    verbose: bool
        If set to True, this function will print some information
    
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
    out_a = remove_sinusoids_single(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_a
    # Now go on to trying to replace sets of frequencies that are close together
    out_b = replace_sinusoid_groups(time, flux, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    return const, slope, f_n, a_n, ph_n


def select_sinusoids(time, flux, flux_err, p_orb, const, slope, f_n, a_n, ph_n, i_chunks, verbose=False):
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
    verbose: bool
        If set to True, this function will print some information

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
    n_points = len(time)
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    
    # obtain the errors on the sine waves (depends on residual and thus model)
    model_lin = linear_curve(time, const, slope, i_chunks)
    model_sin = sum_sines(time, f_n, a_n, ph_n)
    residuals = flux - (model_lin + model_sin)
    errors = formal_uncertainties(time, residuals, flux_err, a_n, i_chunks)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = errors
    
    # find the insignificant frequencies
    remove_sigma = af.remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3, sigma_f=3)
    
    # apply the signal-to-noise threshold
    noise_at_f = scargle_noise_at_freq(f_n, time, residuals, window_width=1.0)
    remove_snr = af.remove_insignificant_snr(a_n, noise_at_f, n_points)
    
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
        harmonics, harmonic_n = af.select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol=freq_res / 2, sigma_f=3)
        passed_harmonic[harmonics] = True
    else:
        harmonics = np.array([], dtype=int)
    if verbose:
        print(f'Number of frequencies passed criteria: {np.sum(passed_both)} of {len(f_n)}. '
              f'Candidate harmonics: {np.sum(passed_harmonic)}, of which {np.sum(passed_both[harmonics])} passed.')
        
    return passed_sigma, passed_snr, passed_both, passed_harmonic
