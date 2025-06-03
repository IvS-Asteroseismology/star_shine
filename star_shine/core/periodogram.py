"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains periodograms functions for time series data.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import astropy.timeseries as apy


# get the number of available cpu threads
n_proc = nb.get_num_threads()


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
        return np.sum(np.abs(a - np.mean(a)) ** 2)  # if mean instead of sum, this is variance

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


@nb.njit(cache=True, parallel=True)
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
            periods[7 * i:7 * i + 7] = np.arange(1, 8) / f

    # stay below the maximum
    periods = periods[periods < np.ptp(time)]

    # and above the minimum
    periods = periods[periods > (2 * np.min(time[1:] - time[:-1]))]

    # compute the dispersion measures
    n_periods = len(periods)
    pd_all = np.zeros(n_periods)
    for i in nb.prange(n_periods):
        fold = fold_time_series_phase(time, periods[i], 0)
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
    # use defaults to get full amplitude spectrum
    freqs, ampls = scargle_parallel(time, resid)

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


def scargle_noise_at_freq(fs, time, resid, window_width=1.):
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
    # use defaults to get full amplitude spectrum
    freqs, ampls = scargle_parallel(time, resid)
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
    win_kernel = cos_term ** 2 + sin_term ** 2

    # Normalise such that win_kernel(nu = 0.0) = 1.0
    spec_win = win_kernel / n_time ** 2

    return spec_win


@nb.njit(cache=True)
def _scargle_core(time, flux, nt, f0, df, nf):
    """Core algorithm of the Scargle periodogram with no weights.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series, mean subtracted.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series, mean subtracted.
    nt: int
        Length of the time series.
    f0: float
        Starting frequency of the periodogram.
    df: float
        Frequency sampling space of the periodogram.
    nf: int
        Length of the frequency array.

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The periodogram spectrum in the chosen units.

    Notes
    -----
    Translated from Fortran (and just as fast when JIT-ted with Numba!)
        Computation of Scargles periodogram without explicit tau
        calculation, with iteration (Method Cuypers)

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # pre-assign some memory
    ss = np.zeros(nf)
    sc = np.zeros(nf)
    ss2 = np.zeros(nf)
    sc2 = np.zeros(nf)

    # here is the actual calculation:
    two_pi = 2 * np.pi
    for i in range(nt):
        t_f0 = (time[i] * two_pi * f0) % two_pi
        sin_f0 = np.sin(t_f0)
        cos_f0 = np.cos(t_f0)
        mc_1_a = 2 * sin_f0 * cos_f0
        mc_1_b = cos_f0 * cos_f0 - sin_f0 * sin_f0

        t_df = (time[i] * two_pi * df) % two_pi
        sin_df = np.sin(t_df)
        cos_df = np.cos(t_df)
        mc_2_a = 2 * sin_df * cos_df
        mc_2_b = cos_df * cos_df - sin_df * sin_df

        sin_f0_s = sin_f0 * flux[i]
        cos_f0_s = cos_f0 * flux[i]
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

    s1 = ((sc ** 2 * (nt - sc2) + ss ** 2 * (nt + sc2) - 2 * ss * sc * ss2) / (nt ** 2 - sc2 ** 2 - ss2 ** 2))

    return s1


@nb.njit(cache=True)
def scargle(time, flux, f0=-1, fn=-1, df=-1, norm='amplitude'):
    """Scargle periodogram with no weights.

    The time array is mean subtracted to reduce correlation between frequencies and phases.
    The flux array is mean subtracted to avoid a large peak at frequency equal to zero.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float, optional
        Starting frequency of the periodogram.
        If left -1, default is f0 = 1/(100*T)
    fn: float, optional
        Last frequency of the periodogram.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    df: float, optional
        Frequency sampling space of the periodogram
        If left -1, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str, optional
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

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_sorter = np.argsort(time)
    time_ms = time[time_sorter] - mean_t
    flux_ms = flux[time_sorter] - mean_s

    # setup
    nt = len(time_ms)
    t_tot = np.ptp(time_ms)
    if f0 == -1:
        f0 = 0.01 / t_tot  # lower than T/100 no good
    if df == -1:
        df = 0.1 / t_tot  # default frequency sampling is about 1/10 of frequency resolution
    if fn == -1:
        fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1

    # do the scargle calculations
    s1 = _scargle_core(time_ms, flux_ms, nt, f0, df, nf)
    f1 = f0 + np.arange(nf) * df

    # conversion to amplitude spectrum (or power density or statistical distribution)
    if not np.isfinite(s1[0]):
        s1[0] = 0  # sometimes there can be a nan value

    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(flux_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / nt) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / nt) * s1 * t_tot
    else:  # unnormalised (PSD?)
        s1 = s1

    return f1, s1


@nb.njit(cache=True, parallel=True)
def scargle_parallel(time, flux, f0=-1, fn=-1, df=-1, norm='amplitude'):
    """Parallel Scargle periodogram with no weights.

    Non-parallel overhead amounts to less than 10%.

    The time array is mean subtracted to reduce correlation between frequencies and phases.
    The flux array is mean subtracted to avoid a large peak at frequency equal to zero.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float, optional
        Starting frequency of the periodogram.
        If left -1, default is f0 = 1/(100*T)
    fn: float, optional
        Last frequency of the periodogram.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    df: float, optional
        Frequency sampling space of the periodogram
        If left -1, default is df = 1/(10*T) = oversampling factor of ten (recommended)
    norm: str, optional
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

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_sorter = np.argsort(time)
    time_ms = time[time_sorter] - mean_t
    flux_ms = flux[time_sorter] - mean_s

    # setup
    nt = len(time_ms)
    t_tot = np.ptp(time_ms)
    if f0 == -1:
        f0 = 0.01 / t_tot  # lower than T/100 no good
    if df == -1:
        df = 0.1 / t_tot  # default frequency sampling is about 1/10 of frequency resolution
    if fn == -1:
        fn = 1 / (2 * np.min(time_ms[1:] - time_ms[:-1]))
    nf = int((fn - f0) / df + 0.001) + 1

    # frequency array, broken into chunks
    f1 = f0 + np.arange(nf) * df
    f1_chunks = np.array_split(f1, n_proc)
    chunk_i = np.array([len(chunk) for chunk in f1_chunks])
    chunk_i = np.array([np.sum(chunk_i[:i]) for i in range(1, len(chunk_i) + 1)])
    chunk_i = np.concatenate((np.zeros(1), chunk_i)).astype(np.int_)

    # parallelised for loop
    s1 = np.zeros(nf)
    for i in nb.prange(n_proc):
        _f0 = f1_chunks[i][0]
        _nf = len(f1_chunks[i])
        s1[chunk_i[i]:chunk_i[i + 1]] = _scargle_core(time_ms, flux_ms, nt, _f0, df, _nf)

    # conversion to amplitude spectrum (or power density or statistical distribution)
    if not np.isfinite(s1[0]):
        s1[0] = 0  # sometimes there can be a nan value

    # convert to the wanted normalisation
    if norm == 'distribution':  # statistical distribution
        s1 /= np.var(flux_ms)
    elif norm == 'amplitude':  # amplitude spectrum
        s1 = np.sqrt(4 / nt) * np.sqrt(s1)
    elif norm == 'density':  # power density
        s1 = (4 / nt) * s1 * t_tot
    else:  # unnormalised (PSD?)
        s1 = s1

    return f1, s1


@nb.njit(cache=True)
def scargle_ampl_phase_single(time, flux, f):
    """Amplitude and phase at one or a set of frequencies from the Scargle periodogram.

    The time array is mean subtracted to reduce correlation between frequencies and phases.
    The flux array is mean subtracted to avoid a large peak at frequency equal to zero.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    f: float
        A frequency to calculate amplitude and phase at.

    Returns
    -------
    tuple
        Two numbers consisting of:
        float
            Amplitude at the given frequency.
        float
            Phase at the given frequency.

    See Also
    --------
    scargle_phase

    Notes
    -----
    For the phase calculation:
    Uses a slightly modified version of the function in Hocke 1997
    ("Phase estimation with the Lomb-Scargle periodogram method")
    https://www.researchgate.net/publication/283359043_Phase_estimation_with_the_Lomb-Scargle_periodogram_method
    (only difference is an extra pi/2 for changing cos phase to sin phase)
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s

    # setup
    nt = len(time_ms)
    pi = np.pi
    two_pi = 2 * pi
    four_pi = 4 * pi

    # define tau
    cos_tau = 0
    sin_tau = 0
    for j in range(nt):
        cos_tau += np.cos(four_pi * f * time_ms[j])
        sin_tau += np.sin(four_pi * f * time_ms[j])
    tau = 1 / (four_pi * f) * np.arctan2(sin_tau, cos_tau)  # tau(f)

    # define the general cos and sin functions
    s_cos = 0
    cos_2 = 0
    s_sin = 0
    sin_2 = 0
    for j in range(nt):
        cos = np.cos(two_pi * f * (time_ms[j] - tau))
        sin = np.sin(two_pi * f * (time_ms[j] - tau))
        s_cos += flux_ms[j] * cos
        cos_2 += cos ** 2
        s_sin += flux_ms[j] * sin
        sin_2 += sin ** 2

    # final calculations
    a_cos = s_cos / cos_2 ** (1 / 2)
    b_sin = s_sin / sin_2 ** (1 / 2)

    # amplitude
    ampl = (a_cos ** 2 + b_sin ** 2) / 2
    ampl = np.sqrt(4 / nt) * np.sqrt(ampl)  # conversion to amplitude

    # sine phase (radians)
    phi = pi / 2 - np.arctan2(b_sin, a_cos) - two_pi * f * tau
    phi = (phi + pi) % two_pi - pi  # make sure the phase stays within + and - pi

    return ampl, phi


@nb.njit(cache=True, parallel=True)
def scargle_ampl_phase(time, flux, fs):
    """Amplitude at one or a set of frequencies from the Scargle periodogram.

    Fast for limited numbers of frequencies, not a replacement for the full periodogram.
    Has an overhead compared to scargle_ampl_phase_single, so do not use for single fs value.
    For 2 frequencies it is just as fast, and 3+ it is more efficient.

    The time array is mean subtracted to reduce correlation between frequencies and phases.
    The flux array is mean subtracted to avoid a large peak at frequency equal to zero.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    fs: numpy.ndarray[Any, dtype[float]]
        A set of frequencies to calculate amplitude and phase at.

    Returns
    -------
    tuple
        Two arrays consisting of:
        numpy.ndarray[Any, dtype[float]]
            Amplitude at the given frequencies
        numpy.ndarray[Any, dtype[float]]
            Phase at the given frequencies

    See Also
    --------
    scargle_phase

    Notes
    -----
    For the phase calculation:
    Uses a slightly modified version of the function in Hocke 1997
    ("Phase estimation with the Lomb-Scargle periodogram method")
    https://www.researchgate.net/publication/283359043_Phase_estimation_with_the_Lomb-Scargle_periodogram_method
    (only difference is an extra pi/2 for changing cos phase to sin phase)
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_ms = time - mean_t
    flux_ms = flux - mean_s

    # setup
    nt = len(time_ms)
    pi = np.pi
    two_pi = 2 * pi
    four_pi = 4 * pi
    fs = np.atleast_1d(fs)
    ampl = np.zeros(len(fs))
    phi = np.zeros(len(fs))

    for i in nb.prange(len(fs)):
        # define tau
        cos_tau = 0
        sin_tau = 0
        for j in range(nt):
            cos_tau += np.cos(four_pi * fs[i] * time_ms[j])
            sin_tau += np.sin(four_pi * fs[i] * time_ms[j])
        tau = 1 / (four_pi * fs[i]) * np.arctan2(sin_tau, cos_tau)  # tau(f)

        # define the general cos and sin functions
        s_cos = 0
        cos_2 = 0
        s_sin = 0
        sin_2 = 0
        for j in range(nt):
            cos = np.cos(two_pi * fs[i] * (time_ms[j] - tau))
            sin = np.sin(two_pi * fs[i] * (time_ms[j] - tau))
            s_cos += flux_ms[j] * cos
            cos_2 += cos ** 2
            s_sin += flux_ms[j] * sin
            sin_2 += sin ** 2

        # final calculations
        a_cos = s_cos / cos_2 ** (1 / 2)
        b_sin = s_sin / sin_2 ** (1 / 2)

        # amplitude
        ampl[i] = (a_cos ** 2 + b_sin ** 2) / 2
        ampl[i] = np.sqrt(4 / nt) * np.sqrt(ampl[i])  # conversion to amplitude

        # sine phase (radians)
        phi[i] = pi / 2 - np.arctan2(b_sin, a_cos) - two_pi * fs[i] * tau
        phi[i] = (phi[i] + pi) % two_pi - pi  # make sure the phase stays within + and - pi

    return ampl, phi


def astropy_scargle(time, flux, f0=0, fn=0, df=0, norm='amplitude'):
    """Wrapper for the astropy Scargle periodogram.

    The time array is mean subtracted to reduce correlation between frequencies and phases.
    The flux array is mean subtracted to avoid a large peak at frequency equal to zero.

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
    Approximation using fft, much faster (in mode='fast') than the single threaded scargle (about x10).
    Note that the astropy implementation uses functions under the hood that use the blas package for
    multithreading by default. Compared to the parallel scargle it is similar in speed.
    Beware of computing narrower frequency windows, as there is inconsistency when doing this.
    It is also generally less accurate.

    Useful extra information: VanderPlas 2018,
    https://ui.adsabs.harvard.edu/abs/2018ApJS..236...16V/abstract
    """
    # time and flux are mean subtracted (reduce correlation and avoid peak at f=0)
    mean_t = np.mean(time)
    mean_s = np.mean(flux)
    time_sorter = np.argsort(time)
    time_ms = time[time_sorter] - mean_t
    flux_ms = flux[time_sorter] - mean_s

    # setup
    n = len(flux)
    t_tot = np.ptp(time_ms)
    f0 = max(f0, 0.01 / t_tot)  # don't go lower than T/100
    if df == 0:
        df = 0.1 / t_tot  # default frequency sampling is about 1/10 of frequency resolution
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
