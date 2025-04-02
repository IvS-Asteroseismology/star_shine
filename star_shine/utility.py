"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains utility functions for data processing, unit conversions
and loading in data (some functions specific to TESS data).

Code written by: Luc IJspeert
"""

import os
import datetime
import fnmatch
import h5py
import numpy as np
import numba as nb

try:
    import pandas as pd  # optional functionality
except ImportError:
    pass
import astropy.io.fits as fits
try:
    import arviz as az  # optional functionality
except ImportError:
    pass

from . import timeseries_functions as tsf
from . import analysis_functions as af
from . import visualisation as vis
from .. import config


@nb.njit(cache=True)
def float_to_str(x, dec=2):
    """Convert float to string for Numba up to some decimal place
    
    Parameters
    ----------
    x: float
        Value to convert
    dec: int
        Number of decimals (be careful with large numbers here)
    
    Returns
    -------
    s: str
        String with the value x
    """
    x_round = np.round(x, dec)
    x_int = int(x_round)
    x_dec = int(np.abs(np.round(x_round - x_int, dec)) * 10**dec)
    s = str(x_int) + '.' + str(x_dec).zfill(dec)
    return s


@nb.njit(cache=True)
def weighted_mean(x, w):
    """Weighted mean since Numba doesn't support numpy.average
    
    Parameters
    ----------
    x: numpy.ndarray[float]
        Values to calculate the mean over
    w: numpy.ndarray[float]
        Weights corresponding to each value
    
    Returns
    -------
    w_mean: float
        Mean of x weighted by w
    """
    w_mean = np.sum(x * w) / np.sum(w)
    return w_mean


@nb.njit(cache=True)
def std_unb(x, n):
    """Unbiased standard deviation

    Parameters
    ----------
    x: numpy.ndarray[float]
        Values to calculate the std over
    n: int
        Number of degrees of freedom

    Returns
    -------
    std: float
        Unbiased standard deviation
    """
    residuals = x - np.mean(x)
    # tested to be faster in numba than np.sum(x**2)
    sum_r_2 = 0
    for r in residuals:
        sum_r_2 += r**2
    std = np.sqrt(sum_r_2 / n)  # unbiased standard deviation of the residuals
    return std


@nb.njit(cache=True)
def decimal_figures(x, n_sf):
    """Determine the number of decimal figures to print given a target
    number of significant figures
    
    Parameters
    ----------
    x: float
        Value to determine the number of decimals for
    n_sf: int
        Number of significant figures to compute
    
    Returns
    -------
    decimals: int
        Number of decimal places to round to
    """
    if (x != 0):
        decimals = (n_sf - 1) - int(np.floor(np.log10(abs(x))))
    else:
        decimals = 1
    return decimals


@nb.njit(cache=True)
def signal_to_noise_threshold(n_points):
    """Determine the signal-to-noise threshold for accepting frequencies
    based on the number of points
    
    Parameters
    ----------
    n_points: int
        Number of data points in the time series
    
    Returns
    -------
    sn_thr: float
        Signal-to-noise threshold for this data set
    
    Notes
    -----
    Baran & Koen 2021, eq 6.
    (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)
    """
    sn_thr = 1.201 * np.sqrt(1.05 * np.log(n_points) + 7.184)
    sn_thr = np.round(sn_thr, 2)  # round to two decimals
    return sn_thr


@nb.njit(cache=True)
def normalise_counts(flux_counts, flux_counts_err, i_chunks):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median.
    
    Parameters
    ----------
    flux_counts: numpy.ndarray[float]
        Flux measurement values in counts of the time series
    flux_counts_err: numpy.ndarray[float]
        Errors in the flux measurements
    i_chunks: numpy.ndarray[float]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    tuple:
        flux_norm: numpy.ndarray[float]
            Normalised flux measurements
        flux_err_norm: numpy.ndarray[float]
            Normalised flux errors (zeros if flux_counts_err is None)
        medians: numpy.ndarray[float]
            Median flux counts per chunk
    
    Notes
    -----
    The result is positive and varies around one.
    The signal is processed per chunk.
    """
    medians = np.zeros(len(i_chunks))
    flux_norm = np.zeros(len(flux_counts))
    flux_err_norm = np.zeros(len(flux_counts))
    for i, ch in enumerate(i_chunks):
        medians[i] = np.median(flux_counts[ch[0]:ch[1]])
        flux_norm[ch[0]:ch[1]] = flux_counts[ch[0]:ch[1]] / medians[i]
        flux_err_norm[ch[0]:ch[1]] = flux_counts_err[ch[0]:ch[1]] / medians[i]
    return flux_norm, flux_err_norm, medians


def sort_chunks(chunk_sorter, i_chunks):
    """Sorts the time chunks based on chunk_sorter and updates the indices accordingly.

    Parameters
    ----------
    chunk_sorter: np.ndarray
        Sort indices of the time chunks based on their means
    i_chunks: numpy.ndarray[int]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    time_sorter: numpy.ndarray[int]
        Sort indices for the full array
    i_chunks: numpy.ndarray[int]
        Updated pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    """
    # Update i_chunks to be in the sorted order
    sorted_i_chunks = i_chunks[chunk_sorter]

    # Create full index array corresponding to the sorted chunks
    time_sorter = np.concatenate([np.arange(ch[0], ch[1]) for ch in sorted_i_chunks])

    # update i_chunks to the full sorted time array
    sorted_chunk_len = sorted_i_chunks[:, 1] - sorted_i_chunks[:, 0]
    index_high = np.cumsum(sorted_chunk_len)
    index_low = np.append([0], index_high[:-1])
    i_chunks = np.vstack((index_low, index_high)).T

    return time_sorter, i_chunks


def load_csv_data(file_name):
    """Load in the data from a single csv file.

    Change column names in the config file.

    Parameters
    ----------
    file_name: str
        File name (including path) of the data.

    Returns
    -------
    time: numpy.ndarray[float]
        Timestamps of the time series
    flux: numpy.ndarray[float]
        Measurement values of the time series
    flux_err: numpy.ndarray[float]
        Errors in the measurement values
    """
    # get the right columns with pandas
    col_names = [config.CN_TIME, config.CN_FLUX, config.CN_FLUX_ERR]
    df = pd.read_csv(file_name, usecols=col_names)

    # convert to numpy arrays
    time, flux, flux_err = df[col_names].values.T

    return time, flux, flux_err


def load_fits_data(file_name):
    """Load in the data from a single fits file.

    The SAP flux is Simple Aperture Photometry, the processed data can be PDC_SAP, KSP_SAP, or other
    depending on the data source. Change column names in the config file.

    Parameters
    ----------
    file_name: str
        File name (including path) of the data.
    
    Returns
    -------
    time: numpy.ndarray[float]
        Timestamps of the time series
    flux: numpy.ndarray[float]
        Measurement values of the time series
    flux_err: numpy.ndarray[float]
        Errors in the measurement values
    qual_flags: numpy.ndarray[int]
        Integer values representing the quality of the
        data points. Zero means good quality.
    crowdsap: float
        Light contamination parameter (1-third_light)

    Notes
    -----
    Originally written for TESS data products, adapted to be flexible.
    """
    # grab the time series data
    with fits.open(file_name, mode='readonly') as hdul:
        # time stamps and flux measurements
        time = hdul[1].data[config.CF_TIME]
        flux = hdul[1].data[config.CF_FLUX]
        flux_err = hdul[1].data[config.CF_FLUX_ERR]

        # quality flags
        qual_flags = hdul[1].data[config.CF_QUALITY]

        # get crowding numbers if found
        if 'CROWDSAP' in hdul[1].header.keys():
            crowdsap = hdul[1].header['CROWDSAP']
        else:
            crowdsap = -1

    return time, flux, flux_err, qual_flags, crowdsap


def load_light_curve(file_list, apply_flags=True):
    """Load in the data from a list of ((TESS specific) fits) files.

    Also stitches the light curves of each individual file together and normalises to the median.
    
    Parameters
    ----------
    file_list: list[str]
        A list of file names (including path) of the data.
    apply_flags: bool
        Whether to apply the quality flags to the time series data
        
    Returns
    -------
    tuple:
        time: numpy.ndarray[float]
            Timestamps of the time series
        flux: numpy.ndarray[float]
            Measurement values of the time series
        flux_err: numpy.ndarray[float]
            Errors in the measurement values
        i_chunks: numpy.ndarray[int]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        medians: numpy.ndarray[float]
            Median flux counts per chunk
    """
    time = np.array([])
    flux = np.array([])
    flux_err = np.array([])
    qual_flags = np.array([])
    i_chunks = np.zeros((0, 2))

    # load the data in list order
    for file in file_list:
        # get the data from the file with one of the following methods
        if file.endswith('.fits') | file.endswith('.fit'):
            ti, fl, err, qf, cro = load_fits_data(file)
        elif file.endswith('.csv') & ('pd' in locals()):
            ti, fl, err = load_csv_data(file)
            qf = np.zeros(len(ti))
            cro = 1
        else:
            ti, fl, err = np.loadtxt(file, usecols=(0, 1, 2), unpack=True)
            qf = np.zeros(len(ti))
            cro = 1

        # keep track of the data belonging to each time chunk
        chunk_index = [len(i_chunks), len(i_chunks) + len(ti)]
        if config.HALVE_CHUNKS & (file.endswith('.fits') | file.endswith('.fit')):
            chunk_index = [[len(i_chunks), len(i_chunks) + len(ti)//2],
                           [len(i_chunks) + len(ti)//2, len(i_chunks) + len(ti)]]
        i_chunks = np.append(i_chunks, chunk_index, axis=0)

        # append all other data
        time = np.append(time, ti)
        flux = np.append(flux, fl)
        flux_err = np.append(flux_err, err)
        qual_flags = np.append(qual_flags, qf)

    # sort chunks by time
    t_start = time[i_chunks[:0]]
    if np.any(np.diff(t_start) < 0):
        chunk_sorter = np.argsort(t_start)  # sort on chunk start time
        time_sorter, i_chunks = sort_chunks(chunk_sorter, i_chunks)
        time = time[time_sorter]
        flux = flux[time_sorter]
        flux_err = flux_err[time_sorter]
        qual_flags = qual_flags[time_sorter]

    # apply quality flags
    if apply_flags:
        # convert quality flags to boolean mask
        quality = (qual_flags == 0)
        time = time[quality]
        flux = flux[quality]
        flux_err = flux_err[quality]

    # clean up (on time and flux)
    finite = np.isfinite(time) & np.isfinite(flux)
    time = time[finite].astype(np.float_)
    flux = flux[finite].astype(np.float_)
    flux_err = flux_err[finite].astype(np.float_)

    # median normalise
    flux, flux_err, medians = normalise_counts(flux, flux_err, i_chunks)

    return time, flux, flux_err, i_chunks, medians


def group_frequencies_for_fit(a_n, g_min=20, g_max=25):
    """Groups frequencies into sets of g_min to g_max for multi-sine fitting
    
    Parameters
    ----------
    a_n: numpy.ndarray[float]
        The amplitudes of a number of sine waves
    g_min: int
        Minimum group size
    g_max: int
        Maximum group size (g_max > g_min)
    
    Returns
    -------
    groups: list[numpy.ndarray[int]]
        List of sets of indices indicating the groups
    
    Notes
    -----
    To make the task of fitting more manageable, the free parameters are binned into groups,
    in which the remaining parameters are kept fixed. Frequencies of similar amplitude are
    grouped together, and the group cut-off is determined by the biggest gaps in amplitude
    between frequencies, but group size is always kept between g_min and g_max. g_min < g_max.
    The idea of using amplitudes is that frequencies of similar amplitude have a similar
    amount of influence on each other.
    """
    # keep track of which freqs have been used with the sorted indices
    not_used = np.argsort(a_n)[::-1]
    groups = []
    while (len(not_used) > 0):
        if (len(not_used) > g_min + 1):
            a_diff = np.diff(a_n[not_used[g_min:g_max + 1]])
            i_max = np.argmin(a_diff)  # the diffs are negative so this is max absolute difference
            i_group = g_min + i_max + 1
            group_i = not_used[:i_group]
        else:
            group_i = np.copy(not_used)
            i_group = len(not_used)
        not_used = np.delete(not_used, np.arange(i_group))
        groups.append(group_i)
    return groups


@nb.njit(cache=True)
def correct_for_crowdsap(signal, crowdsap, i_sectors):
    """Correct the signal for flux contribution of a third source
    
    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).
    
    Returns
    -------
    cor_signal: numpy.ndarray[float]
        Measurement values of the time series corrected for
        contaminating light
    
    Notes
    -----
    Uses the parameter CROWDSAP included with some TESS data.
    flux_corrected = (flux - (1 - crowdsap)) / crowdsap
    where all quantities are median-normalised, including the result.
    This corresponds to subtracting a fraction of (1 - crowdsap) of third light
    from the (non-median-normalised) flux measurements.
    """
    cor_signal = np.zeros(len(signal))
    for i, s in enumerate(i_sectors):
        crowd = min(max(0, crowdsap[i]), 1)  # clip to avoid unphysical output
        cor_signal[s[0]:s[1]] = (signal[s[0]:s[1]] - 1 + crowd) / crowd
    return cor_signal


@nb.njit(cache=True)
def model_crowdsap(signal, crowdsap, i_sectors):
    """Incorporate flux contribution of a third source into the signal

    Parameters
    ----------
    signal: numpy.ndarray[float]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[float]
        Light contamination parameter (1-third_light) listed per sector
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans.
        These can indicate the TESS observation sectors, but taking
        half the sectors is recommended. If only a single curve is
        wanted, set i_half_s = np.array([[0, len(times)]]).

    Returns
    -------
    model: numpy.ndarray[float]
        Model of the signal incorporating light contamination

    Notes
    -----
    Does the opposite as correct_for_crowdsap, to be able to model the effect of
    third light to some degree (can only achieve an upper bound on CROWDSAP).
    """
    model = np.zeros(len(signal))
    for i, s in enumerate(i_sectors):
        crowd = min(max(0, crowdsap[i]), 1)  # clip to avoid unphysical output
        model[s[0]:s[1]] = signal[s[0]:s[1]] * crowd + 1 - crowd
    return model


def save_parameters_hdf5(file_name, sin_mean=None, sin_err=None, sin_hdi=None, sin_select=None, p_orb=None,
                         p_orb_err=None, p_orb_hdi=None, stats=None, i_sectors=None,
                         description='none', data_id='none'):
    """Save the full model parameters of the linear, sinusoid
    and eclipse models to an hdf5 file.

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    sin_mean: None, list[numpy.ndarray[float]]
        Parameter mean values for the linear and sinusoid model in the order they appear below.
        linear parameters: const, slope,
        sinusoid parameters: f_n, a_n, ph_n
    sin_err: None, list[numpy.ndarray[float]]
        Parameter error values for the linear and sinusoid model in the order they appear below.
        linear parameters: c_err, sl_err,
        sinusoid parameters: f_n_err, a_n_err, ph_n_err
    sin_hdi: None, list[numpy.ndarray[float]]
        Parameter hdi values for the linear and sinusoid model in the order they appear below.
        linear parameters: c_hdi, sl_hdi,
        sinusoid parameters: f_n_hdi, a_n_hdi, ph_n_hdi
    sin_select: None, list[numpy.ndarray[bool]]
        Sinusoids that pass certain selection criteria
        passed_sigma, passed_snr, passed_h
    p_orb: None, numpy.ndarray[float]
        p_orb of the EB
    p_orb_err: None, numpy.ndarray[float]
        Error values for the p_orb, p_err
    p_orb_hdi: None, numpy.ndarray[float]
        Hdi values for the p_orb, p_hdi
    stats: None, list[float]
        Some statistics: t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve.
    description: str
        Optional description of the saved results
    data_id: int, str
        Optional identifier for the data set used
    
    Returns
    -------
    None

    Notes
    -----
    The file contains the data sets (array-like) and attributes
    to describe the data, in hdf5 format.
    
    Any missing data is filled up with -1, 0 or True
    """
    # check for Nones
    if sin_mean is None:
        sin_mean = [np.zeros(1) for _ in range(5)]
    if sin_err is None:
        sin_err = [np.zeros(len(sin_mean[0])) for _ in range(2)]
        sin_err += [np.zeros(len(sin_mean[2])) for _ in range(3)]
    if sin_hdi is None:
        sin_hdi = [np.zeros((len(sin_mean[0]), 2)) for _ in range(2)]
        sin_hdi += [np.zeros((len(sin_mean[2]), 2)) for _ in range(3)]
    if sin_select is None:
        sin_select = [np.ones(len(sin_mean[2]), dtype=bool) for _ in range(3)]
    if p_orb is None:
        p_orb = -np.ones(1)
    if p_orb_err is None:
        p_orb_err = -np.ones(1)
    if p_orb_hdi is None:
        p_orb_hdi = -np.ones((1, 2))
    if stats is None:
        stats = [-1 for _ in range(7)]
    if i_sectors is None:
        i_sectors = -np.ones((1, 2))
    # unpack stats
    t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = stats
    # check some input
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    # create the file
    with h5py.File(file_name, 'w') as file:
        file.attrs['identifier'] = os.path.splitext(os.path.basename(file_name))[0]  # the file name without extension
        file.attrs['description'] = description
        file.attrs['data_id'] = data_id
        file.attrs['date_time'] = str(datetime.datetime.now())
        file.attrs['t_tot'] = t_tot  # total time base of observations
        file.attrs['t_mean'] = t_mean  # time reference (zero) point
        file.attrs['t_mean_s'] = t_mean_s  # time reference (zero) point per observing sector
        file.attrs['t_int'] = t_int  # integration time of observations
        file.attrs['n_param'] = n_param  # number of free parameters
        file.attrs['bic'] = bic  # Bayesian Information Criterion of the residuals
        file.attrs['noise_level'] = noise_level  # standard deviation of the residuals
        # orbital period
        file.create_dataset('p_orb', data=np.array([p_orb[0], p_orb_err[0], p_orb_hdi[0, 0], p_orb_hdi[0, 1]]))
        file['p_orb'].attrs['unit'] = 'd'
        file['p_orb'].attrs['description'] = 'Orbital period and error estimates.'
        # the linear model
        # y-intercepts
        file.create_dataset('const', data=sin_mean[0])
        file['const'].attrs['unit'] = 'median normalised flux'
        file['const'].attrs['description'] = 'y-intercept per analysed sector'
        file.create_dataset('c_err', data=sin_err[0])
        file['c_err'].attrs['unit'] = 'median normalised flux'
        file['c_err'].attrs['description'] = 'errors in the y-intercept per analysed sector'
        file.create_dataset('c_hdi', data=sin_hdi[0])
        file['c_hdi'].attrs['unit'] = 'median normalised flux'
        file['c_hdi'].attrs['description'] = 'HDI for the y-intercept per analysed sector'
        # slopes
        file.create_dataset('slope', data=sin_mean[1])
        file['slope'].attrs['unit'] = 'median normalised flux / d'
        file['slope'].attrs['description'] = 'slope per analysed sector'
        file.create_dataset('sl_err', data=sin_err[1])
        file['sl_err'].attrs['unit'] = 'median normalised flux / d'
        file['sl_err'].attrs['description'] = 'error in the slope per analysed sector'
        file.create_dataset('sl_hdi', data=sin_hdi[1])
        file['sl_hdi'].attrs['unit'] = 'median normalised flux / d'
        file['sl_hdi'].attrs['description'] = 'HDI for the slope per analysed sector'
        # sector indices
        file.create_dataset('i_sectors', data=i_sectors)
        file['i_sectors'].attrs['description'] = ('pairs of indices indicating chunks in time defining '
                                                  'the pieces of the piece-wise linear curve')
        # the sinusoid model
        # frequencies
        file.create_dataset('f_n', data=sin_mean[2])
        file['f_n'].attrs['unit'] = '1 / d'
        file['f_n'].attrs['description'] = 'frequencies of a number of sine waves'
        file.create_dataset('f_n_err', data=sin_err[2])
        file['f_n_err'].attrs['unit'] = '1 / d'
        file['f_n_err'].attrs['description'] = 'errors in the frequencies of a number of sine waves'
        file.create_dataset('f_n_hdi', data=sin_hdi[2])
        file['f_n_hdi'].attrs['unit'] = '1 / d'
        file['f_n_hdi'].attrs['description'] = 'HDI for the frequencies of a number of sine waves'
        # amplitudes
        file.create_dataset('a_n', data=sin_mean[3])
        file['a_n'].attrs['unit'] = 'median normalised flux'
        file['a_n'].attrs['description'] = 'amplitudes of a number of sine waves'
        file.create_dataset('a_n_err', data=sin_err[3])
        file['a_n_err'].attrs['unit'] = 'median normalised flux'
        file['a_n_err'].attrs['description'] = 'errors in the amplitudes of a number of sine waves'
        file.create_dataset('a_n_hdi', data=sin_hdi[3])
        file['a_n_hdi'].attrs['unit'] = 'median normalised flux'
        file['a_n_hdi'].attrs['description'] = 'HDI for the amplitudes of a number of sine waves'
        # phases
        file.create_dataset('ph_n', data=sin_mean[4])
        file['ph_n'].attrs['unit'] = 'radians'
        file['ph_n'].attrs['description'] = 'phases of a number of sine waves, with reference point t_mean'
        file.create_dataset('ph_n_err', data=sin_err[4])
        file['ph_n_err'].attrs['unit'] = 'radians'
        file['ph_n_err'].attrs['description'] = 'errors in the phases of a number of sine waves'
        file.create_dataset('ph_n_hdi', data=sin_hdi[4])
        file['ph_n_hdi'].attrs['unit'] = 'radians'
        file['ph_n_hdi'].attrs['description'] = 'HDI for the phases of a number of sine waves'
        # selection criteria
        file.create_dataset('passed_sigma', data=sin_select[0])
        file['passed_sigma'].attrs['description'] = 'sinusoids passing the sigma criterion'
        file.create_dataset('passed_snr', data=sin_select[1])
        file['passed_snr'].attrs['description'] = 'sinusoids passing the signal to noise criterion'
        file.create_dataset('passed_b', data=(sin_select[0] & sin_select[1]))
        file['passed_b'].attrs['description'] = 'sinusoids passing both the sigma and the signal to noise critera'
        file.create_dataset('passed_h', data=sin_select[2])
        file['passed_h'].attrs['description'] = 'harmonic sinusoids passing the sigma criterion'
    return None


def read_parameters_hdf5(file_name, verbose=False, h5py_file_kwargs={}):
    """Read the full model parameters of the linear, sinusoid
    and eclipse models to an hdf5 file.

    Parameters
    ----------
    file_name: str
        File name (including path) for loading the results.
    verbose: bool
        If set to True, this function will print some information.
    h5py_file_kwargs: dict
        Keyword arguments for opening the h5py file.
        Example: {'locking': False}, for a drive that does not support locking.

    Returns
    -------
    results: dict
        Contains:
        sin_mean: None, list[numpy.ndarray[float]]
            Parameter mean values for the linear and sinusoid model in the order they appear below.
            linear parameters: const, slope,
            sinusoid parameters: f_n, a_n, ph_n
        sin_err: None, list[numpy.ndarray[float]]
            Parameter error values for the linear and sinusoid model in the order they appear below.
            linear parameters: c_err, sl_err,
            sinusoid parameters: f_n_err, a_n_err, ph_n_err
        sin_hdi: None, list[numpy.ndarray[float]]
            Parameter hdi values for the linear and sinusoid model in the order they appear below.
            linear parameters: c_hdi, sl_hdi,
            sinusoid parameters: f_n_hdi, a_n_hdi, ph_n_hdi
        sin_select: None, list[numpy.ndarray[bool]]
            Sinusoids that pass certain selection criteria
            passed_sigma, passed_snr, passed_b, passed_h
        ephem: None, numpy.ndarray[float]
            Ephemerides of the EB, p_orb and t_zero
        ephem_err: None, numpy.ndarray[float]
            Error values for the ephemerides, p_err and t_zero_err
        ephem_err: None, numpy.ndarray[float]
            Hdi values for the ephemerides, p_hdi and t_zero_hdi
        phys_mean: None, numpy.ndarray[float]
            Parameter mean values for the physical eclipse model in the order they appear below.
            ecosw, esinw, cosi, phi_0, log_rr, log_sb,
            extra parametrisations: e, w, i, r_sum, r_rat, sb_rat
        phys_err: None, numpy.ndarray[float]
            Parameter error values for the physical eclipse model in the order they appear below.
            ecosw_err, esinw_err, cosi_err, phi_0_err, log_rr_err, log_sb_err,
            Extra parametrisation: e_err, w_err, i_err, r_sum_err, r_rat_err, sb_rat_err
        phys_hdi: None, numpy.ndarray[float]
            Parameter hdi values for the physical eclipse model in the order they appear below.
            ecosw_hdi, esinw_hdi, cosi_hdi, phi_0_hdi, log_rr_hdi, log_sb_hdi,
            Extra parametrisations: e_hdi, w_hdi, i_hdi, r_sum_hdi, r_rat_hdi, sb_rat_hdi
        timings: None, numpy.ndarray[float]
            Eclipse timings of minima and first and last contact points, internal tangency
            and eclipse depth of the primary and secondary:
            t_1, t_2, t_1_1, t_1_2, t_2_1, t_2_2, t_b_1_1, t_b_1_2, t_b_2_1, t_b_2_2, depth_1, depth_2
        timings_err: None, numpy.ndarray[float]
            Parameter error values for the eclipse timings and depths:
            t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
            t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err
        timings_hdi: None, numpy.ndarray[float]
            Parameter hdi values for the eclipse timings and depths:
            t_1_hdi, t_2_hdi, t_1_1_hdi, t_1_2_hdi, t_2_1_hdi, t_2_2_hdi,
            t_b_1_1_hdi, t_b_1_2_hdi, t_b_2_1_hdi, t_b_2_2_hdi, depth_1_hdi, depth_2_hdi
        timings_indiv_err: None, numpy.ndarray[float]
            Parameter error values for the individual eclipse timings and depths:
            t_1_err, t_2_err, t_1_1_err, t_1_2_err, t_2_1_err, t_2_2_err,
            t_b_1_1_err, t_b_1_2_err, t_b_2_1_err, t_b_2_2_err, depth_1_err, depth_2_err
        var_stats: None, list[union(float, numpy.ndarray[float])]
            Variability level diagnostic statistics
            std_1, std_2, std_3, std_4, ratios_1, ratios_2, ratios_3, ratios_4
        stats: None, list[float]
            Some statistics: t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level
        i_sectors: numpy.ndarray[int]
            Pair(s) of indices indicating the separately handled timespans
            in the piecewise-linear curve.
        text: list[str]
            Some information about the file and data:
            identifier, data_id, description and date_time
    """
    # check some input
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    # create the file
    with h5py.File(file_name, 'r', **h5py_file_kwargs) as file:
        identifier = file.attrs['identifier']
        description = file.attrs['description']
        data_id = file.attrs['data_id']
        date_time = file.attrs['date_time']
        t_tot = file.attrs['t_tot']
        t_mean = file.attrs['t_mean']
        t_mean_s = file.attrs['t_mean_s']
        t_int = file.attrs['t_int']
        n_param = file.attrs['n_param']
        bic = file.attrs['bic']
        noise_level = file.attrs['noise_level']
        # orbital period
        p_orb = np.copy(file['p_orb'])
        # the linear model
        # y-intercepts
        const = np.copy(file['const'])
        c_err = np.copy(file['c_err'])
        c_hdi = np.copy(file['c_hdi'])
        # slopes
        slope = np.copy(file['slope'])
        sl_err = np.copy(file['sl_err'])
        sl_hdi = np.copy(file['sl_hdi'])
        # sector indices
        i_sectors = np.copy(file['i_sectors'])
        # the sinusoid model
        # frequencies
        f_n = np.copy(file['f_n'])
        f_n_err = np.copy(file['f_n_err'])
        f_n_hdi = np.copy(file['f_n_hdi'])
        # amplitudes
        a_n = np.copy(file['a_n'])
        a_n_err = np.copy(file['a_n_err'])
        a_n_hdi = np.copy(file['a_n_hdi'])
        # phases
        ph_n = np.copy(file['ph_n'])
        ph_n_err = np.copy(file['ph_n_err'])
        ph_n_hdi = np.copy(file['ph_n_hdi'])
        # passing criteria
        passed_sigma = np.copy(file['passed_sigma'])
        passed_snr = np.copy(file['passed_snr'])
        passed_b = np.copy(file['passed_b'])
        passed_h = np.copy(file['passed_h'])
        # variability to eclipse depth ratios
        ratios_1 = np.copy(file['ratios_1'])
        ratios_2 = np.copy(file['ratios_2'])
        ratios_3 = np.copy(file['ratios_3'])
        ratios_4 = np.copy(file['ratios_4'])
    
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_hdi = [c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi]
    sin_select = [passed_sigma, passed_snr, passed_b, passed_h]
    p_orb = [p_orb[0]]
    p_orb_err = [p_orb[1]]
    p_orb_hdi = [p_orb[2:4]]
    var_stats = [ratios_1[0], ratios_2[0], ratios_3[0], ratios_4[0],
                 ratios_1[1:], ratios_2[1:], ratios_3[1:], ratios_4[1:]]
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    text = [identifier, data_id, description, date_time]
    # put everything in a dict
    results = {'sin_mean': sin_mean, 'sin_err': sin_err, 'sin_hdi': sin_hdi, 'sin_select': sin_select,
               'ephem': p_orb, 'ephem_err': p_orb_err, 'ephem_hdi': p_orb_hdi,
               'var_stats': var_stats, 'stats': stats, 'i_sectors': i_sectors, 'text': text}
    if verbose:
        print(f'Loaded analysis file with identifier: {identifier}, created on {date_time}. \n'
              f'data_id: {data_id}. Description: {description} \n')
    return results


def convert_hdf5_to_ascii(file_name):
    """Convert a save file in hdf5 format to multiple ascii save files
    
    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    
    Returns
    -------
    None
    
    Notes
    -----
    Only saves text files of certain groups of values when they
    are available in the hdf5 file.
    """
    # check the file name
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if (ext != '.hdf5'):
        file_name = file_name.replace(ext, '.hdf5')
    results = read_parameters_hdf5(file_name, verbose=False)
    # unpack all parameters
    const, slope, f_n, a_n, ph_n = results['sin_mean']
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = results['sin_err']
    c_hdi, sl_hdi, f_n_hdi, a_n_hdi, ph_n_hdi = results['sin_hdi']
    p_orb, t_zero = results['ephem']
    p_err, t_zero_err = results['ephem_err']
    p_hdi, t_zero_hdi = results['ephem_hdi']
    passed_sigma, passed_snr, passed_b, passed_h = results['sin_select']
    t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = results['stats']
    target_id, data_id, description, date_time = results['text']
    # check for -1 values and ignore those parameters - errors or hdis are always stored alongside
    # linear model parameters
    if (not np.all(const == -1)) & (not np.all(slope == -1)):
        data = np.column_stack((const, c_err, c_hdi[:, 0], c_hdi[:, 1],
                                slope, sl_err, sl_hdi[:, 0], sl_hdi[:, 1],
                                results['i_sectors'][:, 0], results['i_sectors'][:, 1]))
        hdr = ('const, c_err, c_hdi_l, c_hdi_r, slope, sl_err, sl_hdi_l, sl_hdi_r, sector_start, sector_end')
        file_name_lin = file_name.replace(ext, '_linear.csv')
        np.savetxt(file_name_lin, data, delimiter=',', header=hdr)
    # sinusoid model parameters
    if (not np.all(f_n == -1)) & (not np.all(a_n == -1)) & (not np.all(ph_n == -1)):
        data = np.column_stack((f_n, f_n_err, f_n_hdi[:, 0], f_n_hdi[:, 1],
                                a_n, a_n_err, a_n_hdi[:, 0], a_n_hdi[:, 1],
                                ph_n, ph_n_err, ph_n_hdi[:, 0], ph_n_hdi[:, 1],
                                passed_sigma, passed_snr, passed_b, passed_h))
        hdr = ('f_n, f_n_err, f_n_hdi_l, f_n_hdi_r, a_n, a_n_err, a_n_hdi_l, a_n_hdi_r, '
               'ph_n, ph_n_err, ph_n_hdi_l, ph_n_hdi_r, passed_sigma, passed_snr, passed_b, passed_h')
        file_name_sin = file_name.replace(ext, '_sinusoid.csv')
        np.savetxt(file_name_sin, data, delimiter=',', header=hdr)
    # statistics
    if not np.all(results['stats'] == -1):
        names = ('t_tot', 't_mean', 't_int', 'n_param', 'bic', 'noise_level')
        stats = (t_tot, t_mean, t_int, n_param, bic, noise_level)
        desc = ['Total time base of observations', 'Time reference (zero) point',
                'Integration time of observations', 'Number of free parameters',
                'Bayesian Information Criterion of the residuals', 'Standard deviation of the residuals']
        data = np.column_stack((names, stats, desc))
        description = 'Time series and model statistics'
        hdr = f'{target_id}, {data_id}, {description}\nname, value, description'
        file_name_stats = file_name.replace(ext, '_stats.csv')
        np.savetxt(file_name_stats, data, delimiter=',', header=hdr, fmt='%s')
    return None


def save_inference_data(file_name, inf_data):
    """Save the inference data object from Arviz/PyMC3
    
        Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.
    inf_data: object
        Arviz inference data object

    Returns
    -------
    None
    """
    if inf_data is None:
        return None
    
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_mc = file_name.replace(fn_ext, '_dists.nc4')
    inf_data.to_netcdf(file_name_mc)
    return None


def read_inference_data(file_name):
    """Read the inference data object from Arviz/PyMC3

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the results.

    Returns
    -------
    inf_data: object
        Arviz inference data object
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_mc = file_name.replace(fn_ext, '_dists.nc4')
    inf_data = az.from_netcdf(file_name_mc)
    return inf_data


def save_summary(target_id, save_dir, data_id='none'):
    """Create a summary file from the results of the analysis
    
    Parameters
    ----------
    target_id: int, str
        The TESS Input Catalog number for saving and loading.
        Use the name of the input light curve file if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    data_id: int, str
        Identification for the dataset used
    
    Returns
    -------
    None
    
    Notes
    -----
    Meant both as a quick overview of the results and to facilitate
    the compilation of a catalogue of a set of results
    """
    prew_par = -np.ones(7)
    timings_par = -np.ones(36)
    form_par = -np.ones(48)
    fit_par = -np.ones(39)
    freqs_par = -np.ones(5, dtype=int)
    level_par = -np.ones(12)
    t_tot, t_mean = 0, 0
    # read results
    if not save_dir.endswith(f'{target_id}_analysis'):
        save_dir = os.path.join(save_dir, f'{target_id}_analysis')  # add subdir
    # get period from last prewhitening step
    file_name_3 = os.path.join(save_dir, f'{target_id}_analysis_3.hdf5')
    file_name_5 = os.path.join(save_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name_5):
        results = read_parameters_hdf5(file_name_5, verbose=False)
        p_orb, _ = results['ephem']
        p_err, _ = results['ephem_err']
        p_hdi, _ = results['ephem_hdi']
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = results['stats']
        prew_par = [p_orb, p_err, p_hdi[0], p_hdi[1], n_param, bic, noise_level]
    elif os.path.isfile(file_name_3):
        results = read_parameters_hdf5(file_name_3, verbose=False)
        p_orb, _ = results['ephem']
        p_err, _ = results['ephem_err']
        p_hdi, _ = results['ephem_hdi']
        t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level = results['stats']
        prew_par = [p_orb, p_err, p_hdi[0], p_hdi[1], n_param, bic, noise_level]
    # todo: needs update
    # file header with all variable names
    hdr = ['id', 'stage', 't_tot', 't_mean', 'period', 'p_err', 'p_err_l', 'p_err_u',
           'n_param_prew', 'bic_prew', 'noise_level_prew',
           't_1', 't_2', 't_1_1', 't_1_2', 't_2_1', 't_2_2',
           't_b_1_1', 't_b_1_2', 't_b_2_1', 't_b_2_2', 'depth_1', 'depth_2',
           't_1_err', 't_2_err', 't_1_1_err', 't_1_2_err', 't_2_1_err', 't_2_2_err',
           't_b_1_1_err', 't_b_1_2_err', 't_b_2_1_err', 't_b_2_2_err', 'd_1_err', 'd_2_err',
           't_1_ind_err', 't_2_ind_err', 't_1_1_ind_err', 't_1_2_ind_err', 't_2_1_ind_err', 't_2_2_ind_err',
           't_b_1_1_ind_err', 't_b_1_2_ind_err', 't_b_2_1_ind_err', 't_b_2_2_ind_err', 'd_1_ind_err', 'd_2_ind_err',
           'ecosw_form', 'esinw_form', 'cosi_form', 'phi_0_form', 'log_rr_form', 'log_sb_form',
           'e_form', 'w_form', 'i_form', 'r_sum_form', 'r_rat_form', 'sb_rat_form',
           'ecosw_sig', 'esinw_sig', 'cosi_sig', 'phi_0_sig', 'log_rr_sig', 'log_sb_sig',
           'e_sig', 'w_sig', 'i_sig', 'r_sum_sig', 'r_rat_sig', 'sb_rat_sig',
           'ecosw_low', 'ecosw_upp', 'esinw_low', 'esinw_upp', 'cosi_low', 'cosi_upp',
           'phi_0_low', 'phi_0_upp', 'log_rr_low', 'log_rr_upp', 'log_sb_low', 'log_sb_upp',
           'e_low', 'e_upp', 'w_low', 'w_upp', 'i_low', 'i_upp',
           'r_sum_low', 'r_sum_upp', 'r_rat_low', 'r_rat_upp', 'sb_rat_low', 'sb_rat_upp',
           'ecosw_phys', 'esinw_phys', 'cosi_phys', 'phi_0_phys', 'log_rr_phys', 'log_sb_phys',
           'e_phys', 'w_phys', 'i_phys', 'r_sum_phys', 'r_rat_phys', 'sb_rat_phys',
           'ecosw_err_l', 'ecosw_err_u', 'esinw_err_l', 'esinw_err_u', 'cosi_err_l', 'cosi_err_u',
           'phi_0_err_l', 'phi_0_err_u', 'log_rr_err_l', 'log_rr_err_u', 'log_sb_err_l', 'log_sb_err_u',
           'e_err_l', 'e_err_u', 'w_err_l', 'w_err_u', 'i_err_l', 'i_err_u', 'r_sum_err_l', 'r_sum_err_u',
           'r_rat_err_l', 'r_rat_err_u', 'sb_rat_err_l', 'sb_rat_err_u',
           'n_param_phys', 'bic_phys', 'noise_level_phys',
           'total_freqs', 'passed_sigma', 'passed_snr', 'passed_both', 'passed_harmonics',
           'std_1', 'std_2', 'std_3', 'std_4', 'ratio_1_1', 'ratio_1_2', 'ratio_2_1', 'ratio_2_2',
           'ratio_3_1', 'ratio_3_2', 'ratio_4_1', 'ratio_4_2']
    # descriptions of all variables
    desc = ['target identifier', 'furthest stage the analysis reached', 'total time base of observations in days',
            'time series mean time reference point', 'orbital period in days', 'error in the orbital period',
            'lower HDI error estimate in period', 'upper HDI error estimate in period',
            'number of free parameters after the prewhitening phase', 'BIC after the prewhitening phase',
            'noise level after the prewhitening phase',
            'time of primary minimum with respect to the mean time',
            'time of secondary minimum with respect to the mean time',
            'time of primary first contact with respect to the mean time',
            'time of primary last contact with respect to the mean time',
            'time of secondary first contact with respect to the mean time',
            'time of secondary last contact with respect to the mean time',
            'start of (flat) eclipse bottom left of primary minimum',
            'end of (flat) eclipse bottom right of primary minimum',
            'start of (flat) eclipse bottom left of secondary minimum',
            'end of (flat) eclipse bottom right of secondary minimum',
            'depth of primary minimum', 'depth of secondary minimum',
            'error in time of primary minimum (t_1)', 'error in time of secondary minimum (t_2)',
            'error in time of primary first contact (t_1_1)', 'error in time of primary last contact (t_1_2)',
            'error in time of secondary first contact (t_2_1)', 'error in time of secondary last contact (t_2_2)',
            'error in start of (flat) eclipse bottom left of primary minimum',
            'error in end of (flat) eclipse bottom right of primary minimum',
            'error in start of (flat) eclipse bottom left of secondary minimum',
            'error in end of (flat) eclipse bottom right of secondary minimum',
            'error in depth of primary minimum', 'error in depth of secondary minimum',
            'individual error in time of primary minimum (t_1)', 'individual error in time of secondary minimum (t_2)',
            'individual error in time of primary first contact (t_1_1)',
            'individual error in time of primary last contact (t_1_2)',
            'individual error in time of secondary first contact (t_2_1)',
            'individual error in time of secondary last contact (t_2_2)',
            'individual error in start of (flat) eclipse bottom left of primary minimum',
            'individual error in end of (flat) eclipse bottom right of primary minimum',
            'individual error in start of (flat) eclipse bottom left of secondary minimum',
            'individual error in end of (flat) eclipse bottom right of secondary minimum',
            'individual error in depth of primary minimum', 'individual error in depth of secondary minimum',
            'e*cos(w) from timing formulae', 'e*sin(w) from timing formulae',
            'cosine of inclination from timing formulae', 'phi_0 angle (Kopal 1959) from timing formulae',
            'logarithm of the radius ratio r2/r1 from timing formulae',
            'logarithm of the surface brightness ratio sb2/sb1 from timing formulae',
            'eccentricity from timing formulae', 'argument of periastron (radians) from timing formulae',
            'inclination (radians) from timing formulae',
            'sum of radii divided by the semi-major axis of the relative orbit from timing formulae',
            'radius ratio r2/r1 from timing formulae', 'surface brightness ratio sb2/sb1 from timing formulae',
            'formal uncorrelated error in ecosw', 'formal uncorrelated error in esinw',
            'error estimate for cosi used for formal errors', 'formal uncorrelated error in phi_0',
            'scaled error formal estimate for log_rr', 'scaled formal error estimate for log_sb',
            'formal uncorrelated error in e', 'formal uncorrelated error in w',
            'error estimate for i used for formal errors', 'formal uncorrelated error in r_sum',
            'scaled error formal estimate for r_rat', 'scaled error formal estimate for sb_rat',
            'lower error estimate in ecosw', 'upper error estimate in ecosw',
            'lower error estimate in esinw', 'upper error estimate in esinw',
            'lower error estimate in cosi', 'upper error estimate in cosi',
            'lower error estimate in phi_0', 'upper error estimate in phi_0',
            'lower error estimate in log_rr', 'upper error estimate in log_rr',
            'lower error estimate in log_sb', 'upper error estimate in log_sb',
            'lower error estimate in e', 'upper error estimate in e',
            'lower error estimate in w', 'upper error estimate in w',
            'lower error estimate in i', 'upper error estimate in i',
            'lower error estimate in r_sum', 'upper error estimate in r_sum',
            'lower error estimate in r_rat', 'upper error estimate in r_rat',
            'lower error estimate in sb_rat', 'upper error estimate in sb_rat',
            'e cos(w) of the physical model', 'e sin(w) of the physical model',
            'cos(i) of the physical model', 'phi_0 of the physical model',
            'log of radius ratio of the physical model', 'log of surface brightness ratio of the physical model',
            'eccentricity of the physical model', 'argument of periastron of the physical model',
            'inclination (radians) of the physical model', 'sum of fractional radii of the physical model',
            'radius ratio of the physical model', 'surface brightness ratio of the physical model',
            'lower HDI error estimate in ecosw', 'upper HDI error estimate in ecosw',
            'lower HDI error estimate in esinw', 'upper HDI error estimate in esinw',
            'lower HDI error estimate in cosi', 'upper HDI error estimate in cosi',
            'lower HDI error estimate in phi_0', 'upper HDI error estimate in phi_0',
            'lower HDI error estimate in log_rr', 'upper HDI error estimate in log_rr',
            'lower HDI error estimate in log_sb', 'upper HDI error estimate in log_sb',
            'lower HDI error estimate in e', 'upper HDI error estimate in e',
            'lower HDI error estimate in w', 'upper HDI error estimate in w',
            'lower HDI error estimate in i', 'upper HDI error estimate in i',
            'lower HDI error estimate in r_sum', 'upper HDI error estimate in r_sum',
            'lower HDI error estimate in r_rat', 'upper HDI error estimate in r_rat',
            'lower HDI error estimate in sb_rat', 'upper HDI error estimate in sb_rat',
            'number of parameters after physical model optimisation',
            'BIC after physical model optimisation', 'noise level after physical model optimisation',
            'total number of frequencies', 'number of frequencies that passed the sigma test',
            'number of frequencies that passed the S/R test', 'number of frequencies that passed both tests',
            'number of harmonics that passed both tests',
            'Standard deviation of the residuals of the linear+sinusoid+eclipse model',
            'Standard deviation of the residuals of the linear+eclipse model',
            'Standard deviation of the residuals of the linear+harmonic 1 and 2+eclipse model',
            'Standard deviation of the residuals of the linear+non-harmonic sinusoid+eclipse model',
            'Ratio of the first eclipse depth to std_1', 'Ratio of the second eclipse depth to std_1',
            'Ratio of the first eclipse depth to std_2', 'Ratio of the second eclipse depth to std_2',
            'Ratio of the first eclipse depth to std_3', 'Ratio of the second eclipse depth to std_3',
            'Ratio of the first eclipse depth to std_4', 'Ratio of the second eclipse depth to std_4']
    # record the stage where the analysis finished
    stage = ''
    files_in_dir = []
    for root, dirs, files in os.walk(save_dir):
        for file_i in files:
            files_in_dir.append(os.path.join(root, file_i))
    for i in range(19, 0, -1):
        match_b = [fnmatch.fnmatch(file_i, f'*_analysis_{i}b*') for file_i in files_in_dir]
        if np.any(match_b):
            stage = str(i) + 'b'  # b variant
            break
        else:
            match_a = [fnmatch.fnmatch(file_i, f'*_analysis_{i}*') for file_i in files_in_dir]
            if np.any(match_a):
                stage = str(i)
                break
    stage = stage.rjust(3)  # make the string 3 long
    # compile all results
    obs_par = np.concatenate(([target_id], [stage], [t_tot], [t_mean], prew_par, timings_par, form_par, fit_par,
                              freqs_par, level_par)).reshape((-1, 1))
    data = np.column_stack((hdr, obs_par, desc))
    file_hdr = f'{target_id}, {data_id}\nname, value'  # the actual header used for numpy savetxt
    save_name = os.path.join(save_dir, f'{target_id}_analysis_summary.csv')
    np.savetxt(save_name, data, delimiter=',', fmt='%s', header=file_hdr)
    return None


def sequential_plotting(times, signal, signal_err, i_sectors, target_id, load_dir, save_dir=None, show=True):
    """Due to plotting not working under multiprocessing this function is
    made to make plots after running the analysis in parallel.
    
    Parameters
    ----------
    times: numpy.ndarray[float]
        Timestamps of the time series
    signal: numpy.ndarray[float]
        Measurement values of the time series
    signal_err: numpy.ndarray[float]
        Errors in the measurement values
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
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
        results = read_parameters_hdf5(file_name, verbose=False)
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = results['sin_mean']
        model_linear = tsf.linear_curve(times, const_1, slope_1, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_1, a_n_1, ph_n_1)
        model_1 = model_linear + model_sinusoid
    else:
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = np.array([[], [], [], [], []])
        model_1 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_2.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = results['sin_mean']
        model_linear = tsf.linear_curve(times, const_2, slope_2, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_2, a_n_2, ph_n_2)
        model_2 = model_linear + model_sinusoid
    else:
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = np.array([[], [], [], [], []])
        model_2 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_3.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = results['sin_mean']
        p_orb_3, _ = results['ephem']
        p_err_3, _ = results['ephem_err']
        model_linear = tsf.linear_curve(times, const_3, slope_3, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_3, a_n_3, ph_n_3)
        model_3 = model_linear + model_sinusoid
    else:
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = np.array([[], [], [], [], []])
        p_orb_3, p_err_3 = 0, 0
        model_3 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_4.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = results['sin_mean']
        model_linear = tsf.linear_curve(times, const_4, slope_4, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_4, a_n_4, ph_n_4)
        model_4 = model_linear + model_sinusoid
    else:
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = np.array([[], [], [], [], []])
        model_4 = np.zeros(len(times))
    file_name = os.path.join(load_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name):
        results = read_parameters_hdf5(file_name, verbose=False)
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = results['sin_mean']
        p_orb_5, _ = results['ephem']
        p_err_5, _ = results['ephem_err']
        t_tot, t_mean, t_mean_s, t_int, n_param_5, bic_5, noise_level_5 = results['stats']
        model_linear = tsf.linear_curve(times, const_5, slope_5, i_sectors)
        model_sinusoid = tsf.sum_sines(times, f_n_5, a_n_5, ph_n_5)
        model_5 = model_linear + model_sinusoid
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_5, f_tol=1e-9)
        f_h_5, a_h_5, ph_h_5 = f_n_5[harmonics], a_n_5[harmonics], ph_n_5[harmonics]
    else:
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = np.array([[], [], [], [], []])
        p_orb_5, p_err_5 = 0, 0
        n_param_5, bic_5, noise_level_5 = 0, 0, 0
        model_5 = np.zeros(len(times))
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
        vis.plot_pd_full_output(times, signal, signal_err, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_sectors,
                                save_file=file_name, show=show)
        if np.any([len(fs) != 0 for fs in f_n_i]):
            plot_nr = np.arange(1, len(f_n_i) + 1)[[len(fs) != 0 for fs in f_n_i]][-1]
            plot_data = [eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_sinusoids_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_sinusoids(times, signal, *plot_data, i_sectors, save_file=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_output_{plot_nr}.png')
            else:
                file_name = None
            plot_data = [p_orb_i[plot_nr - 1], p_err_i[plot_nr - 1]] + plot_data
            vis.plot_pd_single_output(times, signal, signal_err, *plot_data, i_sectors, annotate=False,
                                      save_file=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_harmonics_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_harmonics(times, signal, *plot_data, i_sectors, save_file=file_name, show=show)
    except NameError:
        pass  # some variable wasn't loaded (file did not exist)
    except ValueError:
        pass  # no frequencies?
    return None


def plot_all_from_file(file_name, i_sectors=None, load_dir=None, save_dir=None, show=True):
    """Plot all diagnostic plots of the results for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    i_sectors: numpy.ndarray[int]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
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
    times, signal, signal_err = np.loadtxt(file_name, usecols=(0, 1, 2), unpack=True)
    # if sectors not given, take full length
    if i_sectors is None:
        i_sectors = np.array([[0, len(times)]])  # no sector information
    # i_half_s = i_sectors  # in this case no differentiation between half or full sectors
    # do the plotting
    sequential_plotting(times, signal, signal_err, i_sectors, target_id, load_dir, save_dir=save_dir, show=show)
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
    time, flux, flux_err, i_chunks, medians = load_light_curve(all_files, apply_flags=True)
    # do the plotting
    sequential_plotting(time, flux, flux_err, i_chunks, tic, load_dir, save_dir=save_dir, show=show)
    return None
