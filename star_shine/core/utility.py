"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains utility functions for data processing, unit conversions
and loading in data (some functions specific to TESS data).

Code written by: Luc IJspeert
"""

import os
import logging
import numpy as np
import numba as nb

try:
    import pandas as pd  # optional functionality
except ImportError:
    pd = None
    pass
import astropy.io.fits as fits
try:
    import arviz as az  # optional functionality
except ImportError:
    az = None
    pass

from . import visualisation as vis, analysis_functions as af, timeseries_functions as tsf
from .config import get_config


# load configuration
config = get_config()


def get_custom_logger(save_dir, target_id, verbose):
    """Create a custom logger for logging to file and to stdout

    Parameters
    ----------
    save_dir: str
        folder to save the log file
    target_id: str
        Identifier to use for the log file
    verbose: bool
        If set to True, information will be printed by the logger

    Returns
    -------
     : None
    """
    # customize the logger
    logger = logging.getLogger(__name__)  # make an instance of the logging library
    logger.setLevel(logging.INFO)  # set base activation level for logger

    # make formatters for the handlers
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')

    # remove existing handlers to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    # make stream handler
    if verbose:
        s_handler = logging.StreamHandler()  # for printing
        s_handler.setLevel(logging.INFO)  # print everything with level 20 or above
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    # file handler
    logname = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}.log')
    f_handler = logging.FileHandler(logname, mode='a')  # for saving
    f_handler.setLevel(logging.INFO)  # save everything with level 20 or above
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return None


def update_config(file_name='', settings=None):
    """Update the configuration using a file and/or a dictionary.

    First loads the file, then updates settings, so both could be used simultaneously.
    This alters the state of the current configuration, not the configuration file.

    Parameters
    ----------
    file_name: str, optional
        Path to the yaml configuration file.
    settings: dict, optional
        Dictionary to update specific configuration settings.

    Returns
    -------
    None
    """
    # load from file
    if file_name != '':
        config.update_from_file(file_name)

    # update individual settings
    if settings is not None:
        config.update_from_dict(settings)

    return None


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
    str
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
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the mean over
    w: numpy.ndarray[Any, dtype[float]]
        Weights corresponding to each value
    
    Returns
    -------
    float
        Mean of x weighted by w
    """
    w_mean = np.sum(x * w) / np.sum(w)

    return w_mean


@nb.njit(cache=True)
def std_unb(x, n):
    """Unbiased standard deviation

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the std over
    n: int
        Number of degrees of freedom

    Returns
    -------
    float
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
    int
        Number of decimal places to round to
    """
    if x != 0:
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
    float
        signal-to-noise threshold for this data set
    
    Notes
    -----
    Based on Baran & Koen 2021, eq 6.
    (https://ui.adsabs.harvard.edu/abs/2021AcA....71..113B/abstract)
    """
    # equation 6 from Baran & Koen 2021
    sn_thr = 1.201 * np.sqrt(1.05 * np.log(n_points) + 7.184)

    # round to two decimals
    sn_thr = np.round(sn_thr, 2)

    return sn_thr


@nb.njit(cache=True)
def normalise_counts(flux_counts, flux_counts_err, i_chunks):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median.
    
    Parameters
    ----------
    flux_counts: numpy.ndarray[Any, dtype[float]]
        Flux measurement values in counts of the time series
    flux_counts_err: numpy.ndarray[Any, dtype[float]]
        Errors in the flux measurements
    i_chunks: numpy.ndarray[Any, dtype[float]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    tuple:
        flux_norm: numpy.ndarray[Any, dtype[float]]
            Normalised flux measurements
        flux_err_norm: numpy.ndarray[Any, dtype[float]]
            Normalised flux errors (zeros if flux_counts_err is None)
        medians: numpy.ndarray[Any, dtype[float]]
            Median flux counts per chunk
    
    Notes
    -----
    The result is positive and varies around one.
    The flux is processed per chunk.
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
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        time_sorter: numpy.ndarray[Any, dtype[int]]
            Sort indices for the full array
        i_chunks: numpy.ndarray[Any, dtype[int]]
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
    tuple
        A tuple containing the following elements:
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values
    """
    # get the right columns with pandas
    col_names = [config.cn_time, config.cn_flux, config.cn_flux_err]
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
    tuple
        A tuple containing the following elements:
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values
        qual_flags: numpy.ndarray[Any, dtype[int]]
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
        time = hdul[1].data[config.cf_time]
        flux = hdul[1].data[config.cf_flux]
        flux_err = hdul[1].data[config.cf_flux_err]

        # quality flags
        qual_flags = hdul[1].data[config.cf_quality]

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
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        medians: numpy.ndarray[Any, dtype[float]]
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
        if config.halve_chunks & (file.endswith('.fits') | file.endswith('.fit')):
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
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    g_min: int
        Minimum group size
    g_max: int
        Maximum group size (g_max > g_min)
    
    Returns
    -------
    list[numpy.ndarray[Any, dtype[int]]]
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
    while len(not_used) > 0:
        if len(not_used) > g_min + 1:
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
def correct_for_crowdsap(flux, crowdsap, i_chunks):
    """Correct the flux for flux contribution of a third source
    
    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
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
    cor_flux = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        cor_flux[s[0]:s[1]] = (flux[s[0]:s[1]] - 1 + crowd) / crowd

    return cor_flux


@nb.njit(cache=True)
def model_crowdsap(flux, crowdsap, i_chunks):
    """Incorporate flux contribution of a third source into the flux

    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model of the flux incorporating light contamination

    Notes
    -----
    Does the opposite as correct_for_crowdsap, to be able to model the effect of
    third light to some degree (can only achieve an upper bound on CROWDSAP).
    """
    model = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        model[s[0]:s[1]] = flux[s[0]:s[1]] * crowd + 1 - crowd

    return model


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
    object
        Arviz inference data object
    """
    fn_ext = os.path.splitext(os.path.basename(file_name))[1]
    file_name_mc = file_name.replace(fn_ext, '_dists.nc4')
    inf_data = az.from_netcdf(file_name_mc)

    return inf_data


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
        results = read_result_hdf5(file_name, verbose=False)
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = results['sin_mean']
        model_linear = tsf.linear_curve(time, const_1, slope_1, i_chunks)
        model_sinusoid = tsf.sum_sines(time, f_n_1, a_n_1, ph_n_1)
        model_1 = model_linear + model_sinusoid
    else:
        const_1, slope_1, f_n_1, a_n_1, ph_n_1 = np.array([[], [], [], [], []])
        model_1 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_2.hdf5')
    if os.path.isfile(file_name):
        results = read_result_hdf5(file_name, verbose=False)
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = results['sin_mean']
        model_linear = tsf.linear_curve(time, const_2, slope_2, i_chunks)
        model_sinusoid = tsf.sum_sines(time, f_n_2, a_n_2, ph_n_2)
        model_2 = model_linear + model_sinusoid
    else:
        const_2, slope_2, f_n_2, a_n_2, ph_n_2 = np.array([[], [], [], [], []])
        model_2 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_3.hdf5')
    if os.path.isfile(file_name):
        results = read_result_hdf5(file_name, verbose=False)
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = results['sin_mean']
        p_orb_3, _ = results['ephem']
        p_err_3, _ = results['ephem_err']
        model_linear = tsf.linear_curve(time, const_3, slope_3, i_chunks)
        model_sinusoid = tsf.sum_sines(time, f_n_3, a_n_3, ph_n_3)
        model_3 = model_linear + model_sinusoid
    else:
        const_3, slope_3, f_n_3, a_n_3, ph_n_3 = np.array([[], [], [], [], []])
        p_orb_3, p_err_3 = 0, 0
        model_3 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_4.hdf5')
    if os.path.isfile(file_name):
        results = read_result_hdf5(file_name, verbose=False)
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = results['sin_mean']
        model_linear = tsf.linear_curve(time, const_4, slope_4, i_chunks)
        model_sinusoid = tsf.sum_sines(time, f_n_4, a_n_4, ph_n_4)
        model_4 = model_linear + model_sinusoid
    else:
        const_4, slope_4, f_n_4, a_n_4, ph_n_4 = np.array([[], [], [], [], []])
        model_4 = np.zeros(len(time))

    file_name = os.path.join(load_dir, f'{target_id}_analysis_5.hdf5')
    if os.path.isfile(file_name):
        results = read_result_hdf5(file_name, verbose=False)
        const_5, slope_5, f_n_5, a_n_5, ph_n_5 = results['sin_mean']
        p_orb_5, _ = results['ephem']
        p_err_5, _ = results['ephem_err']
        t_tot, t_mean, t_mean_s, t_int, n_param_5, bic_5, noise_level_5 = results['stats']
        model_linear = tsf.linear_curve(time, const_5, slope_5, i_chunks)
        model_sinusoid = tsf.sum_sines(time, f_n_5, a_n_5, ph_n_5)
        model_5 = model_linear + model_sinusoid
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_5, p_orb_5, f_tol=1e-9)
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
        vis.plot_pd_full_output(time, flux, flux_err, models, p_orb_i, p_err_i, f_n_i, a_n_i, i_chunks,
                                file_name=file_name, show=show)
        if np.any([len(fs) != 0 for fs in f_n_i]):
            plot_nr = np.arange(1, len(f_n_i) + 1)[[len(fs) != 0 for fs in f_n_i]][-1]
            plot_data = [eval(f'const_{plot_nr}'), eval(f'slope_{plot_nr}'),
                         eval(f'f_n_{plot_nr}'), eval(f'a_n_{plot_nr}'), eval(f'ph_n_{plot_nr}')]
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_sinusoids_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_sinusoids(time, flux, *plot_data, i_chunks, file_name=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_pd_output_{plot_nr}.png')
            else:
                file_name = None
            plot_data = [p_orb_i[plot_nr - 1], p_err_i[plot_nr - 1]] + plot_data
            vis.plot_pd_single_output(time, flux, flux_err, *plot_data, i_chunks, annotate=False,
                                      file_name=file_name, show=show)
            if save_dir is not None:
                file_name = os.path.join(save_dir, f'{target_id}_frequency_analysis_lc_harmonics_{plot_nr}.png')
            else:
                file_name = None
            vis.plot_lc_harmonics(time, flux, *plot_data, i_chunks, file_name=file_name, show=show)
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
    time, flux, flux_err, i_chunks, medians = load_light_curve(all_files, apply_flags=True)

    # do the plotting
    sequential_plotting(time, flux, flux_err, i_chunks, tic, load_dir, save_dir=save_dir, show=show)

    return None
