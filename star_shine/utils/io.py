"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains io functions for reading and writing data and results.

Code written by: Luc IJspeert
"""

import os
import numpy as np
import numba as nb

import pandas as pd
import astropy.io.fits as fits
try:
    import arviz as az  # optional functionality
except ImportError:
    az = None
    pass

from star_shine.config.helpers import get_config

# load configuration
config = get_config()


@nb.njit(cache=True)
def normalise_counts(flux_counts, flux_counts_err, i_chunks):
    """Median-normalises flux (counts or otherwise, should be positive) by
    dividing by the median.

    Parameters
    ----------
    flux_counts: numpy.ndarray[Any, dtype[float]]
        Flux measurement values in counts of the time series.
    flux_counts_err: numpy.ndarray[Any, dtype[float]]
        Errors in the flux measurements.
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
            chunk_index = [[len(i_chunks), len(i_chunks) + len(ti) // 2],
                           [len(i_chunks) + len(ti) // 2, len(i_chunks) + len(ti)]]
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
