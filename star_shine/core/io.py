"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains io functions for reading and writing data and results.

Code written by: Luc IJspeert
"""

import os
import h5py
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


def load_data_hdf5(file_name, h5py_file_kwargs=None):
    """Load data from an hdf5 file and return it in a dictionary.

    Primarily for the api class Data.

    Parameters
    ----------
    file_name: str
        File name (including path) for loading the data.
    h5py_file_kwargs: dict, optional
        Keyword arguments for opening the h5py file.
        Example: {'locking': False}, for a drive that does not support locking.

    Returns
    -------
    Dict
        Dictionary of the data attributes and fields
    """
    # to avoid dict in function defaults
    if h5py_file_kwargs is None:
        h5py_file_kwargs = {}

    # add everything to a dict
    data_dict = {}

    # load the results from the file
    with h5py.File(file_name, 'r', **h5py_file_kwargs) as file:
        # file description
        data_dict['target_id'] = file.attrs['target_id']
        data_dict['data_id'] = file.attrs['data_id']
        data_dict['date_time'] = file.attrs['date_time']

        # original list of files
        data_dict['data_dir'] = file.attrs['data_dir']
        data_dict['file_list'] = np.copy(file['file_list'])

        # summary statistics
        data_dict['t_tot'] = file.attrs['t_tot']
        data_dict['t_mean'] = file.attrs['t_mean']
        data_dict['t_step'] = file.attrs['t_step']

        # the time series data
        data_dict['time'] = np.copy(file['time'])
        data_dict['flux'] = np.copy(file['flux'])
        data_dict['flux_err'] = np.copy(file['flux_err'])

        # additional information
        data_dict['i_chunks'] = np.copy(file['i_chunks'])
        data_dict['flux_counts_medians'] = np.copy(file['flux_counts_medians'])
        data_dict['t_mean_chunk'] = np.copy(file['t_mean_chunk'])

    return data_dict


def save_data_hdf5(file_name, data_dict):
    """Save data to an hdf5 file.

    Primarily for the api class Data.

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the data.
    data_dict: dict
        Dictionary of the data attributes and fields

    Returns
    -------
    None
    """
    # file name must have hdf5 extension
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if ext != '.hdf5':
        file_name = file_name.replace(ext, '.hdf5')

    # save to hdf5
    with h5py.File(file_name, 'w') as file:
        file.attrs['target_id'] = data_dict['target_id']
        file.attrs['data_id'] = data_dict['data_id']
        file.attrs['date_time'] = data_dict['date_time']

        # original list of files
        file.attrs['data_dir'] = data_dict['data_dir']  # original data directory
        file.create_dataset('file_list', data=data_dict['file_list'])
        file['file_list'].attrs['description'] = 'original list of files for the creation of this data file'

        # summary statistics
        file.attrs['t_tot'] = data_dict['t_tot']  # Total time base of observations
        file.attrs['t_mean'] = data_dict['t_mean']  # Time reference (zero) point of the full light curve
        file.attrs['t_step'] = data_dict['t_step']  # Median time step of observations
        file.create_dataset('t_mean_chunk', data=data_dict['t_mean_chunk'])
        file['t_mean_chunk'].attrs['unit'] = 'time unit of the data (often days)'
        file['t_mean_chunk'].attrs['description'] = 'time reference (zero) point of the each time chunk'

        # the time series data
        file.create_dataset('time', data=data_dict['time'])
        file['time'].attrs['unit'] = 'time unit of the data (often days)'
        file['time'].attrs['description'] = 'timestamps of the observations'
        file.create_dataset('flux', data=data_dict['flux'])
        file['flux'].attrs['unit'] = 'median normalised flux'
        file['flux'].attrs['description'] = 'normalised flux measurements of the observations'
        file.create_dataset('flux_err', data=data_dict['flux_err'])
        file['flux_err'].attrs['unit'] = 'median normalised flux'
        file['flux_err'].attrs['description'] = 'normalised error measurements in the flux'

        # additional information
        file.create_dataset('i_chunks', data=data_dict['i_chunks'])
        file['i_chunks'].attrs['description'] = 'pairs of indices indicating time chunks of the data'
        file.create_dataset('flux_counts_medians', data=data_dict['flux_counts_medians'])
        file['flux_counts_medians'].attrs['unit'] = 'raw flux counts'
        file['flux_counts_medians'].attrs['description'] = 'median flux level per time chunk'

    return None


def load_result_hdf5(file_name, h5py_file_kwargs=None):
    """Load results from an hdf5 file and return it in a dictionary.

    Primarily for the api class Result.

    Parameters
    ----------
    file_name: str
        File name (including path) for loading the results.
    h5py_file_kwargs: dict, optional
        Keyword arguments for opening the h5py file.
        Example: {'locking': False}, for a drive that does not support locking.

    Returns
    -------
    Dict
        Dictionary of the result attributes and fields
    """
    # to avoid dict in function defaults
    if h5py_file_kwargs is None:
        h5py_file_kwargs = {}

    # add everything to a dict
    result_dict = {}

    # load the results from the file
    with h5py.File(file_name, 'r', **h5py_file_kwargs) as file:
        # file description
        result_dict['target_id'] = file.attrs['target_id']
        result_dict['data_id'] = file.attrs['data_id']
        result_dict['date_time'] = file.attrs['date_time']

        # summary statistics
        result_dict['n_param'] = file.attrs['n_param']
        result_dict['bic'] = file.attrs['bic']
        result_dict['noise_level'] = file.attrs['noise_level']

        # linear model parameters
        # y-intercepts
        result_dict['const'] = np.copy(file['const'])
        result_dict['const_err'] = np.copy(file['const_err'])
        result_dict['const_hdi'] = np.copy(file['const_hdi'])
        # slopes
        result_dict['slope'] = np.copy(file['slope'])
        result_dict['slope_err'] = np.copy(file['slope_err'])
        result_dict['slope_hdi'] = np.copy(file['slope_hdi'])

        # sinusoid model parameters
        # frequencies
        result_dict['f_n'] = np.copy(file['f_n'])
        result_dict['f_n_err'] = np.copy(file['f_n_err'])
        result_dict['f_n_hdi'] = np.copy(file['f_n_hdi'])
        # amplitudes
        result_dict['a_n'] = np.copy(file['a_n'])
        result_dict['a_n_err'] = np.copy(file['a_n_err'])
        result_dict['a_n_hdi'] = np.copy(file['a_n_hdi'])
        # phases
        result_dict['ph_n'] = np.copy(file['ph_n'])
        result_dict['ph_n_err'] = np.copy(file['ph_n_err'])
        result_dict['ph_n_hdi'] = np.copy(file['ph_n_hdi'])
        # passing criteria
        result_dict['passing_sigma'] = np.copy(file['passing_sigma'])
        result_dict['passing_snr'] = np.copy(file['passing_snr'])

        # harmonic model
        result_dict['h_base'] = np.copy(file['h_base'])
        result_dict['h_mult'] = np.copy(file['h_mult'])
        result_dict['f_h_err'] = np.copy(file['f_h_err'])
        # passing criteria
        result_dict['passing_harmonic'] = np.copy(file['passing_harmonic'])

    return result_dict


def save_result_hdf5(file_name, result_dict):
    """Save results to an hdf5 file.

    Primarily for the api class Result.

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the result.
    result_dict: dict
        Dictionary of the result attributes and fields

    Returns
    -------
    None
    """
    # file name must have hdf5 extension
    ext = os.path.splitext(os.path.basename(file_name))[1]
    if ext != '.hdf5':
        file_name = file_name.replace(ext, '.hdf5')

    # save to hdf5
    with h5py.File(file_name, 'w') as file:
        file.attrs['target_id'] = result_dict['target_id']
        file.attrs['data_id'] = result_dict['data_id']
        file.attrs['date_time'] = result_dict['date_time']
        file.attrs['n_param'] = result_dict['n_param']  # number of free parameters
        file.attrs['bic'] = result_dict['bic']  # Bayesian Information Criterion of the residuals
        file.attrs['noise_level'] = result_dict['noise_level']  # standard deviation of the residuals

        # the linear model
        # y-intercepts
        file.create_dataset('const', data=result_dict['const'])
        file['const'].attrs['unit'] = 'median normalised flux'
        file['const'].attrs['description'] = 'y-intercept per analysed sector'
        file.create_dataset('const_err', data=result_dict['const_err'])
        file['const_err'].attrs['unit'] = 'median normalised flux'
        file['const_err'].attrs['description'] = 'errors in the y-intercept per analysed sector'
        file.create_dataset('const_hdi', data=result_dict['const_hdi'])
        file['const_hdi'].attrs['unit'] = 'median normalised flux'
        file['const_hdi'].attrs['description'] = 'HDI for the y-intercept per analysed sector'

        # slopes
        file.create_dataset('slope', data=result_dict['slope'])
        file['slope'].attrs['unit'] = 'median normalised flux / d'
        file['slope'].attrs['description'] = 'slope per analysed sector'
        file.create_dataset('slope_err', data=result_dict['slope_err'])
        file['slope_err'].attrs['unit'] = 'median normalised flux / d'
        file['slope_err'].attrs['description'] = 'error in the slope per analysed sector'
        file.create_dataset('slope_hdi', data=result_dict['slope_hdi'])
        file['slope_hdi'].attrs['unit'] = 'median normalised flux / d'
        file['slope_hdi'].attrs['description'] = 'HDI for the slope per analysed sector'

        # the sinusoid model
        # frequencies
        file.create_dataset('f_n', data=result_dict['f_n'])
        file['f_n'].attrs['unit'] = '1 / d'
        file['f_n'].attrs['description'] = 'frequencies of a number of sinusoids'
        file.create_dataset('f_n_err', data=result_dict['f_n_err'])
        file['f_n_err'].attrs['unit'] = '1 / d'
        file['f_n_err'].attrs['description'] = 'errors in the frequencies of a number of sinusoids'
        file.create_dataset('f_n_hdi', data=result_dict['f_n_hdi'])
        file['f_n_hdi'].attrs['unit'] = '1 / d'
        file['f_n_hdi'].attrs['description'] = 'HDI for the frequencies of a number of sinusoids'

        # amplitudes
        file.create_dataset('a_n', data=result_dict['a_n'])
        file['a_n'].attrs['unit'] = 'median normalised flux'
        file['a_n'].attrs['description'] = 'amplitudes of a number of sinusoids'
        file.create_dataset('a_n_err', data=result_dict['a_n_err'])
        file['a_n_err'].attrs['unit'] = 'median normalised flux'
        file['a_n_err'].attrs['description'] = 'errors in the amplitudes of a number of sinusoids'
        file.create_dataset('a_n_hdi', data=result_dict['a_n_hdi'])
        file['a_n_hdi'].attrs['unit'] = 'median normalised flux'
        file['a_n_hdi'].attrs['description'] = 'HDI for the amplitudes of a number of sinusoids'

        # phases
        file.create_dataset('ph_n', data=result_dict['ph_n'])
        file['ph_n'].attrs['unit'] = 'radians'
        file['ph_n'].attrs['description'] = 'phases of a number of sinusoids, with reference point t_mean'
        file.create_dataset('ph_n_err', data=result_dict['ph_n_err'])
        file['ph_n_err'].attrs['unit'] = 'radians'
        file['ph_n_err'].attrs['description'] = 'errors in the phases of a number of sinusoids'
        file.create_dataset('ph_n_hdi', data=result_dict['ph_n_hdi'])
        file['ph_n_hdi'].attrs['unit'] = 'radians'
        file['ph_n_hdi'].attrs['description'] = 'HDI for the phases of a number of sinusoids'

        # sinusoid selection criteria
        file.create_dataset('passing_sigma', data=result_dict['passing_sigma'])
        file['passing_sigma'].attrs['description'] = 'sinusoids passing the sigma criterion'
        file.create_dataset('passing_snr', data=result_dict['passing_snr'])
        file['passing_snr'].attrs['description'] = 'sinusoids passing the signal to noise criterion'

        # harmonic model
        file.create_dataset('h_base', data=result_dict['h_base'])
        file['h_base'].attrs['description'] = 'index of the base harmonic frequency'
        file.create_dataset('h_mult', data=result_dict['h_mult'])
        file['h_mult'].attrs['description'] = 'multiplier of the base harmonic frequency'
        file.create_dataset('f_h_err', data=result_dict['f_h_err'])
        file['f_h_err'].attrs['description'] = 'errors in the harmonic frequencies'

        # harmonic selection criteria
        file.create_dataset('passing_harmonic', data=result_dict['passing_harmonic'])
        file['passing_harmonic'].attrs['description'] = 'harmonic sinusoids passing the sigma criterion'

    return None


def save_result_csv(file_name, result_dict):
    """Save results to several csv files.

    Primarily for the api class Result.

    Parameters
    ----------
    file_name: str
        File name (including path) for saving the result.
    result_dict: dict
        Dictionary of the result attributes and fields

    Returns
    -------
    None
    """
    # file extension
    ext = os.path.splitext(os.path.basename(file_name))[1]

    # linear model parameters
    data = np.column_stack((result_dict['const'], result_dict['const_err'],
                            result_dict['const_hdi'][:, 0], result_dict['const_hdi'][:, 1],
                            result_dict['slope'], result_dict['slope_err'],
                            result_dict['slope_hdi'][:, 0], result_dict['slope_hdi'][:, 1]))

    hdr = 'const, const_err, const_hdi_l, const_hdi_r, slope, slope_err, slope_hdi_l, slope_hdi_r'
    file_name_lin = file_name.replace(ext, '_linear.csv')
    np.savetxt(file_name_lin, data, delimiter=',', header=hdr)

    # sinusoid model parameters
    data = np.column_stack((result_dict['f_n'], result_dict['f_n_err'],
                            result_dict['f_n_hdi'][:, 0], result_dict['f_n_hdi'][:, 1],
                            result_dict['a_n'], result_dict['a_n_err'],
                            result_dict['a_n_hdi'][:, 0], result_dict['a_n_hdi'][:, 1],
                            result_dict['ph_n'], result_dict['ph_n_err'],
                            result_dict['ph_n_hdi'][:, 0], result_dict['ph_n_hdi'][:, 1],
                            result_dict['passing_sigma'], result_dict['passing_snr'],
                            result_dict['h_base'], result_dict['h_mult'],
                            result_dict['f_h_err'], result_dict['passing_harmonic']
                            ))

    hdr = ('f_n, f_n_err, f_n_hdi_l, f_n_hdi_r, a_n, a_n_err, a_n_hdi_l, a_n_hdi_r, '
           'ph_n, ph_n_err, ph_n_hdi_l, ph_n_hdi_r, passing_sigma, passing_snr, '
           'h_base, h_mult, f_h_err, passing_h')
    file_name_sin = file_name.replace(ext, '_sinusoid.csv')
    np.savetxt(file_name_sin, data, delimiter=',', header=hdr)

    # period and statistics
    names = ('n_param', 'bic', 'noise_level')
    stats = (result_dict['n_param'], result_dict['bic'], result_dict['noise_level'])

    desc = ['Number of free parameters', 'Bayesian Information Criterion of the residuals',
            'Standard deviation of the residuals']
    data = np.column_stack((names, stats, desc))
    hdr = f"{result_dict['target_id']}, {result_dict['data_id']}, Model statistics\nname, value, description"
    file_name_stats = file_name.replace(ext, '_stats.csv')
    np.savetxt(file_name_stats, data, delimiter=',', header=hdr, fmt='%s')

    return None


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
    i_chunks = np.zeros((0, 2), dtype=int)

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
        chunk_index = [[len(i_chunks), len(i_chunks) + len(ti)]]
        if config.halve_chunks:
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
    time = time[finite].astype(np.float64)
    flux = flux[finite].astype(np.float64)
    flux_err = flux_err[finite].astype(np.float64)

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
