"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the main functions that link together all functionality.

Code written by: Luc IJspeert
"""
import os
import time as systime
import datetime
import logging
import h5py
import numpy as np
import functools as fct
import multiprocessing as mp

from . import timeseries_functions as tsf
from . import timeseries_fitting as tsfit
from . import mcmc_functions as mcf
from . import analysis_functions as af
from . import utility as ut
from . import visualisation as vis
from .. import config


# initialize logger
logger = logging.getLogger(__name__)


class Data:
    """A class to handle light curve data.

    Attributes
    ----------
    file_list: list[str]
        List of ascii light curve files or (TESS) data product '.fits' files.
    data_dir: str, None
        Root directory where the data files are stored.
    target_id: int
        User defined identification integer for the target under investigation.
    data_id: str
        User defined identification for the dataset used.
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values.
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    flux_counts_medians: numpy.ndarray[Any, dtype[float]]
        Median flux counts per chunk.
    t_tot: float
        Total time base of observations.
    t_mean: float
        Time reference (zero) point of the full light curve.
    t_mean_chunk: numpy.ndarray[Any, dtype[float]]
        Time reference (zero) point per chunk.
    t_int: float
        Integration time of observations (taken to be the median time step by default, may be changed).
    """

    def __init__(self, file_list, data_dir='', target_id='', data_id=''):
        """Initialises the Data object.

        The data is loaded from the given file(s) and some basic processing is done.
        Either a file name, or target id plus file list must be given.

        Parameters
        ----------
        file_list: list[str]
            List of ascii light curve files or (TESS) data product '.fits' files. Exclude the path given to 'data_dir'.
            If only one file is given, its file name is used for target_id (if 'none').
        data_dir: str, optional
            Root directory where the data files are stored. Added to the file name. If empty, it is loaded from config.
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        """
        # initialise attributes before they are assigned values
        self.time = np.zeros((0,), dtype=np.float_)
        self.flux = np.zeros((0,), dtype=np.float_)
        self.flux_err = np.zeros((0,), dtype=np.float_)
        self.i_chunks = np.zeros((0, 2), dtype=np.int_)
        self.flux_counts_medians = np.zeros((0,), dtype=np.float_)
        self.t_tot = 0.
        self.t_mean = 0.
        self.t_mean_chunk = np.zeros((0,), dtype=np.float_)
        self.t_int = 0.

        # set the file list and data directory
        self.file_list = file_list
        if data_dir == '':
            data_dir = config.DATA_DIR
        self.data_dir = data_dir

        # set IDs
        self.target_id = target_id
        self.data_id = data_id

        # guard against empty list
        if len(file_list) == 0:
            logger.info(f"Empty file list provided.")
            return

        # Check if the file(s) exist(s)
        self._check_file_existence()
        if len(self.file_list) == 0:
            logger.info("No existing files in file list")
            return

        # set the target ID
        if self.target_id == '':
            self.target_id = os.path.splitext(os.path.basename(file_list[0]))[0]  # file name is used as identifier

        # load and process the data
        self._load_data()

        return

    def _check_file_existence(self):
        """Checks whether the given file(s) exist.

        Removes missing files from the file list

        Returns
        -------
        None
        """
        # check for missing files in the list
        missing = []
        for i, file in enumerate(self.file_list):
            if not os.path.exists(os.path.join(self.data_dir, file)):
                missing.append(i)

        # log a message if files are missing
        if len(missing) > 0:
            missing_files = [self.file_list[i] for i in missing]

            # add directory to message
            dir_text = ""
            if self.data_dir is not None:
                dir_text = f" in directory {self.data_dir}"
            message = f"Missing files {missing_files}{dir_text}, removing from list."

            logger.info(message)

            # remove the files
            for i in missing:
                del self.file_list[i]
        return None

    def _load_data(self):
        """Load light curve data from the file list.

        Returns
        -------
        None
        """
        # add data_dir if not None
        if self.data_dir is None:
            file_list_dir = self.file_list
        else:
            file_list_dir = [os.path.join(self.data_dir, file) for file in self.file_list]

        # load the data from the list of files
        lc_data = ut.load_light_curve(file_list_dir, apply_flags=config.APPLY_Q_FLAGS)
        time, flux, flux_err, i_chunks, medians = lc_data

        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.i_chunks = i_chunks
        self.flux_counts_medians = medians

        # set derived attributes
        self.t_tot = np.ptp(self.time)
        self.t_mean = np.mean(self.time)
        self.t_mean_chunk = np.array([np.mean(self.time[ch[0]:ch[1]]) for ch in self.i_chunks])
        self.t_int = np.median(np.diff(self.time))  # integration time, taken to be the median time step

        # check for overlapping time stamps
        if np.any(np.diff(self.time) <= 0):
            logger.info("The time array chunks include overlap.")

        return None

    def load(self):
        """Load a data file."""

    def save(self):
        """Save a data file."""

    def plot_light_curve(self, file_name=None, show=True):
        """Plot the light curve data.

        Parameters
        ----------
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot

        Returns
        -------
        None
        """
        vis.plot_lc(self.time, self.flux, self.flux_err, self.i_chunks, file_name=file_name, show=show)
        return None


class Result:
    """A class to handle analysis results.

    Attributes
    ----------
    file_name: str
        File name for the result in question, for saving and loading.
    target_id: str, optional
        User defined identification number or name for the target under investigation.
    data_id: str, optional
        User defined identification name for the dataset used.
    description: str, optional
        User defined description of the result in question.
    n_param: int
        Number of free parameters in the model
    bic: float
        Bayesian Information Criterion of the residuals
    noise_level: float
        The noise level (standard deviation of the residuals)
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
        Uncertainty in the constant for each sector
    sl_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slope for each sector
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sine wave
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitude for each sine wave (these are identical)
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phase for each sine wave
    c_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the constant for each sector
    sl_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the slope for each sector
    f_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the frequency for each sine wave
    a_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the amplitude for each sine wave (these are identical)
    ph_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the phase for each sine wave
    passed_sigma: numpy.ndarray[bool]
        Sinusoids that passed the sigma check
    passed_snr: numpy.ndarray[bool]
        Sinusoids that passed the signal-to-noise check
    passed_both: numpy.ndarray[bool]
        Sinusoids that passed both checks
    passed_harmonic: numpy.ndarray[bool]
        Harmonic sinusoids that passed
    """

    def __init__(self):
        """Initialises the Result object."""
        self.file_name = 'none'

        # descriptive
        self.target_id = 'none'
        self.data_id = 'none'
        self.description = 'none'

        # summary statistics
        self.n_param = -1
        self.bic = -1
        self.noise_level = -1

        # linear model parameters
        # y-intercepts
        self.const = np.zeros(0)
        self.c_err = np.zeros(0)
        self.c_hdi = np.zeros((0, 2))
        # slopes
        self.slope = np.zeros(0)
        self.sl_err = np.zeros(0)
        self.sl_hdi = np.zeros((0, 2))

        # sinusoid model parameters
        # frequencies
        self.f_n = np.zeros(0)
        self.f_n_err = np.zeros(0)
        self.f_n_hdi = np.zeros((0, 2))
        # amplitudes
        self.a_n = np.zeros(0)
        self.a_n_err = np.zeros(0)
        self.a_n_hdi = np.zeros((0, 2))
        # phases
        self.ph_n = np.zeros(0)
        self.ph_n_err = np.zeros(0)
        self.ph_n_hdi = np.zeros((0, 2))
        # passing criteria
        self.passed_sigma = np.zeros(0, dtype=bool)
        self.passed_snr = np.zeros(0, dtype=bool)
        self.passed_both = np.zeros(0, dtype=bool)

        # harmonic model
        self.p_orb = np.zeros(4)
        self.passed_harmonic = np.zeros(0, dtype=bool)

        return

    def setter(self, **kwargs):
        """Fill in the attributes with results.

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly

        Returns
        -------
        None
        """
        # file name must have hdf5 extension
        if 'file_name' in kwargs.keys():
            ext = os.path.splitext(os.path.basename(kwargs['file_name']))[1]
            if ext != '.hdf5':
                kwargs['file_name'] = kwargs['file_name'].replace(ext, '.hdf5')

        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    @classmethod
    def load(cls, file_name, h5py_file_kwargs=None):
        """Load a result file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.

        Returns
        -------
        instance: Result object
            Instance of the Result class with the loaded results.
        """
        # to avoid dict in function defaults
        if h5py_file_kwargs is None:
            h5py_file_kwargs = {}

        # add everything to a dict
        result_dict = {'file_name': file_name}

        # load the results from the file
        with h5py.File(file_name, 'r', **h5py_file_kwargs) as file:
            # file description
            result_dict['target_id'] = file.attrs['target_id']
            result_dict['data_id'] = file.attrs['data_id']
            result_dict['description'] = file.attrs['description']
            result_dict['date_time'] = file.attrs['date_time']

            # summary statistics
            result_dict['n_param'] = file.attrs['n_param']
            result_dict['bic'] = file.attrs['bic']
            result_dict['noise_level'] = file.attrs['noise_level']

            # linear model parameters
            # y-intercepts
            result_dict['const'] = np.copy(file['const'])
            result_dict['c_err'] = np.copy(file['c_err'])
            result_dict['c_hdi'] = np.copy(file['c_hdi'])
            # slopes
            result_dict['slope'] = np.copy(file['slope'])
            result_dict['sl_err'] = np.copy(file['sl_err'])
            result_dict['sl_hdi'] = np.copy(file['sl_hdi'])

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
            result_dict['passed_sigma'] = np.copy(file['passed_sigma'])
            result_dict['passed_snr'] = np.copy(file['passed_snr'])
            result_dict['passed_both'] = np.copy(file['passed_both'])

            # harmonic model
            result_dict['p_orb'] = np.copy(file['p_orb'])
            result_dict['passed_harmonic'] = np.copy(file['passed_harmonic'])

        # initiate the Results instance and fill in the results
        instance = cls()
        instance.setter(**result_dict)

        if config.VERBOSE:
            print(f'Loaded analysis file with target identifier: {result_dict['target_id']}, '
                  f'created on {result_dict['date_time']}. \n'
                  f'Data identifier: {result_dict['data_id']}. Description: {result_dict['description']} \n')

        return instance

    @classmethod
    def load_conditional(cls, file_name):
        """Load a result file into a Result instance only if it exists and if no overwriting.

        Returns
        -------
        instance: Result object, None
            Instance of the Result class with the loaded results.
            Returns None if condition not met.
        """
        # guard for existing file when not overwriting
        if (not os.path.isfile(file_name)) | config.OVERWRITE:
            return None

        # make the Data instance and load the data
        instance = cls()
        instance.load(file_name)

        return instance

    def save(self):
        """Save the results to a file in hdf5 format.

        Returns
        -------
        None
        """
        with h5py.File(self.file_name, 'w') as file:
            file.attrs['target_id'] = self.target_id
            file.attrs['data_id'] = self.data_id
            file.attrs['description'] = self.description
            file.attrs['date_time'] = str(datetime.datetime.now())
            file.attrs['n_param'] = self.n_param  # number of free parameters
            file.attrs['bic'] = self.bic  # Bayesian Information Criterion of the residuals
            file.attrs['noise_level'] = self.noise_level  # standard deviation of the residuals
            # orbital period
            file.create_dataset('p_orb', data=self.p_orb)
            file['p_orb'].attrs['unit'] = 'd'
            file['p_orb'].attrs['description'] = 'Orbital period and error estimates.'
            # the linear model
            # y-intercepts
            file.create_dataset('const', data=self.const)
            file['const'].attrs['unit'] = 'median normalised flux'
            file['const'].attrs['description'] = 'y-intercept per analysed sector'
            file.create_dataset('c_err', data=self.c_err)
            file['c_err'].attrs['unit'] = 'median normalised flux'
            file['c_err'].attrs['description'] = 'errors in the y-intercept per analysed sector'
            file.create_dataset('c_hdi', data=self.c_hdi)
            file['c_hdi'].attrs['unit'] = 'median normalised flux'
            file['c_hdi'].attrs['description'] = 'HDI for the y-intercept per analysed sector'
            # slopes
            file.create_dataset('slope', data=self.slope)
            file['slope'].attrs['unit'] = 'median normalised flux / d'
            file['slope'].attrs['description'] = 'slope per analysed sector'
            file.create_dataset('sl_err', data=self.sl_err)
            file['sl_err'].attrs['unit'] = 'median normalised flux / d'
            file['sl_err'].attrs['description'] = 'error in the slope per analysed sector'
            file.create_dataset('sl_hdi', data=self.sl_hdi)
            file['sl_hdi'].attrs['unit'] = 'median normalised flux / d'
            file['sl_hdi'].attrs['description'] = 'HDI for the slope per analysed sector'
            # the sinusoid model
            # frequencies
            file.create_dataset('f_n', data=self.f_n)
            file['f_n'].attrs['unit'] = '1 / d'
            file['f_n'].attrs['description'] = 'frequencies of a number of sine waves'
            file.create_dataset('f_n_err', data=self.f_n_err)
            file['f_n_err'].attrs['unit'] = '1 / d'
            file['f_n_err'].attrs['description'] = 'errors in the frequencies of a number of sine waves'
            file.create_dataset('f_n_hdi', data=self.f_n_hdi)
            file['f_n_hdi'].attrs['unit'] = '1 / d'
            file['f_n_hdi'].attrs['description'] = 'HDI for the frequencies of a number of sine waves'
            # amplitudes
            file.create_dataset('a_n', data=self.a_n)
            file['a_n'].attrs['unit'] = 'median normalised flux'
            file['a_n'].attrs['description'] = 'amplitudes of a number of sine waves'
            file.create_dataset('a_n_err', data=self.a_n_err)
            file['a_n_err'].attrs['unit'] = 'median normalised flux'
            file['a_n_err'].attrs['description'] = 'errors in the amplitudes of a number of sine waves'
            file.create_dataset('a_n_hdi', data=self.a_n_hdi)
            file['a_n_hdi'].attrs['unit'] = 'median normalised flux'
            file['a_n_hdi'].attrs['description'] = 'HDI for the amplitudes of a number of sine waves'
            # phases
            file.create_dataset('ph_n', data=self.ph_n)
            file['ph_n'].attrs['unit'] = 'radians'
            file['ph_n'].attrs['description'] = 'phases of a number of sine waves, with reference point t_mean'
            file.create_dataset('ph_n_err', data=self.ph_n_err)
            file['ph_n_err'].attrs['unit'] = 'radians'
            file['ph_n_err'].attrs['description'] = 'errors in the phases of a number of sine waves'
            file.create_dataset('ph_n_hdi', data=self.ph_n_hdi)
            file['ph_n_hdi'].attrs['unit'] = 'radians'
            file['ph_n_hdi'].attrs['description'] = 'HDI for the phases of a number of sine waves'
            # selection criteria
            file.create_dataset('passed_sigma', data=self.passed_sigma)
            file['passed_sigma'].attrs['description'] = 'sinusoids passing the sigma criterion'
            file.create_dataset('passed_snr', data=self.passed_snr)
            file['passed_snr'].attrs['description'] = 'sinusoids passing the signal to noise criterion'
            file.create_dataset('passed_both', data=self.passed_both)
            file['passed_both'].attrs[
                'description'] = 'sinusoids passing both the sigma and the signal to noise criteria'
            file.create_dataset('passed_harmonic', data=self.passed_harmonic)
            file['passed_harmonic'].attrs['description'] = 'harmonic sinusoids passing the sigma criterion'

        return None

    def save_conditional(self):
        """Save a result file only if it doesn't exist or if it exists and if no overwriting.

        Returns
        -------
        None
        """
        if (not os.path.isfile(self.file_name)) | config.OVERWRITE:
            self.save()

        return None

    def save_as_csv(self):
        """Write multiple ascii csv files for human readability.

        Returns
        -------
        None
        """
        # file extension
        ext = os.path.splitext(os.path.basename(self.file_name))[1]

        # linear model parameters
        data = np.column_stack((self.const, self.c_err, self.c_hdi[:, 0], self.c_hdi[:, 1],
                                self.slope, self.sl_err, self.sl_hdi[:, 0], self.sl_hdi[:, 1]))
        hdr = 'const, c_err, c_hdi_l, c_hdi_r, slope, sl_err, sl_hdi_l, sl_hdi_r'
        file_name_lin = self.file_name.replace(ext, '_linear.csv')
        np.savetxt(file_name_lin, data, delimiter=',', header=hdr)

        # sinusoid model parameters
        data = np.column_stack((self.f_n, self.f_n_err, self.f_n_hdi[:, 0], self.f_n_hdi[:, 1],
                                self.a_n, self.a_n_err, self.a_n_hdi[:, 0], self.a_n_hdi[:, 1],
                                self.ph_n, self.ph_n_err, self.ph_n_hdi[:, 0], self.ph_n_hdi[:, 1],
                                self.passed_sigma, self.passed_snr, self.passed_both, self.passed_harmonic))
        hdr = ('f_n, f_n_err, f_n_hdi_l, f_n_hdi_r, a_n, a_n_err, a_n_hdi_l, a_n_hdi_r, '
               'ph_n, ph_n_err, ph_n_hdi_l, ph_n_hdi_r, passed_sigma, passed_snr, passed_b, passed_h')
        file_name_sin = self.file_name.replace(ext, '_sinusoid.csv')
        np.savetxt(file_name_sin, data, delimiter=',', header=hdr)

        # period and statistics
        names = ('p_orb', 'p_err', 'p_hdi_l', 'p_hdi_r'  'n_param', 'bic', 'noise_level')
        stats = (self.p_orb[0], self.p_orb[1], self.p_orb[2], self.p_orb[3], self.n_param, self.bic, self.noise_level)
        desc = ['Orbital period', 'Error in the orbital period', 'Left bound HDI of the orbital period',
                'Right bound HDI of the orbital period', 'Number of free parameters',
                'Bayesian Information Criterion of the residuals', 'Standard deviation of the residuals']
        data = np.column_stack((names, stats, desc))
        hdr = f'{self.target_id}, {self.data_id}, Model statistics\nname, value, description'
        file_name_stats = self.file_name.replace(ext, '_stats.csv')
        np.savetxt(file_name_stats, data, delimiter=',', header=hdr, fmt='%s')

        return None


class Pipeline:
    """A class to analyze light curve data.

    Handles the full analysis pipeline of Star Shine.

    Attributes
    ----------
    data: Data object
        Instance of the Data class holding the light curve data.
    result
    save_dir
    save_subdir
    """

    def __init__(self, data, save_dir=''):
        """Initialises the Pipeline object.

        Parameters
        ----------
        data: Data object
            Instance of the Data class with the data tto be analysed.
        save_dir: str, optional
            Root directory where the data files are stored. Added to the file name. If empty, it is loaded from config.

        Notes
        -----
        Creates a directory where all the analysis result files will be stored.
        """
        # set the data and result objects
        self.data = data  # the data to be analysed
        self.result = Result()  # an empty result instance

        # the files will be stored here
        if save_dir == '':
            save_dir = config.SAVE_DIR
        self.save_dir = save_dir
        self.save_subdir = f'{self.data.target_id}_analysis'

        # check the input data
        if not isinstance(data, Data):
            logger.info("Input `data` should be a Data object.")
        elif len(data.time) == 0:
            logger.info("Data object does not contain time series data.")

        # for saving, make a folder if not there yet
        if not os.path.isdir(os.path.join(save_dir, self.save_subdir)):
            os.mkdir(os.path.join(save_dir, self.save_subdir))  # create the subdir

        return

    def iterative_prewhitening(self):
        """Iterative prewhitening of the input flux time series
        in the form of sine waves and a piece-wise linear curve

        After extraction, a final check is done to see whether some
        frequencies are better removed or groups of frequencies are
        better replaced by one frequency.

        Returns
        -------
        self.result: Result object
            Instance of the Result class containing the analysis results

        """
        t_a = systime.time()
        file_name = os.path.join(self.save_dir, self.save_subdir, f'{self.data.target_id}_iterative_prewhitening.hdf5')

        # guard for existing file when not overwriting
        if os.path.isfile(file_name) & (not config.OVERWRITE):
            self.result = Result.load_conditional(file_name)
            return self.result
        self.result = Result()  # otherwise empty the results in case it was filled

        if config.VERBOSE:
            print(f'Looking for frequencies')

        # extract all frequencies with the iterative scheme
        out_a = tsf.extract_sinusoids(self.data.time, self.data.flux, self.data.i_chunks, select=config.SELECT,
                                      verbose=config.VERBOSE)

        # remove any frequencies that end up not making the statistical cut
        out_b = tsf.reduce_sinusoids(self.data.time, self.data.flux, 0, *out_a, self.data.i_chunks,
                                     verbose=config.VERBOSE)
        const, slope, f_n, a_n, ph_n = out_b

        # select frequencies based on some significance criteria
        out_c = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, 0,
                                     const, slope, f_n, a_n, ph_n, self.data.i_chunks, verbose=config.VERBOSE)
        passed_sigma, passed_snr, passed_both, passed_h = out_c

        # main function done, do the rest for this step
        model_linear = tsf.linear_curve(self.data.time, const, slope, self.data.i_chunks)
        model_sinusoid = tsf.sum_sines(self.data.time, f_n, a_n, ph_n)
        resid = self.data.flux - model_linear - model_sinusoid
        n_param = 2 * len(const) + 3 * len(f_n)
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err,
                                                                             a_n, self.data.i_chunks)

        # save the result
        sin_mean = [const, slope, f_n, a_n, ph_n]
        sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
        sin_select = [passed_sigma, passed_snr, passed_h]
        stats = (*t_stats, n_param, bic, noise_level)
        desc = 'Frequency extraction results.'
        ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_select=sin_select, stats=stats,
                            i_sectors=i_sectors, description=desc, data_id=data_id)

        # print some useful info
        t_b = systime.time()
        if config.VERBOSE:
            print(f'\033[1;32;48mExtraction of sinusoids complete.\033[0m')
            print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
                  f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')

        return self.result


def iterative_prewhitening(times, signal, signal_err, i_sectors, t_stats, file_name, data_id='none', overwrite=False,
                           verbose=False):
    """Iterative prewhitening of the input signal in the form of
    sine waves and a piece-wise linear curve

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
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
    After extraction, a final check is done to see whether some
    groups of frequencies are better replaced by one frequency.
    """
    t_a = systime.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_result_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        return const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Looking for frequencies')
    # extract all frequencies with the iterative scheme
    out_a = tsf.extract_sinusoids(times, signal, i_sectors, select='hybrid', verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_b = tsf.reduce_sinusoids(times, signal, 0, *out_a, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    # select frequencies based on some significance criteria
    out_c = tsf.select_sinusoids(times, signal, signal_err, 0, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=verbose)
    passed_sigma, passed_snr, passed_both, passed_h = out_c
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    n_param = 2 * len(const) + 3 * len(f_n)
    bic = tsf.calc_bic(resid, n_param)
    noise_level = ut.std_unb(resid, len(times) - n_param)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_select = [passed_sigma, passed_snr, passed_h]
    stats = (*t_stats, n_param, bic, noise_level)
    desc = 'Frequency extraction results.'
    ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_select=sin_select, stats=stats,
                        i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful info
    t_b = systime.time()
    if verbose:
        print(f'\033[1;32;48mExtraction of sinusoids complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return const, slope, f_n, a_n, ph_n


def optimise_sinusoid(times, signal, signal_err, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                      method='fitter', data_id='none', overwrite=False, verbose=False):
    """Optimise the parameters of the sinusoid and linear model

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
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
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    method: str
        Method of optimisation. Can be 'sampler' or 'fitter'.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
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
    """
    t_a = systime.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_result_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        return const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Starting multi-sinusoid NL-LS optimisation.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # use the chosen optimisation method
    inf_data, par_mean, par_hdi = None, None, None
    if method == 'fitter':
        par_mean = tsfit.fit_multi_sinusoid_per_group(times, signal, const, slope, f_n, a_n, ph_n, i_sectors,
                                                      verbose=verbose)
    else:
        # make model including everything to calculate noise level
        model_lin = tsf.linear_curve(times, const, slope, i_sectors)
        model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
        resid = signal - (model_lin + model_sin)
        n_param = 2 * len(const) + 3 * len(f_n)
        noise_level = ut.std_unb(resid, len(times) - n_param)
        # formal linear and sinusoid parameter errors
        c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
        # do not include those frequencies that have too big uncertainty
        include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
        f_n, a_n, ph_n = f_n[include], a_n[include], ph_n[include]
        f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
        # Monte Carlo sampling of the model
        output = mcf.sample_sinusoid(times, signal, const, slope, f_n, a_n, ph_n, c_err, sl_err, f_n_err, a_n_err,
                                     ph_n_err, noise_level, i_sectors, verbose=verbose)
        inf_data, par_mean, par_hdi = output
    const, slope, f_n, a_n, ph_n = par_mean
    # select frequencies based on some significance criteria
    out_b = tsf.select_sinusoids(times, signal, signal_err, 0, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=verbose)
    passed_sigma, passed_snr, passed_both, passed_h = out_b
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    n_param = 2 * len(const) + 3 * len(f_n)
    bic = tsf.calc_bic(resid, n_param)
    noise_level = ut.std_unb(resid, len(times) - n_param)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_hdi = par_hdi
    sin_select = [passed_sigma, passed_snr, passed_h]
    stats = (t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level)
    desc = 'Multi-sinusoid NL-LS optimisation results.'
    ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_hdi=sin_hdi, sin_select=sin_select,
                        stats=stats, i_sectors=i_sectors, description=desc, data_id=data_id)
    ut.save_inference_data(file_name, inf_data)
    # print some useful info
    t_b = systime.time()
    if verbose:
        print(f'\033[1;32;48mOptimisation of sinusoids complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return const, slope, f_n, a_n, ph_n


def couple_harmonics(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                     data_id='none', overwrite=False, verbose=False):
    """Find the orbital period and couple harmonic frequencies to the orbital period

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
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
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
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
    
    Notes
    -----
    Performs a global period search, if the period is unknown.
    If a period is given, it is locally refined for better performance.
    
    Removes theoretical harmonic candidate frequencies within the frequency
    resolution, then extracts a single harmonic at the theoretical location.
    
    Removes any frequencies that end up not making the statistical cut.
    """
    t_a = systime.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_result_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        p_orb, _ = results['ephem']
        return p_orb, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Coupling the harmonic frequencies to the orbital frequency.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # if given, the input p_orb is refined locally, otherwise the period is searched for globally
    if (p_orb == 0):
        p_orb, mult = tsf.find_orbital_period(times, signal, f_n)
        # notify the user if a multiple was chosen
        if (mult != 1):
            logger.info(f'Multiple of the period chosen: {mult}')
    else:
        p_orb = tsf.refine_orbital_period(p_orb, times, f_n)
    # if time series too short, or no harmonics found, log and warn and maybe cut off the analysis
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=freq_res / 2)
    if (t_tot / p_orb < 1.1):
        out_a = const, slope, f_n, a_n, ph_n  # return previous results
    elif (len(harmonics) < 2):
        out_a = const, slope, f_n, a_n, ph_n  # return previous results
    else:
        # couple the harmonics to the period. likely removes more frequencies that need re-extracting
        out_a = tsf.fix_harmonic_frequency(times, signal, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                           verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_b = tsf.reduce_sinusoids(times, signal, p_orb, *out_a, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_b
    # select frequencies based on some significance criteria
    out_c = tsf.select_sinusoids(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=verbose)
    passed_sigma, passed_snr, passed_both, passed_h = out_c
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid, n_param)
    noise_level = ut.std_unb(resid, len(times) - n_param)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_select = [passed_sigma, passed_snr, passed_h]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Harmonic frequencies coupled.'
    ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_select=sin_select, ephem=ephem,
                        ephem_err=ephem_err, stats=stats, i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful info
    t_b = systime.time()
    if verbose:
        rnd_p_orb = max(ut.decimal_figures(p_err, 2), ut.decimal_figures(p_orb, 2))
        print(f'\033[1;32;48mOrbital harmonic frequencies coupled.\033[0m')
        print(f'\033[0;32;48mp_orb: {p_orb:.{rnd_p_orb}f} (+-{p_err:.{rnd_p_orb}f}), \n'
              f'{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return p_orb, const, slope, f_n, a_n, ph_n


def add_sinusoids(times, signal, signal_err, p_orb, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                  data_id='none', overwrite=False, verbose=False):
    """Find and add more (harmonic and non-harmonic) frequencies if possible

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
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
    First looks for additional harmonic frequencies at the integer multiples
    of the orbital frequency.

    Then looks for any additional non-harmonic frequencies taking into account
    the existing harmonics.

    Finally, removes any frequencies that end up not making the statistical cut.
    """
    t_a = systime.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_result_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        return const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Looking for additional frequencies.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    n_f_init = len(f_n)
    # start by looking for more harmonics
    out_a = tsf.extract_harmonics(times, signal, p_orb, i_sectors, f_n, a_n, ph_n, verbose=verbose)
    # look for any additional non-harmonics with the iterative scheme
    out_b = tsf.extract_sinusoids(times, signal, i_sectors, p_orb, *out_a[2:], select='hybrid', verbose=verbose)
    # remove any frequencies that end up not making the statistical cut
    out_c = tsf.reduce_sinusoids(times, signal, p_orb, *out_b, i_sectors, verbose=verbose)
    const, slope, f_n, a_n, ph_n = out_c
    # select frequencies based on some significance criteria
    out_d = tsf.select_sinusoids(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=verbose)
    passed_sigma, passed_snr, passed_both, passed_h = out_d
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid, n_param)
    noise_level = ut.std_unb(resid, len(times) - n_param)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_select = [passed_sigma, passed_snr, passed_h]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Additional non-harmonic extraction.'
    ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_select=sin_select, ephem=ephem,
                        ephem_err=ephem_err, stats=stats, i_sectors=i_sectors, description=desc, data_id=data_id)
    # print some useful info
    t_b = systime.time()
    if verbose:
        print(f'\033[1;32;48mExtraction of {len(f_n) - n_f_init} additional sinusoids complete.\033[0m')
        print(f'\033[0;32;48m{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return const, slope, f_n, a_n, ph_n


def optimise_sinusoid_h(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors, t_stats, file_name,
                        method='fitter', data_id='none', overwrite=False, verbose=False):
    """Optimise the parameters of the sinusoid and linear model with coupled harmonics

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
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
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    file_name: str
        File name (including path) for saving the results. Also used to
        load previous analysis results if found.
    method: str
        Method of optimisation. Can be 'sampler' or 'fitter'.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
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
    """
    t_a = systime.time()
    # guard for existing file when not overwriting
    if os.path.isfile(file_name) & (not overwrite):
        results = ut.read_result_hdf5(file_name, verbose=verbose)
        const, slope, f_n, a_n, ph_n = results['sin_mean']
        p_orb, _ = results['ephem']
        return p_orb, const, slope, f_n, a_n, ph_n
    
    if verbose:
        print(f'Starting multi-sine NL-LS optimisation with harmonics.')
    t_tot, t_mean, t_mean_s, t_int = t_stats
    # use the chosen optimisation method
    inf_data, par_mean, sin_hdi, ephem_hdi = None, None, None, None
    if method == 'fitter':
        par_mean = tsfit.fit_multi_sinusoid_harmonics_per_group(times, signal, p_orb, const, slope, f_n, a_n, ph_n,
                                                                i_sectors, verbose=verbose)
    else:
        # make model including everything to calculate noise level
        model_lin = tsf.linear_curve(times, const, slope, i_sectors)
        model_sin = tsf.sum_sines(times, f_n, a_n, ph_n)
        resid = signal - (model_lin + model_sin)
        harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
        n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
        noise_level = ut.std_unb(resid, len(times) - n_param)
        # formal linear and sinusoid parameter errors
        c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
        p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
        # do not include those frequencies that have too big uncertainty
        include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
        f_n, a_n, ph_n = f_n[include], a_n[include], ph_n[include]
        f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
        # Monte Carlo sampling of the model
        output = mcf.sample_sinusoid_h(times, signal, p_orb, const, slope, f_n, a_n, ph_n, p_err, c_err, sl_err,
                                       f_n_err, a_n_err, ph_n_err, noise_level, i_sectors, verbose=verbose)
        inf_data, par_mean, par_hdi = output
        sin_hdi = par_hdi[1:]
        ephem_hdi = np.array([par_hdi[0], [-1, -1]])
    p_orb, const, slope, f_n, a_n, ph_n = par_mean
    # select frequencies based on some significance criteria
    out_b = tsf.select_sinusoids(times, signal, signal_err, p_orb, const, slope, f_n, a_n, ph_n, i_sectors,
                                 verbose=verbose)
    passed_sigma, passed_snr, passed_both, passed_h = out_b
    # main function done, do the rest for this step
    model_linear = tsf.linear_curve(times, const, slope, i_sectors)
    model_sinusoid = tsf.sum_sines(times, f_n, a_n, ph_n)
    resid = signal - model_linear - model_sinusoid
    # calculate number of parameters, BIC and noise level
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9)
    n_param = 2 * len(const) + 1 + 2 * len(harmonics) + 3 * (len(f_n) - len(harmonics))
    bic = tsf.calc_bic(resid, n_param)
    noise_level = ut.std_unb(resid, len(times) - n_param)
    c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(times, resid, signal_err, a_n, i_sectors)
    p_err, _, _ = af.linear_regression_uncertainty(p_orb, t_tot, sigma_t=t_int / 2)
    # save the result
    sin_mean = [const, slope, f_n, a_n, ph_n]
    sin_err = [c_err, sl_err, f_n_err, a_n_err, ph_n_err]
    sin_select = [passed_sigma, passed_snr, passed_h]
    ephem = np.array([p_orb, -1])
    ephem_err = np.array([p_err, -1])
    stats = [t_tot, t_mean, t_mean_s, t_int, n_param, bic, noise_level]
    desc = 'Multi-sine NL-LS optimisation results with coupled harmonics.'
    ut.save_result_hdf5(file_name, sin_mean=sin_mean, sin_err=sin_err, sin_hdi=sin_hdi, sin_select=sin_select,
                        ephem=ephem, ephem_err=ephem_err, ephem_hdi=ephem_hdi, stats=stats, i_sectors=i_sectors,
                        description=desc, data_id=data_id)
    ut.save_inference_data(file_name, inf_data)
    # print some useful info
    t_b = systime.time()
    if verbose:
        rnd_p_orb = max(ut.decimal_figures(p_err, 2), ut.decimal_figures(p_orb, 2))
        print(f'\033[1;32;48mOptimisation with coupled harmonics complete.\033[0m')
        print(f'\033[0;32;48mp_orb: {p_orb:.{rnd_p_orb}f} (+-{p_err:.{rnd_p_orb}f}), \n'
              f'{len(f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. '
              f'Time taken: {t_b - t_a:1.1f}s\033[0m\n')
    return p_orb, const, slope, f_n, a_n, ph_n


def analyse_frequencies(times, signal, signal_err, i_sectors, t_stats, target_id, save_dir, data_id='none',
                        overwrite=False, save_ascii=False, verbose=False):
    """Recipe for the extraction of sinusoids from light curves.

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    target_id: int, str
        The TESS Input Catalog number for saving and loading.
        Use the name of the input light curve file if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    save_ascii: bool
        Additionally save ascii conversions of the output (hdf5)
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    const_i: list[numpy.ndarray[Any, dtype[float]]]
        y-intercepts of a piece-wise linear curve for each stage of the analysis
    slope_i: list[numpy.ndarray[Any, dtype[float]]]
        slopes of a piece-wise linear curve for each stage of the analysis
    f_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Frequencies of a number of sine waves for each stage of the analysis
    a_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Amplitudes of a number of sine waves for each stage of the analysis
    ph_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Phases of a number of sine waves for each stage of the analysis

    Notes
    -----
    The followed recipe is:

    1) Extract all frequencies
        We start by extracting the frequency with the highest amplitude one by one,
        directly from the Lomb-Scargle periodogram until the BIC does not significantly
        improve anymore. This step involves a final cleanup of the frequencies.

    2) Multi-sinusoid NL-LS optimisation
        The sinusoid parameters are optimised with a non-linear least-squares method,
        using groups of frequencies to limit the number of free parameters.
    """
    t_a = systime.time()
    # signal_err = # signal errors are largely ignored for now. The likelihood assumes the same errors and uses the MLE
    arg_dict = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}  # these stay the same
    # -------------------------------------------------------
    # --- [1] --- Initial iterative extraction of frequencies
    # -------------------------------------------------------
    file_name_1 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_1.hdf5')
    out_1 = iterative_prewhitening(times, signal, signal_err, i_sectors, t_stats, file_name_1, **arg_dict)
    const_1, slope_1, f_n_1, a_n_1, ph_n_1 = out_1
    if (len(f_n_1) == 0):
        logger.info('No frequencies found.')
        const_i = [const_1]
        slope_i = [slope_1]
        f_n_i = [f_n_1]
        a_n_i = [a_n_1]
        ph_n_i = [ph_n_1]
        return const_i, slope_i, f_n_i, a_n_i, ph_n_i
    # ----------------------------------------------------------------
    # --- [2] --- Multi-sinusoid non-linear least-squares optimisation
    # ----------------------------------------------------------------
    file_name_2 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_2.hdf5')
    out_2 = optimise_sinusoid(times, signal, signal_err, const_1, slope_1, f_n_1, a_n_1, ph_n_1, i_sectors, t_stats,
                              file_name_2, method='fitter', **arg_dict)
    const_2, slope_2, f_n_2, a_n_2, ph_n_2 = out_2
    # save final freqs and linear curve in ascii format
    if save_ascii:
        ut.convert_hdf5_to_ascii(file_name_2)
    # make lists for output
    const_i = [const_1, const_2]
    slope_i = [slope_1, slope_2]
    f_n_i = [f_n_1, f_n_2]
    a_n_i = [a_n_1, a_n_2]
    ph_n_i = [ph_n_1, ph_n_2]
    # final timing and message
    t_b = systime.time()
    logger.info(f'Frequency extraction done. Total time elapsed: {t_b - t_a:1.1f}s.')
    return const_i, slope_i, f_n_i, a_n_i, ph_n_i


def analyse_harmonics(times, signal, signal_err, i_sectors, p_orb, t_stats, target_id, save_dir, data_id='none',
                      overwrite=False, save_ascii=False, verbose=False):
    """Recipe for the extraction of harmonic sinusoids from EB light curves.

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_half_s = np.array([[0, len(times)]]).
    p_orb: float
        The orbital period. Set to 0 to search for the best period.
        If the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    t_stats: list[float]
        Some time series statistics: t_tot, t_mean, t_mean_s, t_int
    target_id: int, str
        The TESS Input Catalog number for saving and loading.
        Use the name of the input light curve file if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    save_ascii: bool
        Additionally save ascii conversions of the output (hdf5)
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    p_orb_i: list[float]
        Orbital period at each stage of the analysis
    const_i: list[numpy.ndarray[Any, dtype[float]]]
        y-intercepts of a piece-wise linear curve for each stage of the analysis
    slope_i: list[numpy.ndarray[Any, dtype[float]]]
        slopes of a piece-wise linear curve for each stage of the analysis
    f_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Frequencies of a number of sine waves for each stage of the analysis
    a_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Amplitudes of a number of sine waves for each stage of the analysis
    ph_n_i: list[numpy.ndarray[Any, dtype[float]]]
        Phases of a number of sine waves for each stage of the analysis

    Notes
    -----
    The followed recipe is:
    
    3) Measure the orbital period and couple the harmonic frequencies
        Global search done with combined phase dispersion, Lomb-Scargle and length/
        filling factor of the harmonic series in the list of frequencies.
        Then sets the frequencies of the harmonics to their new values, coupling them
        to the orbital period. This step involves a final cleanup of the frequencies.
        [Note: it is possible to provide a fixed period if it is already known.
        It will still be optimised]

    4) Attempt to extract additional frequencies
        The decreased number of free parameters (2 vs. 3), the BIC, which punishes for free
        parameters, may allow the extraction of more harmonics.
        It is also attempted to extract more frequencies like in step 1 again, now taking
        into account the presence of harmonics.
        This step involves a final cleanup of the frequencies.

    5) Multi-sinusoid NL-LS optimisation with coupled harmonics
        Optimise all sinusoid parameters with a non-linear least-squares method,
        using groups of frequencies to limit the number of free parameters
        and including the orbital period and the coupled harmonics.
    """
    t_a = systime.time()
    # signal_err = # signal errors are largely ignored for now. The likelihood assumes the same errors and uses the MLE
    t_tot, t_mean, t_mean_s, t_int = t_stats
    freq_res = 1.5 / t_tot  # Rayleigh criterion
    arg_dict = {'data_id': data_id, 'overwrite': overwrite, 'verbose': verbose}  # these stay the same
    # read in the frequency analysis results
    file_name = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_2.hdf5')
    if not os.path.isfile(file_name):
        if verbose:
            print(f'No frequency analysis results found ({file_name})')
        return [], [], [], [], [], []
    results_2 = ut.read_result_hdf5(file_name, verbose=verbose)
    const_2, slope_2, f_n_2, a_n_2, ph_n_2 = results_2['sin_mean']
    # --------------------------------------------------------------------------
    # --- [3] --- Measure the orbital period and couple the harmonic frequencies
    # --------------------------------------------------------------------------
    file_name_3 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_3.hdf5')
    out_3 = couple_harmonics(times, signal, signal_err, p_orb, const_2, slope_2, f_n_2, a_n_2, ph_n_2, i_sectors,
                             t_stats, file_name_3, **arg_dict)
    p_orb_3, const_3, slope_3, f_n_3, a_n_3, ph_n_3 = out_3
    # save info and exit in the following cases (and log message)
    harmonics, harmonic_n = af.find_harmonics_from_pattern(f_n_2, p_orb_3, f_tol=freq_res / 2)
    if (t_tot / p_orb_3 < 1.1):
        logger.info(f'Period over time-base is less than two: {t_tot / p_orb_3}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
    elif (len(harmonics) < 2):
        logger.info(f'Not enough harmonics found: {len(harmonics)}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
        # return previous results
    elif (t_tot / p_orb_3 < 2):
        logger.info(f'Period over time-base is less than two: {t_tot / p_orb_3}; '
                    f'period (days): {p_orb_3}; time-base (days): {t_tot}')
    if (t_tot / p_orb_3 < 1.1) | (len(harmonics) < 2):
        p_orb_i = [p_orb_3]
        const_i = [const_2]
        slope_i = [slope_2]
        f_n_i = [f_n_2]
        a_n_i = [a_n_2]
        ph_n_i = [ph_n_2]
        return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i
    # -----------------------------------------------------
    # --- [4] --- Attempt to extract additional frequencies
    # -----------------------------------------------------
    file_name_4 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_4.hdf5')
    out_4 = add_sinusoids(times, signal, signal_err, p_orb_3, f_n_3, a_n_3, ph_n_3, i_sectors, t_stats, file_name_4,
                          **arg_dict)
    const_4, slope_4, f_n_4, a_n_4, ph_n_4 = out_4
    # -----------------------------------------------
    # --- [5] --- Optimisation with coupled harmonics
    # -----------------------------------------------
    file_name_5 = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}_analysis_5.hdf5')
    out_5 = optimise_sinusoid_h(times, signal, signal_err, p_orb_3, const_4, slope_4, f_n_4, a_n_4, ph_n_4, i_sectors,
                                t_stats, file_name_5, method='fitter', **arg_dict)
    p_orb_5, const_5, slope_5, f_n_5, a_n_5, ph_n_5 = out_5
    # save final freqs and linear curve in ascii format
    if save_ascii:
        ut.convert_hdf5_to_ascii(file_name_5)
    # make lists for output
    p_orb_i = [p_orb_3, p_orb_3, p_orb_5]
    const_i = [const_3, const_4, const_5]
    slope_i = [slope_3, slope_4, slope_5]
    f_n_i = [f_n_3, f_n_4, f_n_5]
    a_n_i = [a_n_3, a_n_4, a_n_5]
    ph_n_i = [ph_n_3, ph_n_4, ph_n_5]
    # final timing and message
    t_b = systime.time()
    logger.info(f'Harmonic analysis done. Total time elapsed: {t_b - t_a:1.1f}s.')
    return p_orb_i, const_i, slope_i, f_n_i, a_n_i, ph_n_i


def customize_logger(save_dir, target_id, verbose):
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
    logger.setLevel(logging.INFO)  # set base activation level for logger
    # make formatters for the handlers
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')
    # remove existing handlers to avoid duplicate messages
    if (logger.hasHandlers()):
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


def analyse_light_curve(times, signal, signal_err, p_orb, i_sectors, target_id, save_dir, stage='all', method='fitter',
                        data_id='none', overwrite=False, save_ascii=False, verbose=False):
    """Do all steps of the analysis (or fewer)

    Parameters
    ----------
    times: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    signal: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    signal_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    p_orb: float
        Orbital period of the eclipsing binary in days (set zero if unkown)
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    target_id: int, str
        The TESS Input Catalog number for saving and loading.
        Use the name of the input light curve file if not available.
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    stage: str
        Which analysis stages to do: 'all'/'a' for everything
        'frequencies'/'freq'/'f' for just the iterative prewhitening
        'harmonics'/'harm'/'h' for up to and including the harmonic coupling only
        'timings'/'t' for up to and including finding the eclipse timings
    method: str
        Method of EB light curve model optimisation. Can be 'sampler' or 'fitter'.
        Sampler gives extra error estimates on the eclipse parameters
        Fitter is much faster and still accurate
    data_id: str
        User defined identification for the dataset used
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    save_ascii: bool
        Additionally save ascii conversions of the output (hdf5)
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    None
    
    Notes
    -----
    If only an orbital period is desired, go for stage='harmonics'.
    """
    t_a = systime.time()
    # for saving, make a folder if not there yet
    if not os.path.isdir(os.path.join(save_dir, f'{target_id}_analysis')):
        os.mkdir(os.path.join(save_dir, f'{target_id}_analysis'))  # create the subdir
    # create a log
    customize_logger(save_dir, target_id, verbose)  # log stuff to a file and/or stdout
    logger.info('Start of analysis')  # info to save to log
    # time series stats
    t_tot = np.ptp(times)  # total time base of observations
    t_mean = np.mean(times)  # mean time of observations
    t_mean_s = np.array([np.mean(times[s[0]:s[1]]) for s in i_sectors])  # mean time per observation sector
    t_int = np.median(np.diff(times))  # integration time, taken to be the median time step
    t_stats = [t_tot, t_mean, t_mean_s, t_int]
    # keyword arguments in common between some functions
    kw_args = {'save_dir': save_dir, 'data_id': data_id, 'overwrite': overwrite, 'save_ascii': save_ascii,
               'verbose': verbose}
    # define the lists of stages to compare against
    stg_1 = ['harmonics', 'harm', 'h', 'timings', 't', 'all', 'a']
    # do the analysis -------------------------------------------------------------------------------
    out_a = analyse_frequencies(times, signal, signal_err, i_sectors, t_stats, target_id, **kw_args)
    # need outputs of len 2 to continue
    if (not (len(out_a[0]) < 2)) & (stage in stg_1):
        out_b = analyse_harmonics(times, signal, signal_err, i_sectors, p_orb, t_stats, target_id, **kw_args)
    else:
        out_b = ([], [], [], [], [], [])
    # create summary file
    ut.save_summary(target_id, save_dir, data_id=data_id)
    t_b = systime.time()
    logger.info(f'End of analysis. Total time elapsed: {t_b - t_a:1.1f}s.')  # info to save to log
    return None


def analyse_lc_from_file(file_name, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='none',
                         save_dir=None, overwrite=False, verbose=False):
    """Do all steps of the analysis for a given light curve file

    Parameters
    ----------
    file_name: str
        Path to a file containing the light curve data, with
        timestamps, normalised flux, error values as the
        first three columns, respectively.
    p_orb: float
        Orbital period of the eclipsing binary in days
    i_sectors: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating the separately handled timespans
        in the piecewise-linear curve. These can indicate the TESS
        observation sectors, but taking half the sectors is recommended.
        If only a single curve is wanted, set
        i_sectors = np.array([[0, len(times)]]).
    stage: str
        Which analysis stages to do: 'all'/'a' for everything
        'frequencies'/'freq'/'f' for just the iterative prewhitening
        'harmonics'/'harm'/'h' for up to and including the harmonic coupling only
        'timings'/'t' for up to and including finding the eclipse timings
    method: str
        Method of EB light curve model optimisation. Can be 'sampler' or 'fitter'.
        Sampler gives extra error estimates on the eclipse parameters
        Fitter is much faster and still accurate
    data_id: str
        User defined identification for the dataset used
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information

    Returns
    -------
    None

    Notes
    -----
    If save_dir is not given, results are saved in the same directory as
    the given light curve file (will create a subfolder)
    
    The input text files are expected to have three columns with in order:
    times (bjd), signal (flux), signal_err (flux error)
    And the timestamps should be in ascending order.
    The expected text file format is space separated.
    """
    target_id = os.path.splitext(os.path.basename(file_name))[0]  # file name is used as target identifier
    if save_dir is None:
        save_dir = os.path.dirname(file_name)
    # load the data
    times, signal, signal_err = np.loadtxt(file_name, usecols=(0, 1, 2), unpack=True)
    # if sectors not given, take full length
    if i_sectors is None:
        i_sectors = np.array([[0, len(times)]])  # no sector information
    i_half_s = i_sectors  # in this case no differentiation between half or full sectors
    # do the analysis
    analyse_light_curve(times, signal, signal_err, p_orb, i_half_s, target_id, save_dir, stage=stage, method=method,
                        data_id=data_id, overwrite=overwrite, save_ascii=False, verbose=verbose)
    return None


def analyse_lc_from_tic(tic, all_files, p_orb=0, stage='all', method='fitter', data_id='none', save_dir=None,
                        overwrite=False, verbose=False):
    """Do all steps of the analysis for a given TIC number
    
    Parameters
    ----------
    tic: int
        The TESS Input Catalog (TIC) number for loading/saving the data
        and later reference.
    all_files: list[str]
        List of all the TESS data product '.fits' files. The files
        with the corresponding TIC number are selected.
    p_orb: float
        Orbital period of the eclipsing binary in days
    stage: str
        Which analysis stages to do: 'all'/'a' for everything
        'frequencies'/'freq'/'f' for just the iterative prewhitening
        'harmonics'/'harm'/'h' for up to and including the harmonic coupling only
        'timings'/'t' for up to and including finding the eclipse timings
    method: str
        Method of EB light curve model optimisation. Can be 'sampler' or 'fitter'.
        Sampler gives extra error estimates on the eclipse parameters
        Fitter is much faster and still accurate
    data_id: str
        User defined identification for the dataset used
    save_dir: str
        Path to a directory for saving the results. Also used to load
        previous analysis results.
    overwrite: bool
        If set to True, overwrite old results in the same directory as
        save_dir, or (if False) to continue from the last save-point.
    verbose: bool
        If set to True, this function will print some information
    
    Returns
    -------
    None
    
    Notes
    -----
    If save_dir is not given, results are saved in the same directory as
    the given light curve files (will create a subfolder)
    
    Expects to find standardly formatted fits files from the TESS mission
    in the locations provided with all_files.
    """
    if save_dir is None:
        save_dir = os.path.dirname(all_files[0])
    # load the data
    lc_data = ut.load_light_curve(tic, all_files, apply_flags=config.APPLY_Q_FLAGS)
    times, sap_signal, signal, signal_err, sectors, t_sectors, crowdsap = lc_data
    i_sectors = ut.convert_tess_t_sectors(times, t_sectors)
    lc_processed = ut.stitch_chunks(times, signal, signal_err, i_sectors)
    times, signal, signal_err, sector_medians, t_combined, i_half_s = lc_processed
    # do the analysis
    analyse_light_curve(times, signal, signal_err, p_orb, i_half_s, tic, save_dir, stage=stage, method=method,
                        data_id=data_id, overwrite=overwrite, save_ascii=False, verbose=verbose)
    return None


def analyse_set(target_list, function='analyse_lc_from_tic', n_threads=os.cpu_count() - 2, **kwargs):
    """Analyse a set of light curves in parallel
    
    Parameters
    ----------
    target_list: list[str], list[int]
        List of either file names or TIC identifiers to analyse
    function: str
        Name  of the function to use for the analysis
        Choose from [analyse_lc_from_tic, analyse_lc_from_file]
    n_threads: int
        Number of threads to use.
        Uses two fewer than the available amount by default.
    **kwargs: dict
        Extra arguments to 'function': refer to each function's
        documentation for a list of all possible arguments.
    
    Returns
    -------
    None
    
    Raises
    ------
    NotImplementedError:
        If 'p_orb' or 'i_sectors' is given. These cannot be set for each
        target separately, so this will break the parallelisation code.
    """
    if 'p_orb' in kwargs.keys():
        # Use mp.Pool.starmap for this
        raise NotImplementedError('keyword p_orb found in kwargs: this functionality is not yet implemented')
    if 'i_sectors' in kwargs.keys():
        # Use mp.Pool.starmap for this
        raise NotImplementedError('keyword i_sectors found in kwargs: this functionality is not yet implemented')
    
    t1 = systime.time()
    with mp.Pool(processes=n_threads) as pool:
        pool.map(fct.partial(eval(function), **kwargs), target_list, chunksize=1)
    t2 = systime.time()
    print(f'Finished analysing set in: {(t2 - t1):1.2} s ({(t2 - t1) / 3600:1.2} h) for {len(target_list)} targets,\n'
          f'using {n_threads} threads ({(t2 - t1) * n_threads / len(target_list):1.2} s '
          f'average per target single threaded).')
    return None
