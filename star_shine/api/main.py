"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the main functions that link together all functionality.

Code written by: Luc IJspeert
"""
import os
import time as systime
import datetime
import h5py
import numpy as np

from star_shine.core import timeseries_functions as tsf
from star_shine.core import timeseries_fitting as tsfit
from star_shine.core import analysis_functions as af
from star_shine.core import mcmc_functions as mcf
from star_shine.core import utility as ut
from star_shine.core import visualisation as vis
from star_shine.core.config import get_config


# load configuration
config = get_config()


class Data:
    """A class to handle light curve data.

    Attributes
    ----------
    file_list: list[str]
        List of ascii light curve files or (TESS) data product '.fits' files.
    data_dir: str
        Root directory where the data files are stored.
    target_id: str
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
    p_orb: float
        The orbital period. Set to 0 to search for the best period.
        If the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    f_min: float
        Minimum frequency for extraction and periodograms
    f_max: float
        Maximum frequency for extraction and periodograms
    """

    def __init__(self, target_id='', data_id=''):
        """Initialises the Data object.

        The data is loaded from the given file(s) and some basic processing is done.
        Either a file name, or target id plus file list must be given.

        Parameters
        ----------
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        """
        self.file_list = []
        self.data_dir = ''
        self.target_id = target_id
        self.data_id = data_id

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

        self.p_orb = 0.
        self.f_min = 0.
        self.f_max = 0.

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

            if config.verbose:
                print(message)

            # remove the files
            for i in missing:
                del self.file_list[i]

        return None

    def setter(self, **kwargs):
        """Fill in the attributes with data.

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly

        Returns
        -------
        None
        """
        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    @classmethod
    def load_data(cls, file_list, data_dir='', target_id='', data_id=''):
        """Load light curve data from the file list.

        Parameters
        ----------
        data_dir: str, optional
            Root directory where the data files are stored. Added to the file name. If empty, it is loaded from config.
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        file_list: list[str]
            List of ascii light curve files or (TESS) data product '.fits' files. Exclude the path given to 'data_dir'.
            If only one file is given, its file name is used for target_id (if 'none').

        Returns
        -------
        Data
            Instance of the Data class with the loaded data.
        """
        instance = cls()

        # set the file list and data directory
        if data_dir == '':
            data_dir = config.data_dir
        instance.setter(file_list=file_list, data_dir=data_dir)

        # guard against empty list
        if len(file_list) == 0:
            if config.verbose:
                print("Empty file list provided.")
            return

        # Check if the file(s) exist(s)
        instance._check_file_existence()
        if len(instance.file_list) == 0:
            if config.verbose:
                print("No existing files in file list")
            return

        # set IDs
        if target_id == '':
            target_id = os.path.splitext(os.path.basename(file_list[0]))[0]  # file name is used as identifier
        instance.setter(target_id=target_id, data_id=data_id)

        # add data_dir for loading files, if not None
        if instance.data_dir is None:
            file_list_dir = instance.file_list
        else:
            file_list_dir = [os.path.join(instance.data_dir, file) for file in instance.file_list]

        # load the data from the list of files
        lc_data = ut.load_light_curve(file_list_dir, apply_flags=config.apply_q_flags)
        instance.setter(time=lc_data[0], flux=lc_data[1], flux_err=lc_data[2], i_chunks=lc_data[3], medians=lc_data[4])

        # check for overlapping time stamps
        if np.any(np.diff(instance.time) <= 0):
            if config.verbose:
                print("The time array chunks include overlap.")

        # set derived attributes
        instance.t_tot = np.ptp(instance.time)
        instance.t_mean = np.mean(instance.time)
        instance.t_mean_chunk = np.array([np.mean(instance.time[ch[0]:ch[1]]) for ch in instance.i_chunks])
        instance.t_int = np.median(np.diff(instance.time))  # integration time, taken to be the median time step

        instance.f_min = 0.01 / instance.t_tot
        instance.f_max = ut.frequency_upper_threshold(instance.time, func='min')

        return instance

    @classmethod
    def load(cls, file_name, h5py_file_kwargs):
        """Load a data file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the data from
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.

        Returns
        -------
        Data
            Instance of the Data class with the loaded data.
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
            data_dict['description'] = file.attrs['description']
            data_dict['date_time'] = file.attrs['date_time']

            # original list of files
            data_dict['data_dir'] = file.attrs['data_dir']
            data_dict['file_list'] = np.copy(file['file_list'])

            # summary statistics
            data_dict['t_tot'] = file.attrs['t_tot']
            data_dict['t_mean'] = file.attrs['t_mean']
            data_dict['t_int'] = file.attrs['t_int']
            data_dict['p_orb'] = file.attrs['p_orb']

            # the time series data
            data_dict['time'] = np.copy(file['time'])
            data_dict['flux'] = np.copy(file['flux'])
            data_dict['flux_err'] = np.copy(file['flux_err'])

            # additional information
            data_dict['i_chunks'] = np.copy(file['i_chunks'])
            data_dict['flux_counts_medians'] = np.copy(file['flux_counts_medians'])
            data_dict['t_mean_chunk'] = np.copy(file['t_mean_chunk'])

        # initiate the Results instance and fill in the results
        instance = cls()
        instance.setter(**data_dict)

        if config.verbose:
            print(f"Loaded data file with target identifier: {data_dict['target_id']}, "
                  f"created on {data_dict['date_time']}. \n"
                  f"Data identifier: {data_dict['data_id']}. \n")

        return instance

    def save(self, file_name):
        """Save the data to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the data to

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
            file.attrs['target_id'] = self.target_id
            file.attrs['data_id'] = self.data_id
            file.attrs['description'] = 'Star Shine data file'
            file.attrs['date_time'] = str(datetime.datetime.now())

            # original list of files
            file.attrs['data_dir'] = self.data_dir  # original data directory
            file.create_dataset('file_list', data=self.file_list)
            file['file_list'].attrs['description'] = 'original list of files for the creation of this data file'

            # summary statistics
            file.attrs['t_tot'] = self.t_tot  # Total time base of observations
            file.attrs['t_mean'] = self.t_mean  # Time reference (zero) point of the full light curve
            file.attrs['t_int'] = self.t_int  # Integration time of observations (median time step by default)
            file.attrs['p_orb'] = self.p_orb  # orbital period, if applicable

            # the time series data
            file.create_dataset('time', data=self.time)
            file['time'].attrs['unit'] = 'time unit of the data (often days)'
            file['time'].attrs['description'] = 'timestamps of the observations'
            file.create_dataset('flux', data=self.flux)
            file['flux'].attrs['unit'] = 'median normalised flux'
            file['flux'].attrs['description'] = 'normalised flux measurements of the observations'
            file.create_dataset('flux_err', data=self.flux_err)
            file['flux_err'].attrs['unit'] = 'median normalised flux'
            file['flux_err'].attrs['description'] = 'normalised error measurements in the flux'

            # additional information
            file.create_dataset('i_chunks', data=self.i_chunks)
            file['i_chunks'].attrs['description'] = 'pairs of indices indicating time chunks of the data'
            file.create_dataset('flux_counts_medians', data=self.flux_counts_medians)
            file['flux_counts_medians'].attrs['unit'] = 'raw flux counts'
            file['flux_counts_medians'].attrs['description'] = 'median flux level per time chunk'
            file.create_dataset('t_mean_chunk', data=self.t_mean_chunk)
            file['t_mean_chunk'].attrs['unit'] = 'time unit of the data (often days)'
            file['t_mean_chunk'].attrs['description'] = 'time reference (zero) point of the each time chunk'

        return None

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

    def plot_periodogram(self, plot_per_chunk=False, file_name=None, show=True):
        """Plot the light curve data.

        Parameters
        ----------
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
        vis.plot_pd(self.time, self.flux, self.i_chunks, plot_per_chunk=plot_per_chunk, file_name=file_name, show=show)
        return None


class Result:
    """A class to handle analysis results.

    Attributes
    ----------
    target_id: str, optional
        User defined identification number or name for the target under investigation.
    data_id: str, optional
        User defined identification name for the dataset used.
    description: str, optional
        User defined description of the result in question.
    n_param: int
        Number of free parameters in the model.
    bic: float
        Bayesian Information Criterion of the residuals.
    noise_level: float
        The noise level (standard deviation of the residuals).
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves.
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves.
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves.
    c_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the constant for each sector.
    sl_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slope for each sector.
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sine wave.
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitude for each sine wave (these are identical).
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phase for each sine wave.
    c_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the constant for each sector.
    sl_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the slope for each sector.
    f_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the frequency for each sine wave.
    a_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the amplitude for each sine wave (these are identical).
    ph_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the phase for each sine wave.
    passed_sigma: numpy.ndarray[bool]
        Sinusoids that passed the sigma check.
    passed_snr: numpy.ndarray[bool]
        Sinusoids that passed the signal-to-noise check.
    passed_both: numpy.ndarray[bool]
        Sinusoids that passed both checks.
    p_orb: float
        Orbital period.
    p_err: float
        Error in the orbital period.
    p_hdi: numpy.ndarray[2, dtype[float]]
        HDI for the period.
    passed_harmonic: numpy.ndarray[bool]
        Harmonic sinusoids that passed.
    """

    def __init__(self):
        """Initialises the Result object."""
        # descriptive
        self.target_id = ''
        self.data_id = ''
        self.description = ''

        # summary statistics
        self.n_param = -1
        self.bic = -1.
        self.noise_level = -1.

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
        self.p_orb = 0.
        self.p_err = 0.
        self.p_hdi = np.zeros(2)
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
        Result
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

        if config.verbose:
            print(f"Loaded analysis file with target identifier: {result_dict['target_id']}, "
                  f"created on {result_dict['date_time']}. \n"
                  f"Data identifier: {result_dict['data_id']}. Description: {result_dict['description']} \n")

        return instance

    @classmethod
    def load_conditional(cls, file_name):
        """Load a result file into a Result instance only if it exists and if no overwriting.

        Parameters
        ----------
        file_name: str
            File name to load the results from

        Returns
        -------
        Result
            Instance of the Result class with the loaded results.
            Returns empty Result if condition not met.
        """
        # guard for existing file when not overwriting
        if (not os.path.isfile(file_name)) | config.overwrite:
            instance = cls()
            return instance

        # make the Data instance and load the data
        instance = cls.load(file_name)

        return instance

    def save(self, file_name):
        """Save the results to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the results to

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

    def save_as_csv(self, file_name):
        """Write multiple ascii csv files for human readability.

        Parameters
        ----------
        file_name: str
            File name to save the results to

        Returns
        -------
        None
        """
        # file extension
        ext = os.path.splitext(os.path.basename(file_name))[1]

        # linear model parameters
        data = np.column_stack((self.const, self.c_err, self.c_hdi[:, 0], self.c_hdi[:, 1],
                                self.slope, self.sl_err, self.sl_hdi[:, 0], self.sl_hdi[:, 1]))
        hdr = 'const, c_err, c_hdi_l, c_hdi_r, slope, sl_err, sl_hdi_l, sl_hdi_r'
        file_name_lin = file_name.replace(ext, '_linear.csv')
        np.savetxt(file_name_lin, data, delimiter=',', header=hdr)

        # sinusoid model parameters
        data = np.column_stack((self.f_n, self.f_n_err, self.f_n_hdi[:, 0], self.f_n_hdi[:, 1],
                                self.a_n, self.a_n_err, self.a_n_hdi[:, 0], self.a_n_hdi[:, 1],
                                self.ph_n, self.ph_n_err, self.ph_n_hdi[:, 0], self.ph_n_hdi[:, 1],
                                self.passed_sigma, self.passed_snr, self.passed_both, self.passed_harmonic))
        hdr = ('f_n, f_n_err, f_n_hdi_l, f_n_hdi_r, a_n, a_n_err, a_n_hdi_l, a_n_hdi_r, '
               'ph_n, ph_n_err, ph_n_hdi_l, ph_n_hdi_r, passed_sigma, passed_snr, passed_b, passed_h')
        file_name_sin = file_name.replace(ext, '_sinusoid.csv')
        np.savetxt(file_name_sin, data, delimiter=',', header=hdr)

        # period and statistics
        names = ('p_orb', 'p_err', 'p_hdi_l', 'p_hdi_r'  'n_param', 'bic', 'noise_level')
        stats = (self.p_orb, self.p_err, self.p_hdi[0], self.p_hdi[1], self.n_param, self.bic, self.noise_level)
        desc = ['Orbital period', 'Error in the orbital period', 'Left bound HDI of the orbital period',
                'Right bound HDI of the orbital period', 'Number of free parameters',
                'Bayesian Information Criterion of the residuals', 'Standard deviation of the residuals']
        data = np.column_stack((names, stats, desc))
        hdr = f"{self.target_id}, {self.data_id}, Model statistics\nname, value, description"
        file_name_stats = file_name.replace(ext, '_stats.csv')
        np.savetxt(file_name_stats, data, delimiter=',', header=hdr, fmt='%s')

        return None

    def save_conditional(self, file_name):
        """Save a result file only if it doesn't exist or if it exists and if no overwriting.

        Parameters
        ----------
        file_name: str
            File name to load the results from

        Returns
        -------
        None
        """
        if (not os.path.isfile(file_name)) | config.overwrite:
            self.save(file_name)

            # save csv files if configured
            if config.save_ascii:
                self.save_as_csv(file_name)

        return None

    def model_linear(self, time, i_chunks):
        """Returns a piece-wise linear curve for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The model time series of a (set of) straight line(s)
        """
        curve = tsf.linear_curve(time, self.const, self.slope, i_chunks)

        return curve

    def model_sinusoid(self, time):
        """Returns a sum of sine waves for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model time series of a sum of sine waves. Varies around 0.
        """
        curve = tsf.sum_sines(time, self.f_n, self.a_n, self.ph_n)

        return curve


class Pipeline:
    """A class to analyze light curve data.

    Handles the full analysis pipeline of Star Shine.

    Attributes
    ----------
    data: Data object
        Instance of the Data class holding the light curve data.
    result: Result object
        Instance of the Result class holding the parameters of the result.
    save_dir: str
        Root directory where the save files are stored.
    save_subdir: str
        Sub-directory that is made to contain the save files.
    logger: Logger object
        Instance of the logging library.
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
            save_dir = config.save_dir
        self.save_dir = save_dir
        self.save_subdir = f"{self.data.target_id}_analysis"

        # for saving, make a folder if not there yet
        full_dir = os.path.join(save_dir, self.save_subdir)
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)  # create the subdir

        # initialise custom logger
        self.logger = ut.get_custom_logger(full_dir, self.data.target_id, config.verbose)

        # check the input data
        if not isinstance(data, Data):
            self.logger.info("Input `data` should be a Data object.")
        elif len(data.time) == 0:
            self.logger.info("Data object does not contain time series data.")

        return

    def model_linear(self):
        """Returns a piece-wise linear curve for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The model time series of a (set of) straight line(s)
        """
        curve = self.result.model_linear(self.data.time, self.data.i_chunks)

        return curve

    def model_sinusoid(self):
        """Returns a sum of sine waves for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model time series of a sum of sine waves. Varies around 0.
        """
        curve = self.result.model_sinusoid(self.data.time)

        return curve

    def iterative_prewhitening(self):
        """Iterative prewhitening of the input flux time series in the form of sine waves and a piece-wise linear curve.

        After extraction, a final check is done to see whether some frequencies are better removed or groups of
        frequencies are better replaced by one frequency.

        Continues from last results if frequency list is not empty.

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        n_f_init = len(self.result.f_n)
        if config.verbose:
            print(f"{n_f_init} frequencies. Looking for more...")

        # start by looking for more harmonics
        if self.result.p_orb != 0:
            out_a = tsf.extract_harmonics(self.data.time, self.data.flux, self.result.p_orb, self.data.i_chunks,
                                          config.bic_thr, self.result.f_n, self.result.a_n, self.result.ph_n,
                                          verbose=config.verbose)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        # extract all frequencies with the iterative scheme
        out_b = tsf.extract_sinusoids(self.data.time, self.data.flux, self.data.i_chunks, self.result.p_orb,
                                      self.result.f_n, self.result.a_n, self.result.ph_n, bic_thr=config.bic_thr,
                                      snr_thr=config.snr_thr, stop_crit=config.stop_crit, select=config.select,
                                      f0=self.data.f_min, fn=self.data.f_max, fit_each_step=config.optimise_step,
                                      verbose=config.verbose)
        self.result.setter(const=out_b[0], slope=out_b[1], f_n=out_b[2], a_n=out_b[3], ph_n=out_b[4])

        # remove any frequencies that end up not making the statistical cut
        out_c = tsf.reduce_sinusoids(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(const=out_c[0], slope=out_c[1], f_n=out_c[2], a_n=out_c[3], ph_n=out_c[4])

        # select frequencies based on some significance criteria
        out_d = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope,
                                     self.result.f_n, self.result.a_n, self.result.ph_n, self.data.i_chunks,
                                     verbose=config.verbose)
        self.result.setter(passed_sigma=out_d[0], passed_snr=out_d[1], passed_both=out_d[2], passed_harmonic=out_d[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3], ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Iterative prewhitening results.')

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            print(f"\033[1;32;48mExtraction of sinusoids complete.\033[0m")
            print(f"\033[0;32;48m{len(self.result.f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. "
                  f"Time taken: {t_b - t_a:1.1f}s\033[0m\n")

        # log if nothing found
        if len(self.result.f_n) == 0:
            self.logger.info("No frequencies found.")

        return self.result

    def optimise_sinusoid(self):
        """Optimise the parameters of the sinusoid and linear model

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        if config.verbose:
            print("Starting multi-sinusoid NL-LS optimisation.")

        # use the chosen optimisation method
        inf_data, par_mean, par_hdi = None, None, None
        if config.optimise == 'fitter':
            par_mean = tsfit.fit_multi_sinusoid_per_group(self.data.time, self.data.flux, self.result.const,
                                                          self.result.slope, self.result.f_n, self.result.a_n,
                                                          self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        else:
            # make model including everything to calculate noise level
            resid = self.data.flux - self.model_linear() - self.model_sinusoid()
            n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
            noise_level = ut.std_unb(resid, len(self.data.time) - n_param)

            # formal linear and sinusoid parameter errors
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(self.data.time, resid,
                                                                                 self.data.flux_err, self.result.a_n,
                                                                                 self.data.i_chunks)

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            output = mcf.sample_sinusoid(self.data.time, self.data.flux, self.result.const, self.result.slope,
                                         f_n, a_n, ph_n,  self.result.c_err, self.result.sl_err, f_n_err, a_n_err,
                                         ph_n_err, noise_level, self.data.i_chunks, verbose=config.verbose)
            inf_data, par_mean, par_hdi = output

        self.result.setter(const=par_mean[0], slope=par_mean[1], f_n=par_mean[2], a_n=par_mean[3], ph_n=par_mean[4],
                           c_hdi=par_hdi[0], sl_hdi=par_hdi[1], f_n_hdi=par_hdi[2], a_n_hdi=par_hdi[3],
                           ph_n_hdi=par_hdi[4])

        # select frequencies based on some significance criteria
        out_b = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, 0, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_b[0], passed_snr=out_b[1], passed_both=out_b[2], passed_harmonic=out_b[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3], ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Multi-sinusoid NL-LS optimisation results.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            print(f"\033[1;32;48mOptimisation of sinusoids complete.\033[0m")
            print(f"\033[0;32;48m{len(self.result.f_n)} frequencies, {self.result.n_param} free parameters, "
                  f"BIC: {self.result.bic:1.2f}. Time taken: {t_b - t_a:1.1f}s\033[0m\n")

        return self.result

    def couple_harmonics(self):
        """Find the orbital period and couple harmonic frequencies to the orbital period

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results

        Notes
        -----
        Performs a global period search, if the period is unknown.
        If a period is given, it is locally refined for better performance.

        Removes theoretical harmonic candidate frequencies within the frequency
        resolution, then extracts a single harmonic at the theoretical location.

        Removes any frequencies that end up not making the statistical cut.
        """
        t_a = systime.time()
        if config.verbose:
            print("Coupling the harmonic frequencies to the orbital frequency...")

        # if given, the input p_orb is refined locally, otherwise the period is searched for globally
        if self.data.p_orb == 0:
            self.result.p_orb = tsf.find_orbital_period(self.data.time, self.data.flux, self.result.f_n)
        else:
            self.result.p_orb = tsf.refine_orbital_period(self.data.p_orb, self.data.time, self.result.f_n)

        # if time series too short, or no harmonics found, log and warn and maybe cut off the analysis
        freq_res = 1.5 / self.data.t_tot  # Rayleigh criterion
        harmonics, harmonic_n = af.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=freq_res / 2)
        if (self.data.t_tot / self.result.p_orb > 1.1) & (len(harmonics) > 1):
            # couple the harmonics to the period. likely removes more frequencies that need re-extracting
            out_a = tsf.fix_harmonic_frequency(self.data.time, self.data.flux, self.result.p_orb,  self.result.const,
                                               self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                               self.data.i_chunks, verbose=config.verbose)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        # remove any frequencies that end up not making the statistical cut
        out_b = tsf.reduce_sinusoids(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(const=out_b[0], slope=out_b[1], f_n=out_b[2], a_n=out_b[3], ph_n=out_b[4])

        # select frequencies based on some significance criteria
        out_c = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope, self.result.f_n, self.result.a_n,
                                     self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_c[0], passed_snr=out_c[1], passed_both=out_c[2], passed_harmonic=out_c[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        harmonics, harmonic_n = af.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
        n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_d = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_d[0], sl_err=out_d[1], f_n_err=out_d[2], a_n_err=out_d[3], ph_n_err=out_d[4])
        p_err, _, _ = af.linear_regression_uncertainty(self.result.p_orb, self.data.t_tot,
                                                       sigma_t=self.data.t_int / 2)
        self.result.setter(p_orb=np.array([self.result.p_orb, p_err, 0, 0]))

        # set the result description
        self.result.setter(description='Harmonic frequencies coupled to the orbital period.')

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            rnd_p_orb = max(ut.decimal_figures(p_err, 2), ut.decimal_figures(self.result.p_orb, 2))
            print(f"\033[1;32;48mOrbital harmonic frequencies coupled.\033[0m")
            print(f"\033[0;32;48mp_orb: {self.result.p_orb:.{rnd_p_orb}f} (+-{p_err:.{rnd_p_orb}f}), \n"
                  f"{len(self.result.f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. "
                  f"Time taken: {t_b - t_a:1.1f}s\033[0m\n")

        # log if short time span or few harmonics
        if self.data.t_tot / self.result.p_orb < 1.1:
            self.logger.info(f"Period over time-base is less than two: {self.data.t_tot / self.result.p_orb}; "
                             f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")
        elif len(harmonics) < 2:
            self.logger.info(f"Not enough harmonics found: {len(harmonics)}; "
                             f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")

        return self.result

    def optimise_sinusoid_h(self):
        """Optimise the parameters of the sinusoid and linear model with coupled harmonics

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        if config.verbose:
            print("Starting multi-sine NL-LS optimisation with harmonics.")

        # use the chosen optimisation method
        par_hdi = np.zeros((6, 2))
        if config.optimise == 'fitter':
            par_mean = tsfit.fit_multi_sinusoid_harmonics_per_group(self.data.time, self.data.flux, self.result.p_orb,
                                                                    self.result.const, self.result.slope,
                                                                    self.result.f_n, self.result.a_n, self.result.ph_n,
                                                                    self.data.i_chunks, verbose=config.verbose)
        else:
            # make model including everything to calculate noise level
            resid = self.data.flux - self.model_linear() - self.model_sinusoid()
            harmonics, harmonic_n = af.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
            n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
            noise_level = ut.std_unb(resid, len(self.data.time) - n_param)

            # formal linear and sinusoid parameter errors
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(self.data.time, resid,
                                                                                 self.data.flux_err, self.result.a_n,
                                                                                 self.data.i_chunks)
            p_err, _, _ = af.linear_regression_uncertainty(self.result.p_orb, self.data.t_tot,
                                                           sigma_t=self.data.t_int/2)

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            output = mcf.sample_sinusoid_h(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                           self.result.slope, f_n, a_n, ph_n, self.result.p_err, self.result.c_err,
                                           self.result.sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                                           self.data.i_chunks, verbose=config.verbose)
            inf_data, par_mean, par_hdi = output

        self.result.setter(p_orb=par_mean[0], const=par_mean[1], slope=par_mean[2], f_n=par_mean[3], a_n=par_mean[4],
                           ph_n=par_mean[5], p_hdi=par_hdi[0], c_hdi=par_hdi[1], sl_hdi=par_hdi[2], f_n_hdi=par_hdi[3],
                           a_n_hdi=par_hdi[4], ph_n_hdi=par_hdi[5])

        # select frequencies based on some significance criteria
        out_b = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope, self.result.f_n, self.result.a_n,
                                     self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_b[0], passed_snr=out_b[1], passed_both=out_b[2], passed_harmonic=out_b[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        harmonics, harmonic_n = af.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
        n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        p_err, _, _ = af.linear_regression_uncertainty(self.result.p_orb, self.data.t_tot, sigma_t=self.data.t_int/2)
        self.result.setter(p_err=p_err, c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3],
                           ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Multi-sine NL-LS optimisation results with coupled harmonics.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            rnd_p_orb = max(ut.decimal_figures(self.result.p_err, 2),
                            ut.decimal_figures(self.result.p_orb, 2))
            print(f"\033[1;32;48mOptimisation with coupled harmonics complete.\033[0m")
            print(f"\033[0;32;48mp_orb: {self.result.p_orb:.{rnd_p_orb}f} (+-{self.result.p_err:.{rnd_p_orb}f}), \n"
                  f"{len(self.result.f_n)} frequencies, {self.result.n_param} free parameters, "
                  f"BIC: {self.result.bic:1.2f}. Time taken: {t_b - t_a:1.1f}s\033[0m\n")

        return self.result

    def run(self):
        """Run the analysis pipeline on the given data.

        Runs a predefined sequence of analysis steps. Stops at the stage defined in the configuration file.
        Files are saved automatically at the end of each stage, taking into account the desired behaviour for
        overwriting.

        The followed recipe is:

        1) Extract all frequencies
            We start by extracting the frequency with the highest amplitude one by one,
            directly from the Lomb-Scargle periodogram until the BIC does not significantly
            improve anymore. This step involves a final cleanup of the frequencies.

        2) Multi-sinusoid NL-LS optimisation
            The sinusoid parameters are optimised with a non-linear least-squares method,
            using groups of frequencies to limit the number of free parameters.

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

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        # this list determines which analysis steps are taken and in what order
        step_names = ['iterative_prewhitening', 'optimise_sinusoid', 'couple_harmonics', 'iterative_prewhitening',
                      'optimise_sinusoid_h']

        # run steps until config number
        if config.stop_at_stage != 0:
            step_names = step_names[:config.stop_at_stage]

        # tag the start of the analysis
        t_a = systime.time()
        self.logger.info("Start of analysis")

        # run this sequence for each analysis step of the pipeline
        for step in range(len(step_names)):
            file_name = os.path.join(self.save_dir, self.save_subdir, f"{self.data.target_id}_result_{step + 1}.hdf5")

            # Load existing file if not overwriting
            self.result = Result.load_conditional(file_name)  # returns empty Result if no file

            # if existing results were loaded, go to the next step
            if self.result.target_id != '':
                continue

            # do the analysis step
            analysis_step = getattr(self, step_names[step])
            analysis_step()

            # save the results if conditions are met
            self.result.save_conditional(file_name)

        # final message and timing
        t_b = systime.time()
        self.logger.info(f"End of analysis. Total time elapsed: {t_b - t_a:1.1f}s.")  # info to save to log

        return self.result
