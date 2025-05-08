"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the data class for handling the user defined data to analyse.

Code written by: Luc IJspeert
"""
import os
import numpy as np

from star_shine.core import periodogram as pdg
from star_shine.core import visualisation as vis
from star_shine.core import utility as ut
from star_shine.core import io
from star_shine.config.helpers import get_config
from star_shine.config import data_properties as dp


# load configuration
config = get_config()


class Data:
    """A class to handle light curve data.

    Attributes
    ----------
    file_list: list[str]
        List of ascii light curve files or (TESS) data product '.fits' files.
    data_dir: str
        Root directory where the data files to be analysed are located.
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
    p_orb: float
        The orbital period. Set to 0 to search for the best period.
        If the orbital period is known with certainty beforehand, it can
        be provided as initial value and no new period will be searched.
    t_tot: float
        Total time base of observations.
    t_mean: float
        Time reference (zero) point of the full light curve.
    t_mean_chunk: numpy.ndarray[Any, dtype[float]]
        Time reference (zero) point per chunk.
    t_step: float
        Median time step of observations.
    snr_threshold: float
        Signal-to-noise threshold for the acceptance of sinusoids.
    f_nyquist: float
        Nyquist frequency (max f) for extraction and periodograms.
    f_resolution: float
        Frequency resolution for extraction and periodograms.
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
        self._time = np.zeros((0,))
        self._time_changed = True  # Flag to indicate if time has changed
        self._flux = np.zeros((0,))
        self._flux_changed = True   # Flag to indicate if flux has changed
        self.flux_err = np.zeros((0,))
        self.i_chunks = np.zeros((0, 2), dtype=int)
        self.flux_counts_medians = np.zeros((0,))

        # Orbital period
        self.p_orb = 0.

        # independent data properties
        self.t_tot = 0.
        self.t_mean = 0.
        self.t_mean_chunk = np.zeros((0,))
        self.t_step = 0.

        # data properties relying on config
        self.snr_threshold = 0.
        self.f_nyquist = 0.
        self.f_resolution = 0.

        # defaults for periodograms
        self.pd_f0 = 0.
        self.pd_df = 0.
        self.pd_fn = 0.

        # cache
        self._periodogram_f = np.zeros((0,))
        self._periodogram_a = np.zeros((0,))

        return

    def _check_file_existence(self, logger=None):
        """Checks whether the given file(s) exist.

        Removes missing files from the file list

        Parameters
        ----------
        logger: logging.Logger, optional
            Instance of the logging library.
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

            if logger is not None:
                logger.warning(message)

            # remove the files
            for i in missing:
                del self.file_list[i]

        return None

    @property
    def time(self):
        """Getter for the time attribute."""
        return self._time

    @time.setter
    def time(self, value):
        """Setter for the time attribute with flag update."""
        self._time = value
        self._time_changed = True

    @property
    def flux(self):
        """Getter for the flux attribute."""
        return self._flux

    @flux.setter
    def flux(self, value):
        """Setter for the flux attribute with flag update."""
        self._flux = value
        self._flux_changed = True

    def setter(self, **kwargs):
        """Fill in the attributes with data.

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly
        """
        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    def get_dict(self):
        """Make a dictionary of the attributes.

        Primarily for saving to file.

        Returns
        -------
        data_dict: dict
            Dictionary of the data attributes and fields
        """
        # make a dictionary of the fields to be saved
        data_dict = {}
        data_dict['target_id'] = self.target_id
        data_dict['data_id'] = self.data_id
        data_dict['description'] = 'Star Shine data file'
        data_dict['date_time'] = ut.datetime_formatted()

        # original list of files
        data_dict['data_dir'] = self.data_dir
        data_dict['file_list'] = self.file_list

        # Orbital period
        data_dict['p_orb'] = self.p_orb

        # summary statistics
        data_dict['t_tot'] = self.t_tot
        data_dict['t_mean'] = self.t_mean
        data_dict['t_mean_chunk'] = self.t_mean_chunk
        data_dict['t_step'] = self.t_step

        # the time series data
        data_dict['time'] = self.time
        data_dict['flux'] = self.flux
        data_dict['flux_err'] = self.flux_err

        # additional information
        data_dict['i_chunks'] = self.i_chunks
        data_dict['flux_counts_medians'] = self.flux_counts_medians

        return data_dict

    @classmethod
    def load_data(cls, file_list, data_dir='', target_id='', data_id='', logger=None):
        """Load light curve data from the file list.

        Parameters
        ----------
        file_list: list[str]
            List of ascii light curve files or (TESS) data product '.fits' files. Exclude the path given to 'data_dir'.
            If only one file is given, its file name is used for target_id (if 'none').
        data_dir: str, optional
            Root directory where the data files to be analysed are located. Added to the file name.
            If empty, it is loaded from config.
        target_id: str, optional
            User defined identification number or name for the target under investigation. If empty, the file name
            of the first file in file_list is used.
        data_id: str, optional
            User defined identification name for the dataset used.
        logger: logging.Logger, optional
            Instance of the logging library.

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
            if logger is not None:
                logger.warning("Empty file list provided.")
            return cls()

        # Check if the file(s) exist(s)
        instance._check_file_existence(logger=logger)
        if len(instance.file_list) == 0:
            if logger is not None:
                logger.warning("No existing files in file list")
            return cls()

        # set IDs
        if target_id == '':
            target_id = os.path.splitext(os.path.basename(file_list[0]))[0]  # file name is used as identifier
        instance.setter(target_id=target_id, data_id=data_id)

        # add data_dir for loading files, if not None
        if instance.data_dir == '':
            file_list_dir = instance.file_list
        else:
            file_list_dir = [os.path.join(instance.data_dir, file) for file in instance.file_list]

        # load the data from the list of files
        lc_data = io.load_light_curve(file_list_dir, apply_flags=config.apply_q_flags)
        instance.setter(time=lc_data[0], flux=lc_data[1], flux_err=lc_data[2], i_chunks=lc_data[3], medians=lc_data[4])

        # check for overlapping time stamps
        if np.any(np.diff(instance.time) <= 0) & (logger is not None):
            logger.warning("The time array chunks include overlap.")

        # calculate properties of the data
        instance.update_properties()

        if logger is not None:
            logger.info(f"Loaded data from external file(s).")

        return instance

    @classmethod
    def load(cls, file_name, data_dir='', h5py_file_kwargs=None, logger=None):
        """Load a data file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the data from
        data_dir: str, optional
            Root directory where the data files to be analysed are located. Added to the file name.
            If empty, it is loaded from config.
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.
        logger: logging.Logger, optional
            Instance of the logging library.

        Returns
        -------
        Data
            Instance of the Data class with the loaded data.
        """
        # initiate the Data instance
        instance = cls()

        # set the file list and data directory
        if data_dir == '':
            data_dir = config.data_dir
        instance.setter(data_dir=data_dir)

        # add data_dir for loading file, if not None
        if instance.data_dir != '':
            file_name = os.path.join(instance.data_dir, file_name)

        # io module handles opening the file
        data_dict = io.load_data_hdf5(file_name, h5py_file_kwargs=h5py_file_kwargs)

        # fill in the data
        instance.setter(**data_dict)

        if logger is not None:
            logger.info(f"Loaded data file with target identifier: {data_dict['target_id']}, "
                        f"created on {data_dict['date_time']}. Data identifier: {data_dict['data_id']}.")

        return instance

    def save(self, file_name):
        """Save the data to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the data to
        """
        # make a dictionary of the fields to be saved
        data_dict = self.get_dict()

        # io module handles writing to file
        io.save_data_hdf5(file_name, data_dict)

        return None

    def update_properties(self):
        """Calculate the properties of the data and fill them in.

        Running this function again will re-evaluate the properties. This can be useful if the configuration changed.
        """
        # set independent data properties
        self.t_tot = np.ptp(self.time)
        self.t_mean = np.mean(self.time)
        self.t_mean_chunk = np.array([np.mean(self.time[ch[0]:ch[1]]) for ch in self.i_chunks])
        self.t_step = np.median(np.diff(self.time))

        # set data properties relying on config
        self.snr_threshold = dp.signal_to_noise_threshold(self.time)
        self.f_nyquist = dp.nyquist_frequency(self.time)
        self.f_resolution = dp.frequency_resolution(self.time)

        # set periodogram defaults
        self.pd_f0 = 0.01 / self.t_tot  # lower than T/100 no good
        self.pd_df = 0.1 / self.t_tot  # default frequency sampling is about 1/10 of frequency resolution
        self.pd_fn = 1 / (2 * np.min(self.time[1:] - self.time[:-1]))

        return None

    def periodogram(self):
        """Compute the Lomb-Scargle periodogram of the time series

        Returns
        -------
        tuple
            Contains the frequencies numpy.ndarray[Any, dtype[float]]
            and the spectrum numpy.ndarray[Any, dtype[float]]

        Notes
        -----
        Return values are cached until time or flux is updated.
        """
        if self._time_changed or self._flux_changed:
            # calculate the periodogram
            f, a = pdg.scargle_parallel(self.time, self.flux, f0=-1, fn=-1, df=-1, norm='amplitude')
            self._periodogram_f, self._periodogram_a = f, a
            # Reset flags after computation
            self._time_changed, self._flux_changed = False, False
        else:
            # return the cached values
            f, a = self._periodogram_f, self._periodogram_a

        return f, a

    def plot_light_curve(self, file_name=None, show=True):
        """Plot the light curve data.

        Parameters
        ----------
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot
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
        """
        vis.plot_pd(self.time, self.flux, self.i_chunks, plot_per_chunk=plot_per_chunk, file_name=file_name, show=show)

        return None
