"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the data class for handling the user defined data to analyse.

Code written by: Luc IJspeert
"""
import os
import numpy as np

from star_shine.core import time_series as tms, utility as ut, visualisation as vis
from star_shine.core import io
from star_shine.config.helpers import get_config


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
    flux_counts_medians: numpy.ndarray[Any, dtype[float]]
        Median flux counts per chunk.
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

        # the time series is stored here
        self.time_series = None

        # additional time series properties not in time_series
        self.flux_counts_medians = np.zeros((0,))

        return

    def __repr__(self):
        return (f"Data(target_id={self.target_id!r}, "
                f"data_id={self.data_id!r}, "
                f"data_dir={self.data_dir!r}, "
                f"file_list={self.file_list!r})")

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

    def get_dict(self):
        """Make a dictionary of the attributes.

        Primarily for saving to file.

        Returns
        -------
        data_dict: dict
            Dictionary of the data attributes and fields
        """
        # make a dictionary of the fields to be saved
        data_dict = dict()
        data_dict['target_id'] = self.target_id
        data_dict['data_id'] = self.data_id
        data_dict['date_time'] = ut.datetime_formatted()

        # original list of files
        data_dict['data_dir'] = self.data_dir
        data_dict['file_list'] = self.file_list

        # summary statistics
        data_dict['t_tot'] = self.time_series.t_tot
        data_dict['t_mean'] = self.time_series.t_mean
        data_dict['t_mean_chunk'] = self.time_series.t_mean_chunk
        data_dict['t_step'] = self.time_series.t_step

        # the time series data
        data_dict['time'] = self.time_series.time
        data_dict['flux'] = self.time_series.flux
        data_dict['flux_err'] = self.time_series.flux_err

        # additional information
        data_dict['i_chunks'] = self.time_series.i_chunks
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
        instance.file_list = file_list
        instance.data_dir = data_dir

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
        instance.target_id = target_id
        instance.data_id = data_id

        # add data_dir for loading files, if not None
        if instance.data_dir == '':
            file_list_dir = instance.file_list
        else:
            file_list_dir = [os.path.join(instance.data_dir, file) for file in instance.file_list]

        # load the data from the list of files
        lc_data = io.load_light_curve(file_list_dir, apply_flags=config.apply_q_flags)

        # make a TimeSeries instance
        instance.time_series = tms.TimeSeries(lc_data[0], lc_data[1], lc_data[2], lc_data[3])
        instance.flux_counts_medians = lc_data[4]

        # check for overlapping time stamps
        if np.any(np.diff(instance.time_series.time) <= 0) & (logger is not None):
            logger.warning("The time array chunks include overlap.")

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
        instance.data_dir = data_dir

        # add data_dir for loading file, if not None
        if instance.data_dir != '':
            file_name = os.path.join(instance.data_dir, file_name)

        # io module handles opening the file
        data_dict = io.load_data_hdf5(file_name, h5py_file_kwargs=h5py_file_kwargs)

        # general properties
        instance.target_id = data_dict['target_id']
        instance.data_id = data_dict['data_id']
        instance.file_list = data_dict['file_list']

        # make a TimeSeries instance
        instance.time_series = tms.TimeSeries(data_dict['time'], data_dict['flux'], data_dict['flux'],
                                              data_dict['i_chunks'])
        instance.flux_counts_medians = data_dict['flux_counts_medians']

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

    def plot_time_series(self, file_name=None, show=True):
        """Plot the time series data.

        Parameters
        ----------
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot
        """
        vis.plot_lc(self.time_series.time, self.time_series.flux, self.time_series.flux_err, self.time_series.i_chunks,
                    file_name=file_name, show=show)

        return None

    def plot_periodogram(self, plot_per_chunk=False, file_name=None, show=True):
        """Plot the periodogram of the time series.

        Parameters
        ----------
        plot_per_chunk: bool
            If True, plot the periodogram of each separate time chunk in one plot.
        file_name: str, optional
            File path to save the plot
        show: bool, optional
            If True, display the plot
        """
        vis.plot_pd(self.time_series.time, self.time_series.flux, self.time_series.i_chunks,
                    plot_per_chunk=plot_per_chunk, file_name=file_name, show=show)

        return None
