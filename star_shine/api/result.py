"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the result class for handling the analysis results.

Code written by: Luc IJspeert
"""
import os
import numpy as np

from star_shine.core import utility as ut
from star_shine.core import io
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class Result:
    """A class to handle analysis results.

    Attributes
    ----------
    target_id: str
        User defined identification number or name for the target under investigation.
    data_id: str
        User defined identification name for the dataset used.
    n_param: int
        Number of free parameters in the model.
    bic: float
        Bayesian Information Criterion of the residuals.
    noise_level: float
        The noise level (standard deviation of the residuals).
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    const_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the constant for each time chunk.
    const_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the constant for each time chunk.
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    slope_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slope for each time chunk.
    slope_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the slope for each time chunk.
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sinusoids.
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sinusoid.
    f_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the frequency for each sinusoid.
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids.
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitude for each sinusoid (these are identical).
    a_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the amplitude for each sinusoid.
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sinusoids.
    ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phase for each sinusoid.
    ph_n_hdi: numpy.ndarray[Any, dtype[float]]
        HDI bounds for the phase for each sinusoid.
    passing_sigma: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the sigma check.
    passing_snr: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the signal-to-noise check.
    h_base: numpy.ndarray[Any, dtype[int]]
        Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
    f_h_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each harmonic sinusoid.
    passing_harmonic: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the harmonic check.
    """

    def __init__(self):
        """Initialises the Result object."""
        # descriptive
        self.target_id = ''
        self.data_id = ''

        # summary statistics
        self.n_param = -1
        self.bic = -1.
        self.noise_level = -1.

        # attribute lists per model
        self.linear_property_list = ['const', 'const_err', 'const_hdi', 'slope', 'slope_err', 'slope_hdi']
        self.sinusoid_property_list = ['f_n', 'f_n_err', 'f_n_hdi', 'a_n', 'a_n_err', 'a_n_hdi',
                                       'ph_n', 'ph_n_err', 'ph_n_hdi', 'passing_sigma', 'passing_snr',
                                       'h_base', 'h_mult', 'f_h_err', 'passing_harmonic']

        # linear model parameters
        # y-intercepts
        self.const = np.zeros((0,))
        self.const_err = np.zeros((0,))
        self.const_hdi = np.zeros((0, 2))
        # slopes
        self.slope = np.zeros((0,))
        self.slope_err = np.zeros((0,))
        self.slope_hdi = np.zeros((0, 2))

        # sinusoid model parameters
        # frequencies
        self.f_n = np.zeros((0,))
        self.f_n_err = np.zeros((0,))
        self.f_n_hdi = np.zeros((0, 2))
        # amplitudes
        self.a_n = np.zeros((0,))
        self.a_n_err = np.zeros((0,))
        self.a_n_hdi = np.zeros((0, 2))
        # phases
        self.ph_n = np.zeros((0,))
        self.ph_n_err = np.zeros((0,))
        self.ph_n_hdi = np.zeros((0, 2))
        # passing criteria
        self.passing_sigma = np.zeros((0,), dtype=bool)
        self.passing_snr = np.zeros((0,), dtype=bool)

        # harmonic model
        self.h_base = np.zeros((0,), dtype=int)
        self.h_mult = np.zeros((0,), dtype=int)
        self.f_h_err = np.zeros((0,))
        # passing criteria
        self.passing_harmonic = np.zeros((0,), dtype=bool)

        return

    def get_dict(self):
        """Make a dictionary of the attributes.

        Primarily for saving to file.

        Returns
        -------
        dict
            Dictionary of the result attributes and fields
        """
        # make a dictionary of the fields to be saved
        result_dict = dict()
        result_dict['target_id'] = self.target_id
        result_dict['data_id'] = self.data_id
        result_dict['date_time'] = ut.datetime_formatted()

        result_dict['n_param'] = self.n_param  # number of free parameters
        result_dict['bic'] = self.bic  # Bayesian Information Criterion of the residuals
        result_dict['noise_level'] = self.noise_level  # standard deviation of the residuals

        # the linear model
        for key in self.linear_property_list:
            result_dict[key] = getattr(self, key)

        # the sinusoid model
        for key in self.sinusoid_property_list:
            result_dict[key] = getattr(self, key)

        return result_dict

    def from_dict(self, **kwargs):
        """Fill in the attributes with results in a dictionary (or keyword arguments).

        Parameters
        ----------
        kwargs:
            Accepts any of the class attributes as keyword input and sets them accordingly.
        """
        # set any attribute that exists if it is in the kwargs
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return None

    def from_time_series_model(self, ts_model, target_id=None, data_id=None):
        """Fill in the Result attributes with results from a TimeSeriesModel.

        Parameters
        ----------
        ts_model: tms.TimeSeriesModel
            Instance of TimeSeriesModel containing the time series and model parameters.
        target_id: str
            User defined identification number or name for the target under investigation.
        data_id: str
            User defined identification name for the dataset used.
        """
        # descriptive
        self.target_id = target_id
        self.data_id = data_id

        # summary statistics
        self.n_param = ts_model.n_param
        self.bic = ts_model.bic()
        self.noise_level = ut.std_unb(ts_model.residual(), ts_model.n_time - ts_model.n_param)

        # linear model parameters
        for key in self.linear_property_list:
            if '_hdi' not in key:  # avoid hdi for now
                setattr(self, key, getattr(ts_model.linear, key))

        # sinusoid model parameters
        ts_model.remove_excluded()  # clean up before transfer
        for key in self.sinusoid_property_list:
            if '_hdi' not in key:  # avoid hdi for now
                setattr(self, key, getattr(ts_model.sinusoid, key))

        return None

    @classmethod
    def load(cls, file_name, h5py_file_kwargs=None, logger=None):
        """Load a result file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        h5py_file_kwargs: dict, optional
            Keyword arguments for opening the h5py file.
            Example: {'locking': False}, for a drive that does not support locking.
        logger: logging.Logger, optional
            Instance of the logging library.

        Returns
        -------
        Result
            Instance of the Result class with the loaded results.
        """
        # guard for not existing file
        if not os.path.isfile(file_name):
            instance = cls()
            return instance

        # add everything to a dict
        result_dict = io.load_result_hdf5(file_name, h5py_file_kwargs=h5py_file_kwargs)

        # initiate the Results instance and fill in the results
        instance = cls()
        instance.from_dict(**result_dict)

        if logger is not None:
            logger.info(f"Loaded result file with target identifier: {result_dict['target_id']}, "
                        f"created on {result_dict['date_time']}. Data identifier: {result_dict['data_id']}.")

        return instance

    def to_time_series_model(self, ts_model):
        """Copy the parameters into an existing TimeSeriesModel instance.

        Parameters
        ----------
        ts_model: tms.TimeSeriesModel
            Instance of TimeSeriesModel containing the time series and model parameters.

        Returns
        -------
        tms.TimeSeriesModel
            Instance of TimeSeriesModel containing the time series and model parameters.
        """
        # linear model
        ts_model.set_linear_model(self.const, self.slope)

        # set the errors
        ts_model.linear._const_err = self.const_err
        ts_model.linear._slope_err = self.slope_err

        # update numbers
        ts_model.linear.update_n()

        # sinusoid model
        ts_model.set_sinusoids(self.f_n, self.a_n, self.ph_n, self.h_base, self.h_mult)

        # set the errors
        ts_model.sinusoid._f_n_err = self.f_n_err
        ts_model.sinusoid._a_n_err = self.a_n_err
        ts_model.sinusoid._ph_n_err = self.ph_n_err
        ts_model.sinusoid._f_h_err = self.f_h_err

        # set the criterion passing masks
        ts_model.sinusoid._passing_sigma = self.passing_sigma
        ts_model.sinusoid._passing_snr = self.passing_snr
        ts_model.sinusoid._passing_harmonic = self.passing_harmonic

        # update numbers
        ts_model.sinusoid.update_n()

        return ts_model

    def save(self, file_name):
        """Save the results to a file in hdf5 format.

        Parameters
        ----------
        file_name: str
            File name to save the results to
        """
        # get a dictionary of the fields to be saved
        result_dict = self.get_dict()

        # io module handles writing to file
        io.save_result_hdf5(file_name, result_dict)

        # save csv files if configured
        if config.save_ascii:
            self.save_as_csv(file_name)

        return None

    def save_as_csv(self, file_name):
        """Write multiple ascii csv files for human readability.

        Parameters
        ----------
        file_name: str
            File name to save the results to
        """
        # get a dictionary of the fields to be saved
        result_dict = self.get_dict()

        # io module handles writing to file
        io.save_result_csv(file_name, result_dict)

        return None
