"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the result class for handling the analysis results.

Code written by: Luc IJspeert
"""
import os
import datetime
import h5py
import numpy as np

from star_shine.core import timeseries_functions as tsf
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


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
