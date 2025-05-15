"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the result class for handling the sinusoid model. Includes a piece-wise linear trend
and harmonics.

Code written by: Luc IJspeert
"""
from copy import deepcopy
import numpy as np

from star_shine.core import timeseries as ts, goodness_of_fit as gof, frequency_sets as frs
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class LinearModel:
    """This class handles the linear model.

    Attributes
    ----------
    _const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    _slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    _linear_model: numpy.ndarray[Any, dtype[float]]
        Time series model of the piece-wise linear curve.
    """

    def __init__(self, n_time, n_chunks):
        """Initialises the Result object.

        Parameters
        ----------
        n_time: int
            Number of points in the time series.
        n_chunks: int
            Number of chunks in the time series.
        """
        # linear model parameters
        self._const = np.zeros((n_chunks,))  # y-intercepts
        self._slope = np.zeros((n_chunks,))  # slopes

        # number of time chunks
        self.n_chunks = n_chunks  # does not change
        # number of parameters
        self.n_param = 2 * self.n_chunks  # does not change

        # current model
        self._linear_model = np.zeros((n_time,))

    @property
    def const(self):
        """Get the current model constants.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The y-intercepts of a piece-wise linear curve.
        """
        return self._const.copy()

    @property
    def slope(self):
        """Get the current model slopes.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The slopes of a piece-wise linear curve.
        """
        return self._slope.copy()

    @property
    def linear_model(self):
        """Get the current linear model.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Time series model of the piece-wise linear curve.
        """
        return self._linear_model.copy()

    def get_linear_parameters(self):
        """Get a copy of the current linear parameters.

        Returns
        -------
        tuple
            Consisting of two numpy.ndarray[Any, dtype[float]] for const, slope.
        """
        return self.const, self.slope

    def set_linear_model(self, time, const_new, slope_new, i_chunks):
        """Set the linear model according to the new parameters.

        Parameters
        ----------
        time: np.ndarray
            Timestamps of the time series.
        const_new: np.ndarray
            New y-intercepts.
        slope_new: np.ndarray
            New slopes.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # make the new model
        self._linear_model = ts.linear_curve(time, const_new, slope_new, i_chunks)

        # set the parameters
        self._const = const_new
        self._slope = slope_new

        return

    def update_linear_model(self, time, residual, i_chunks):
        """Update the linear model using the residual flux.

        Parameters
        ----------
        time: np.ndarray
            Timestamps of the time series.
        residual: np.ndarray
            Residual flux.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # get new parameters
        const_new, slope_new = ts.linear_pars(time, residual, i_chunks)

        # set the new parameters and model
        self.set_linear_model(time, const_new, slope_new, i_chunks)
        return


class SinusoidModel:
    """This class handles the sinusoid model.

    Attributes
    ----------
    _f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves.
    _a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves.
    _ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves.
    _p_orb: float
        Orbital period.
    _sinusoid_model: numpy.ndarray[Any, dtype[float]]
        Time series model of the sinusoids.
    """

    def __init__(self, n_time):
        """Initialises the Result object.

        Parameters
        ----------
        n_time: int
            Number of points in the time series.
        """
        # sinusoid model parameters
        self._f_n = np.zeros((0,))  # frequencies
        self._a_n = np.zeros((0,))  # amplitudes
        self._ph_n = np.zeros((0,))  # phases

        # harmonic model parameters
        self._p_orb = 0.

        # number of sinusoids
        self.n_sin = 0
        # number of harmonics
        self.n_harm = 0

        # current model
        self._sinusoid_model = np.zeros((0,))

    @property
    def f_n(self):
        """Get a copy of the current model frequencies.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sine waves.
        """
        return self._f_n.copy()

    @property
    def a_n(self):
        """Get a copy of the current model amplitudes.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves.
        """
        return self._a_n.copy()

    @property
    def ph_n(self):
        """Get a copy of the current model phases.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves.
        """
        return self._ph_n.copy()

    @property
    def p_orb(self):
        """Get the current model period.

        Returns
        -------
        float
            The orbital period of the harmonic model.
        """
        return self._p_orb

    @property
    def sinusoid_model(self):
        """Get a copy of the current sinusoid model.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Time series model of the sinusoids.
        """
        return self._sinusoid_model.copy()

    @property
    def n_param(self):
        """Get the number of parameters of the model."""
        return int(self.n_harm > 0) + 2 * self.n_harm + 3 * self.n_sin

    def get_sinusoid_parameters(self):
        """Get a copy of the current sinusoid parameters.

        Returns
        -------
        tuple
            Consisting of three numpy.ndarray[Any, dtype[float]] for f_n, a_n, ph_n.
        """
        return self.f_n, self.a_n, self.ph_n

    def get_harmonics(self):
        """Get a list of indices of the harmonics in the model"""
        harmonics, harmonic_n = frs.find_harmonics_from_pattern(self._f_n, self._p_orb, f_tol=1e-9)

        return harmonics, harmonic_n

    def set_sinusoids(self, time, f_n_new, a_n_new, ph_n_new):
        """Set the current sinusoid model with the new parameters.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sine waves.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves.
        """
        # make the new model
        self._sinusoid_model = ts.sum_sines(time, f_n_new, a_n_new, ph_n_new)

        # set the sinusoid properties
        self._f_n = f_n_new
        self._a_n = a_n_new
        self._ph_n = ph_n_new
        self.n_sin = len(self._f_n)

        return None

    def add_sinusoids(self, time, f_n_new, a_n_new, ph_n_new):
        """Remove the sinusoids at the provided indices from the list.

        Meant for adding a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sine waves. May also be a float.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves. May also be a float.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves. May also be a float.
        """
        f_n_new = np.atleast_1d(f_n_new)
        a_n_new = np.atleast_1d(a_n_new)
        ph_n_new = np.atleast_1d(ph_n_new)

        # get the current model at the indices
        new_model = ts.sum_sines_st(time, f_n_new, a_n_new, ph_n_new)

        # update the model
        self._sinusoid_model += new_model

        # remove the sinusoid properties
        self._f_n = np.append(self._f_n, f_n_new)
        self._a_n = np.append(self._a_n, a_n_new)
        self._ph_n = np.append(self._ph_n, ph_n_new)
        self.n_sin = len(self._f_n)

        return None

    def update_sinusoids(self, time, f_n_new, a_n_new, ph_n_new, indices):
        """Update the current sinusoid model with changes at the given indices.

        Meant for updating a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sine waves. Include all indices.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves. Include all indices.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves. Include all indices.
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids to update.
        """
        # get the current model at the indices
        cur_model_i = ts.sum_sines_st(time, self._f_n[indices], self._a_n[indices], self._ph_n[indices])

        # make the new model at the indices
        new_model = ts.sum_sines_st(time, f_n_new[indices], a_n_new[indices], ph_n_new[indices])

        # update the model
        self._sinusoid_model = self._sinusoid_model - cur_model_i + new_model

        # change the sinusoid properties
        self._f_n[indices] = f_n_new[indices]
        self._a_n[indices] = a_n_new[indices]
        self._ph_n[indices] = ph_n_new[indices]

        return None

    def remove_sinusoids(self, time, indices):
        """Remove the sinusoids at the provided indices from the list.

        Meant for updating a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        indices: numpy.ndarray[Any, dtype[int]]
            Indices of the sinusoids to remove.
        """
        # get the current model at the indices
        cur_model_i = ts.sum_sines_st(time, self._f_n[indices], self._a_n[indices], self._ph_n[indices])

        # update the model
        self._sinusoid_model = self._sinusoid_model - cur_model_i

        # remove the sinusoid properties
        self._f_n = np.delete(self._f_n, indices)
        self._a_n = np.delete(self._a_n, indices)
        self._ph_n = np.delete(self._ph_n, indices)
        self.n_sin = len(self._f_n)

        return None


class TimeSeriesModel:
    """This class handles the full time series model.

    Attributes
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    linear: LinearModel
        Model of the piece-wise linear curve.
    sinusoid: SinusoidModel
        Model of the sinusoids.
    """

    def __init__(self, time, flux, i_chunks):
        """Initialises the Result object.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # time series
        self.time = time
        self.flux = flux
        self.i_chunks = i_chunks

        # some numbers
        self.n_t = len(time)
        self.n_chunks = len(i_chunks)

        # set time properties
        self.t_tot = np.ptp(self.time)
        self.t_mean = np.mean(self.time)
        self.t_mean_chunk = np.array([np.mean(self.time[ch[0]:ch[1]]) for ch in self.i_chunks])
        self.t_step = np.median(np.diff(self.time))

        # time series models making up the full model
        self.linear = LinearModel(self.n_t, self.n_chunks)
        self.sinusoid = SinusoidModel(self.n_t)

    @property
    def n_param(self):
        """Return the number of parameters of the time series model.

        Returns
        -------
        int
            Number of free parameters in the model.
        """
        return self.linear.n_param + self.sinusoid.n_param

    def copy(self):
        """Creates a copy of the TimeSeriesModel object."""
        return deepcopy(self)

    def get_parameters(self):
        """Get the current model parameters.

        Returns
        -------
        tuple
            Consisting of five numpy.ndarray[Any, dtype[float]] for const, slope, f_n, a_n, ph_n.
        """
        return *self.linear.get_linear_parameters(), *self.sinusoid.get_sinusoid_parameters()

    def full_model(self):
        """The full time series model.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Combined time series model.
        """
        return self.linear.linear_model + self.sinusoid.sinusoid_model

    def residual(self):
        """The residual of the flux minus the model.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Flux minus the current time series model.
        """
        return self.flux - self.full_model()

    def bic(self):
        """Calculate the BIC of the residual.

        Returns
        -------
        float
            BIC of the current time series model.
        """
        return gof.calc_bic(self.residual(), self.n_param)

    def set_linear_model(self, const_new, slope_new):
        """Delegates to set_linear_model of LinearModel."""
        self.linear.set_linear_model(self.time, const_new, slope_new, self.i_chunks)

    def update_linear_model(self):
        """Delegates to update_linear_model of LinearModel."""
        self.linear.update_linear_model(self.time, self.flux - self.sinusoid.sinusoid_model, self.i_chunks)

    def set_sinusoids(self, f_n_new, a_n_new, ph_n_new):
        """Delegates to set_sinusoids of SinusoidModel."""
        self.sinusoid.set_sinusoids(self.time, f_n_new, a_n_new, ph_n_new)

    def add_sinusoids(self, f_n_new, a_n_new, ph_n_new):
        """Delegates to add_sinusoids of SinusoidModel."""
        self.sinusoid.add_sinusoids(self.time, f_n_new, a_n_new, ph_n_new)

    def update_sinusoids(self, f_n_new, a_n_new, ph_n_new, indices):
        """Delegates to update_sinusoids of SinusoidModel."""
        self.sinusoid.update_sinusoids(self.time, f_n_new, a_n_new, ph_n_new, indices)

    def remove_sinusoids(self, indices):
        """Delegates to remove_sinusoids of SinusoidModel."""
        self.sinusoid.remove_sinusoids(self.time, indices)
