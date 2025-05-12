"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the result class for handling the sinusoid model. Includes a piece-wise linear trend
and harmonics.

Code written by: Luc IJspeert
"""
import os
import numpy as np

from star_shine.core import timeseries as ts
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


class ModelLinear:
    """This class handles the linear model.

    Attributes
    ----------
    _const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    _slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    _current_model_linear: numpy.ndarray[Any, dtype[float]]
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

        # current model
        self._current_model_linear = np.zeros((n_time,))

    def set_linear_model(self, time, residual, i_chunks):
        """"""
        const, slope = ts.linear_pars(time, residual, i_chunks)
        self._current_model_linear = ts.linear_curve(time, const, slope, i_chunks)

        return




class ModelSinusoid:
    """This class handles the sinusoid model.

    Attributes
    ----------
    _const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    _slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    _f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves.
    _a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves.
    _ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sine waves.
    _p_orb: float
        Orbital period.
    _current_model_sinusoid: numpy.ndarray[Any, dtype[float]]
        Time series model of the sinusoids.
    """

    def __init__(self, n_time):
        """Initialises the Result object.

        Parameters
        ----------
        n_time: int
            Number of points in the time series.
        """
        # linear model parameters
        self._const = np.zeros((0,))  # y-intercepts
        self._slope = np.zeros((0,))  # slopes

        # sinusoid model parameters
        self._f_n = np.zeros((0,))  # frequencies
        self._a_n = np.zeros((0,))  # amplitudes
        self._ph_n = np.zeros((0,))  # phases

        # harmonic model parameters
        self._p_orb = 0.

        # current model
        self._current_model_sinusoid = np.zeros((0,))

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
        self._current_model_sinusoid = ts.sum_sines(time, f_n_new, a_n_new, ph_n_new)

        # set the sinusoid properties
        self._f_n = f_n_new
        self._a_n = a_n_new
        self._ph_n = ph_n_new

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
            The frequencies of a number of sine waves.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves.
        """
        # get the current model at the indices
        new_model = ts.sum_sines_st(time, f_n_new, a_n_new, ph_n_new)

        # update the model
        self._current_model_sinusoid += new_model

        # remove the sinusoid properties
        self._f_n = np.append(self._f_n, f_n_new)
        self._a_n = np.append(self._a_n, a_n_new)
        self._ph_n = np.append(self._ph_n, ph_n_new)

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
            The frequencies of a number of sine waves.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sine waves.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sine waves.
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids to update.
        """
        # get the current model at the indices
        cur_model = ts.sum_sines_st(time, self._f_n[indices], self._a_n[indices], self._ph_n[indices])

        # make the new model at the indices
        new_model = ts.sum_sines_st(time, f_n_new, a_n_new, ph_n_new)

        # update the model
        self._current_model_sinusoid = self._current_model_sinusoid - cur_model + new_model

        # change the sinusoid properties
        self._f_n[indices] = f_n_new
        self._a_n[indices] = a_n_new
        self._ph_n[indices] = ph_n_new

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
        cur_model = ts.sum_sines_st(time, self._f_n[indices], self._a_n[indices], self._ph_n[indices])

        # update the model
        self._current_model_sinusoid = self._current_model_sinusoid - cur_model

        # remove the sinusoid properties
        self._f_n = np.delete(self._f_n, indices)
        self._a_n = np.delete(self._a_n, indices)
        self._ph_n = np.delete(self._ph_n, indices)

        return None


class TimeSeriesModel:
    """This class handles the full time series model.

    Attributes
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    model_linear: ModelLinear
        Model of the piece-wise linear curve.
    model_sinusoid: ModelSinusoid
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
        """
        # time series
        self.time = time
        self.flux = flux
        self.i_chunks = i_chunks

        # current model
        self.model_linear = ModelLinear(len(time), len(i_chunks))
        self.model_sinusoid = ModelSinusoid(len(time))
