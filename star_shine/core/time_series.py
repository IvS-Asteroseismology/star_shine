"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains classes for handling the time series, including combining it with the full model.

Code written by: Luc IJspeert
"""
import numpy as np

from star_shine.core import model as mdl, goodness_of_fit as gof, periodogram as pdg
from star_shine.config import data_properties as dp


class TimeSeries:
    """This class handles the time series.

    Attributes
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series.
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    n_time: int
        Number of data points in the time series.
    n_chunks: int
        Number of time chunks in the time series.
    t_tot: float
        Total time base of observations.
    t_mean: float
        Time reference (zero) point of the full light curve.
    t_mean_chunk: numpy.ndarray[Any, dtype[float]]
        Time reference (zero) point per chunk.
    t_step: float
        Median time step of observations.
    pd_f0: float
        Starting frequency of the periodogram.
    pd_fn: float
        Last frequency of the periodogram.
    pd_df: float
        Frequency sampling space of the periodogram
    pd_freqs: numpy.ndarray[Any, dtype[float]]
        Frequencies at which the periodogram was calculated
    pd_ampls: numpy.ndarray[Any, dtype[float]]
        The periodogram amplitudes.
    f_resolution: float
        Frequency resolution of the time series
    snr_threshold: float
        Signal-to-noise threshold for this data set.
    """

    def __init__(self, time, flux, flux_err, i_chunks):
        """Initialises the Result object.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series.
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # time series
        self.time = time
        self.flux = flux
        self.flux_err = flux_err
        self.i_chunks = i_chunks

        # some numbers
        self.n_time = len(time)
        self.n_chunks = len(i_chunks)

        # set time properties
        self.t_tot = np.ptp(self.time)
        self.t_mean = np.mean(self.time)
        self.t_mean_chunk = np.array([np.mean(self.time[ch[0]:ch[1]]) for ch in self.i_chunks])
        self.t_step = np.median(np.diff(self.time))

        # settings for periodograms
        self.pd_f0 = 0.01 / self.t_tot  # lower than T/100 no good
        self.pd_df = 0.1 / self.t_tot  # default frequency sampling is about 1/10 of frequency resolution
        self.pd_fn = 0.  # set by update_properties

        # other properties that rely on config
        self.f_resolution = 0.
        self.snr_threshold = 0.
        self.update_properties()  # update these with a function

        # periodogram
        out = pdg.scargle_parallel(self.time, self.flux, f0=self.pd_f0, fn=self.pd_fn, df=self.pd_df, norm='amplitude')
        self.pd_freqs = out[0]
        self.pd_ampls = out[1]

    def update_properties(self):
        """Calculate the properties of the data and fill them in.

        Running this function again will re-evaluate some properties, for if the configuration changed.
        """
        # set data properties relying on config
        self.pd_fn = dp.nyquist_frequency(self.time)
        self.f_resolution = dp.frequency_resolution(self.time)
        self.snr_threshold = dp.signal_to_noise_threshold(self.time)

        return None


class TimeSeriesModel(TimeSeries):
    """This class handles the full time series model.

    Attributes
    ----------
    linear: star_shine.core.time_series.LinearModel
        Model of the piece-wise linear curve.
    sinusoid: star_shine.core.time_series.SinusoidModel
        Model of the sinusoids.
    """

    def __init__(self, time, flux, flux_err, i_chunks):
        """Initialises the Result object.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        flux: numpy.ndarray[Any, dtype[float]]
            Measurement values of the time series.
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # instantiate time series
        super().__init__(time, flux, flux_err, i_chunks)

        # time series models making up the full model
        self.linear = mdl.LinearModel(self.n_time)
        self.sinusoid = mdl.SinusoidModel(self.n_time)

    @staticmethod
    def from_time_series(time_series):
        """Create a TimeSeriesModel instance from a TimeSeries instance."""
        return TimeSeriesModel(time=time_series.time, flux=time_series.flux, flux_err=time_series.flux_err,
                               i_chunks=time_series.i_chunks)

    @property
    def n_param(self):
        """Return the number of parameters of the time series model.

        Returns
        -------
        int
            Number of free parameters in the model.
        """
        return self.linear.n_param + self.sinusoid.n_param

    def get_parameters(self):
        """Get the current model parameters.

        Returns
        -------
        tuple
            Consisting of five numpy.ndarray[Any, dtype[float]] for const, slope, f_n, a_n, ph_n.
        """
        return *self.linear.get_linear_parameters(), *self.sinusoid.get_sinusoid_parameters()

    def model(self):
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
        return self.flux - self.model()

    def bic(self):
        """Calculate the BIC of the residual.

        Returns
        -------
        float
            BIC of the current time series model.
        """
        return gof.calc_bic(self.residual(), self.n_param)

    def periodogram(self, subtract_model=True):
        """Get the Lomb-Scargle periodogram of the time series.

        Uses the cached values for the original data, if the model is subtracted the periodogram is always recomputed.

        Parameters
        ----------
        subtract_model: bool
            Subtract the time series model from the data.

        Returns
        -------
        tuple
            Contains the frequencies numpy.ndarray[Any, dtype[float]]
            and the spectrum numpy.ndarray[Any, dtype[float]]
        """
        if subtract_model:
            f0, fn, df = self.pd_f0, self.pd_fn, self.pd_df
            f, a = pdg.scargle_parallel(self.time, self.residual(), f0=f0, fn=fn, df=df, norm='amplitude')
        else:
            f, a = self.pd_freqs, self.pd_ampls

        return f, a

    def calc_model(self, indices=None):
        """Calculate the full time series model (disregarding include).

        Parameters
        ----------
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids to include.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Combined time series model.
        """
        model_linear = self.linear.calc_linear_model(self.time, self.i_chunks)
        model_sinusoid = self.sinusoid.calc_sinusoid_model(self.time, indices=indices)

        return model_linear + model_sinusoid

    def calc_periodogram(self):
        """Calculate Lomb-Scargle periodogram of the time series (disregarding include).

        Returns
        -------
        tuple
            Contains the frequencies numpy.ndarray[Any, dtype[float]]
            and the spectrum numpy.ndarray[Any, dtype[float]]
        """
        f0, fn, df = self.pd_f0, self.pd_fn, self.pd_df
        f, a = pdg.scargle_parallel(self.time, self.flux - self.calc_model(), f0=f0, fn=fn, df=df, norm='amplitude')

        return f, a

    def set_linear_model(self, const_new, slope_new):
        """Delegates to set_linear_model of LinearModel."""
        self.linear.set_linear_model(self.time, const_new, slope_new, self.i_chunks)

    def update_linear_model(self):
        """Delegates to update_linear_model of LinearModel."""
        self.linear.update_linear_model(self.time, self.flux - self.sinusoid.sinusoid_model, self.i_chunks)

    def update_linear_uncertainties(self):
        """Delegates to update_linear_model_err of LinearModel."""
        self.linear.update_linear_uncertainties(self.time, self.residual(), self.i_chunks)

    def set_sinusoids(self, *args, **kwargs):
        """Delegates to set_sinusoids of SinusoidModel."""
        self.sinusoid.set_sinusoids(self.time, *args, **kwargs)

    def add_sinusoids(self, *args, **kwargs):
        """Delegates to add_sinusoids of SinusoidModel."""
        self.sinusoid.add_sinusoids(self.time, *args, **kwargs)

    def update_sinusoids(self, *args, **kwargs):
        """Delegates to update_sinusoids of SinusoidModel."""
        self.sinusoid.update_sinusoids(self.time, *args, **kwargs)

    def remove_sinusoids(self, indices):
        """Delegates to remove_sinusoids of SinusoidModel."""
        self.sinusoid.remove_sinusoids(self.time, indices)

    def include_sinusoids(self, indices):
        """Delegates to include_sinusoids of SinusoidModel."""
        self.sinusoid.include_sinusoids(self.time, indices)

    def exclude_sinusoids(self, indices):
        """Delegates to exclude_sinusoids of SinusoidModel."""
        self.sinusoid.exclude_sinusoids(self.time, indices)

    def remove_excluded(self):
        """Delegates to remove_excluded of SinusoidModel."""
        self.sinusoid.remove_excluded()

    def update_sinusoid_uncertainties(self):
        """Delegates to update_sinusoid_uncertainties of SinusoidModel."""
        self.sinusoid.update_sinusoid_uncertainties(self.time, self.residual(), self.flux_err, self.i_chunks)

    def update_sinusoid_uncertainties_harmonic(self):
        """Delegates to update_sinusoid_uncertainties_harmonic of SinusoidModel."""
        self.sinusoid.update_sinusoid_uncertainties_harmonic()

    def update_sinusoid_passing_sigma(self):
        """Delegates to update_sinusoid_passing_sigma of SinusoidModel."""
        self.sinusoid.update_sinusoid_passing_sigma()

    def update_sinusoid_passing_snr(self, window_width=1.):
        """Delegates to update_sinusoid_passing_snr of SinusoidModel."""
        self.sinusoid.update_sinusoid_passing_snr(self.time, self.residual(), window_width=window_width)

    def update_sinusoid_passing_harmonic(self):
        """Delegates to update_sinusoid_passing_harmonic of SinusoidModel."""
        self.sinusoid.update_sinusoid_passing_harmonic(self.f_resolution)
