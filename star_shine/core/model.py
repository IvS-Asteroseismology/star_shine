"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains classes for handling the models for the time series. Includes a piece-wise linear model,
sum of sinusoids, and harmonics.

Code written by: Luc IJspeert
"""
import numpy as np
import numba as nb

from star_shine.core import frequency_sets as frs, periodogram as pdg, utility as ut


@nb.njit(cache=True, parallel=True)
def linear_curve(time, const, slope, i_chunks, t_shift=True):
    """Returns a piece-wise linear curve for the given time points
    with slopes and y-intercepts.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve
    slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    t_shift: bool
        Mean center the time axis

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        The model time series of a (set of) straight line(s)

    Notes
    -----
    Assumes the constants and slopes are determined with respect to the sector mean time as zero point.
    """
    curve = np.zeros(len(time))
    for i in nb.prange(len(const)):
        s = i_chunks[i]
        if t_shift:
            t_sector_mean = np.mean(time[s[0]:s[1]])
        else:
            t_sector_mean = 0

        curve[s[0]:s[1]] = const[i] + slope[i] * (time[s[0]:s[1]] - t_sector_mean)

    return curve


@nb.njit(cache=True, parallel=True)
def linear_pars(time, flux, i_chunks):
    """Calculate the slopes and y-intercepts of a linear trend with the MLE.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        y_inter: numpy.ndarray[Any, dtype[float]]
            The y-intercepts of a piece-wise linear curve
        slope: numpy.ndarray[Any, dtype[float]]
            The slopes of a piece-wise linear curve

    Notes
    -----
    Source: https://towardsdatascience.com/linear-regression-91eeae7d6a2e
    Determines the constants and slopes with respect to the sector mean time as zero point to avoid correlations.
    """
    y_inter = np.zeros(len(i_chunks))
    slope = np.zeros(len(i_chunks))

    for i in nb.prange(len(i_chunks)):
        s = i_chunks[i]

        # mean and mean subtracted quantities
        x_m = np.mean(time[s[0]:s[1]])
        x_ms = (time[s[0]:s[1]] - x_m)
        y_m = np.mean(flux[s[0]:s[1]])
        y_ms = (flux[s[0]:s[1]] - y_m)

        # sums
        s_xx = np.sum(x_ms ** 2)
        s_xy = np.sum(x_ms * y_ms)

        # parameters
        slope[i] = s_xy / s_xx
        # y_inter[i] = y_m - slope[i] * x_m  # original non-mean-centered formula
        y_inter[i] = y_m  # mean-centered value

    return y_inter, slope


@nb.njit(cache=True)
def sum_sines_st(time, f_n, a_n, ph_n, t_shift=True):
    """A sum of sinusoids at times t, given the frequencies, amplitudes and phases.

    Single threaded version. Better for one to a few sinusoids.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sinusoids
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sinusoids
    t_shift: bool
        Mean center the time axis

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of a sum of sinusoids. Varies around 0.

    Notes
    -----
    Assumes the phases are determined with respect to the mean time as zero point by default.
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    model_sines = np.zeros(len(time))
    for i in range(len(f_n)):
        model_sines += a_n[i] * np.sin((2 * np.pi * f_n[i] * (time - mean_t)) + ph_n[i])

    return model_sines


@nb.njit(cache=True, parallel=True)
def sum_sines(time, f_n, a_n, ph_n, t_shift=True):
    """A sum of sinusoids at times t, given the frequencies, amplitudes and phases.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sinusoids
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids
    ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sinusoids
    t_shift: bool
        Mean center the time axis

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of a sum of sinusoids. Varies around 0.

    Notes
    -----
    Assumes the phases are determined with respect to the mean time as zero point by default.
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    model_sines = np.zeros(len(time))
    for i in nb.prange(len(f_n)):
        model_sines += a_n[i] * np.sin((2 * np.pi * f_n[i] * (time - mean_t)) + ph_n[i])

    return model_sines


@nb.njit(cache=True, parallel=True)
def sum_sines_deriv(time, f_n, a_n, ph_n, deriv=1, t_shift=True):
    """The time derivative of a sum of sinusoids at times t.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    f_n: list[float], numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sinusoids
    a_n: list[float], numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids
    ph_n: list[float], numpy.ndarray[Any, dtype[float]]
        The phases of a number of sinusoids
    deriv: int
        Number of time derivatives taken (>= 1)
    t_shift: bool
        Mean center the time axis

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model time series of a sum of sinusoid derivatives. Varies around 0.

    Notes
    -----
    Assumes the phases are determined with respect to the mean time as zero point by default.
    """
    if t_shift:
        mean_t = np.mean(time)
    else:
        mean_t = 0

    model_sines = np.zeros(len(time))
    mod_2 = deriv % 2
    mod_4 = deriv % 4
    ph_cos = (np.pi / 2) * mod_2  # alternate between cosine and sine
    sign = (-1) ** ((mod_4 - mod_2) // 2)  # (1, -1, -1, 1, 1, -1, -1... for deriv=1, 2, 3...)

    for i in nb.prange(len(f_n)):
        for j in range(len(time)):
            model_sines[j] += (sign * (2 * np.pi * f_n[i]) ** deriv * a_n[i] *
                               np.sin((2 * np.pi * f_n[i] * (time[j] - mean_t)) + ph_n[i] + ph_cos))

    return model_sines


class LinearModel:
    """This class handles the linear model.

    Attributes
    ----------
    _const: numpy.ndarray[Any, dtype[float]]
        The y-intercepts of a piece-wise linear curve.
    _slope: numpy.ndarray[Any, dtype[float]]
        The slopes of a piece-wise linear curve.
    _const_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the constant for each time chunk.
    _slope_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the slope for each time chunk.
    _linear_model: numpy.ndarray[Any, dtype[float]]
        Time series model of the piece-wise linear curve.
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

        # linear parameter uncertainties
        self._const_err = np.zeros((0,))
        self._slope_err = np.zeros((0,))

        # number of time chunks
        self.n_chunks = 0

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
    def const_err(self):
        """Get the current model constants.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The y-intercepts of a piece-wise linear curve.
        """
        return self._const_err.copy()

    @property
    def slope_err(self):
        """Get the current model slopes.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The slopes of a piece-wise linear curve.
        """
        return self._slope_err.copy()

    @property
    def linear_model(self):
        """Get the current linear model.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Time series model of the piece-wise linear curve.
        """
        return self._linear_model.copy()

    @property
    def n_param(self):
        """Get the number of parameters of the model."""
        return 2 * self.n_chunks

    def update_n(self):
        """Update the current numbers of sinusoids, harmonics, and base frequencies."""
        self.n_chunks = len(self._const)

        return None

    def get_linear_parameters(self):
        """Get a copy of the current linear parameters.

        Returns
        -------
        tuple
            Consisting of two numpy.ndarray[Any, dtype[float]] for const, slope.
        """
        return self.const, self.slope

    def calc_linear_model(self, time, i_chunks):
        """Calculate the current linear model.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Time series model of the piece-wise linear curve.
        """
        return linear_curve(time, self.const, self.slope, i_chunks)

    def set_linear_model(self, time, const_new, slope_new, i_chunks):
        """Set the linear model according to the new parameters.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        const_new: numpy.ndarray[Any, dtype[float]]
            New y-intercepts.
        slope_new: numpy.ndarray[Any, dtype[float]]
            New slopes.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # make the new model
        self._linear_model = linear_curve(time, const_new, slope_new, i_chunks)

        # set the parameters
        self._const = const_new
        self._slope = slope_new

        # set the numbers
        self.update_n()

        return

    def update_linear_model(self, time, residual, i_chunks):
        """Update the linear model using the residual flux.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        residual: numpy.ndarray[Any, dtype[float]]
            Residual flux (flux minus model).
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        # get new parameters
        const_new, slope_new = linear_pars(time, residual, i_chunks)

        # set the new parameters and model
        self.set_linear_model(time, const_new, slope_new, i_chunks)

        return

    def update_linear_uncertainties(self, time, residual, i_chunks):
        """Update the linear model parameter errors using the residual flux.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        residual: numpy.ndarray[Any, dtype[float]]
            Residual flux (flux minus model).
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        self._const_err, self._slope_err = ut.formal_uncertainties_linear(time, residual, i_chunks)

        return


class SinusoidModel:
    """This class handles the sinusoid model.

    Attributes
    ----------
    _f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sinusoids.
    _a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids.
    _ph_n: numpy.ndarray[Any, dtype[float]]
        The phases of a number of sinusoids.
    _include: numpy.ndarray[Any, dtype[bool]]
        Include state of the sinusoids in the model.
    _harmonics: numpy.ndarray[Any, dtype[bool]]
        Boolean mask indicating harmonics in _f_n. False indicates a free sinusoid.
    _h_base: numpy.ndarray[Any, dtype[int]]
        Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
    _h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
    _combinations: dict[int, list[int]]
        Indices of the child frequencies in _f_n, and the corresponding parents.
    _combination_n: dict[int, numpy.ndarray[Any, dtype[int]]]
        Indices of the child frequencies in _f_n, and the corresponding parent multiplier.
    _f_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sinusoid.
    _a_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the amplitude for each sinusoid (these are identical).
    _ph_n_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the phase for each sinusoid.
    _f_h_err: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each harmonic sinusoid.
    _passing_sigma: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the sigma check.
    _passing_snr: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the signal-to-noise check.
    _passing_harmonic: numpy.ndarray[Any, dtype[bool]]
        Sinusoids that passed the harmonic check.
    n_sin: int
        Number of sinusoids (including harmonics).
    n_harm: int
        Number of harmonic sinusoids.
    n_base: int
        Number of harmonic series/base frequencies.
    _sinusoid_model: numpy.ndarray[Any, dtype[float]]
        Current time series model of the sinusoids.
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

        # for consistency of indices in the removal of sinusoids
        self._include = np.zeros((0,), dtype=bool)

        # harmonic model parameters
        self._harmonics = np.zeros((0,), dtype=bool)
        self._h_base = -np.ones((0,), dtype=int)
        self._h_mult = np.zeros((0,), dtype=int)

        # combination model parameters [wip: logic not implemented yet]
        self._combinations = dict()
        self._combination_n = dict()

        # sinusoid parameter uncertainties
        self._f_n_err = np.zeros((0,))
        self._a_n_err = np.zeros((0,))
        self._ph_n_err = np.zeros((0,))

        # harmonic parameter uncertainties
        self._f_h_err = np.zeros((0,))

        # passing criteria
        self._passing_sigma = np.zeros((0,), dtype=bool)
        self._passing_snr = np.zeros((0,), dtype=bool)
        self._passing_harmonic = np.zeros((0,), dtype=bool)

        # number of sinusoids
        self.n_sin = 0  # number of sinusoids (including harm)
        self.n_harm = 0  # number of harmonics
        self.n_base = 0  # number of harmonic series

        # current model
        self._sinusoid_model = np.zeros((n_time,))

    @property
    def f_n(self):
        """Get a copy of the current model frequencies (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sinusoids.
        """
        return self._f_n.copy()

    @property
    def a_n(self):
        """Get a copy of the current model amplitudes (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sinusoids.
        """
        return self._a_n.copy()

    @property
    def ph_n(self):
        """Get a copy of the current model phases (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The phases of a number of sinusoids.
        """
        return self._ph_n.copy()

    @property
    def include(self):
        """Get a copy of the current include state of the sinusoids.

        Returns
        -------
        numpy.ndarray[Any, dtype[bool]]
            Include state of the sinusoids in the model.
        """
        return self._include.copy()

    @property
    def has_excludes(self):
        """Check if there are any not included sinusoids."""
        return np.any(~self._include)

    @property
    def harmonics(self):
        """Get a copy of the current model harmonics (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[bool]]
            Boolean mask indicating harmonics in _f_n. False indicates a free sinusoid.
        """
        return self._harmonics.copy()

    @property
    def h_base(self):
        """Get a copy of the current model harmonic base frequency indices (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[int]]
            Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
        """
        return self._h_base.copy()

    @property
    def h_mult(self):
        """Get a copy of the current model harmonic multiplier numbers (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[int]]
            Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
        """
        return self._h_mult.copy()

    @property
    def f_base(self):
        """Get the current model base frequencies (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The base frequencies of the harmonic model.
        """
        return self._f_n[self._h_mult == 1]

    @property
    def f_n_err(self):
        """Get a copy of the model frequency errors (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The errors in the frequencies of a number of sinusoids.

        Notes
        -----
        These need to be updated manually (with update_sinusoid_uncertainties).
        """
        return self._f_n_err.copy()

    @property
    def a_n_err(self):
        """Get a copy of the model amplitude errors (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The errors in the amplitudes of a number of sinusoids.

        Notes
        -----
        These need to be updated manually (with update_sinusoid_uncertainties).
        """
        return self._a_n_err.copy()

    @property
    def ph_n_err(self):
        """Get a copy of the model phase errors (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The errors in the phases of a number of sinusoids.

        Notes
        -----
        These need to be updated manually (with update_sinusoid_uncertainties).
        """
        return self._ph_n_err.copy()

    @property
    def f_h_err(self):
        """Get a copy of the model harmonic frequency errors (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The errors in the frequencies of a number of (harmonic) sinusoids.

        Notes
        -----
        These need to be updated manually (with update_sinusoid_uncertainties_harmonic).
        """
        return self._f_h_err.copy()

    @property
    def passing_sigma(self):
        """Get a copy of the passing sigma mask (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[bool]]
            Sinusoids that passed the sigma check.

        Notes
        -----
        These need to be updated manually (with update_sinusoid_passing_sigma).
        """
        return self._passing_sigma.copy()

    @property
    def passing_snr(self):
        """Get a copy of the passing snr mask (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[bool]]
            Sinusoids that passed the signal-to-noise check.
        """
        return self._passing_snr.copy()

    @property
    def passing_harmonic(self):
        """Get a copy of the passing harmonic mask (disregarding include).

        Returns
        -------
        numpy.ndarray[Any, dtype[bool]]
            Sinusoids that passed the harmonic check.
        """
        return self._passing_harmonic.copy()

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
        return self.n_base + 2 * self.n_harm + 3 * (self.n_sin - self.n_harm)

    def update_n(self):
        """Update the current numbers of sinusoids, harmonics, and base frequencies."""
        self.n_sin = len(self._f_n[self._include])
        self.n_harm = len(self._f_n[self._include][self._harmonics[self._include]])
        self.n_base = len(np.unique(self._h_base[(self._h_base != -1) & self._include]))

        return None

    def get_sinusoid_parameters(self, exclude=True):
        """Get a copy of the current sinusoid parameters.

        Parameters
        ----------
        exclude: bool
            Exclude the sinusoids intended for removal.

        Returns
        -------
        tuple
            Consisting of three numpy.ndarray[Any, dtype[float]] for f_n, a_n, ph_n.
        """
        if exclude:
            f_n, a_n, ph_n = self._f_n[self._include], self._a_n[self._include], self._ph_n[self._include]
        else:
            f_n, a_n, ph_n = self._f_n, self._a_n, self._ph_n

        return f_n, a_n, ph_n

    def get_harmonic_parameters(self, exclude=True):
        """Get a copy of the current harmonic parameters.

        Parameters
        ----------
        exclude: bool
            Exclude the sinusoids intended for removal.

        Returns
        -------
        tuple
            Consisting of three numpy.ndarray[Any, dtype[float]] for f_n, a_n, ph_n.
        """
        if exclude:
            harmonics = self._harmonics[self._include]
            h_base, h_mult = self._h_base[self._include], self._h_mult[self._include]
        else:
            harmonics, h_base, h_mult = self._harmonics, self._h_base, self._h_mult

        return harmonics, h_base, h_mult

    def get_h_base_map(self):
        """Get the indices of each base harmonic frequency and a map to recreate h_base[harmonics].

        Returns
        -------
        tuple
            numpy.ndarray[Any, dtype[int]]
                Indices of the base harmonic frequencies in _f_n.
            numpy.ndarray[Any, dtype[int]]
                Map of harmonics to base frequencies.
        """
        return np.unique(self.h_base[self._harmonics], return_inverse=True)

    def get_f_index(self, f):
        """Get the index in f_n of a given frequency.

        Parameters
        ----------
        f: float
            Frequency to get the index of.

        Returns
        -------
        int
            Index of f in f_n.
        """
        return np.abs(self._f_n - f).argmin()

    def calc_sinusoid_model(self, time, indices=None):
        """Calculate the current sinusoid model (disregarding include).

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids to include.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Time series model of the sinusoids.
        """
        if indices is None:
            model = sum_sines(time, self.f_n, self.a_n, self.ph_n)
        else:
            model = sum_sines(time, self.f_n[indices], self.a_n[indices], self.ph_n[indices])

        return model

    def _check_removed_h_base(self, indices):
        """If a harmonic base frequency was removed, remove the whole harmonic series.

        Parameters
        ----------
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids that were removed.
        """
        if np.any(self._harmonics):
            for i in np.arange(len(self._f_n))[self._h_mult == 1]:
                if i in indices:
                    series_mask = self._h_base == i
                    self._harmonics[series_mask] = False
                    self._h_base[series_mask] = -1
                    self._h_mult[series_mask] = 0

        return None

    def enforce_harmonics(self):
        """Enforce harmonic relations in the sinusoids.

        This will set all harmonic frequencies to the integer multiple of the base frequency that they are
        assigned to. Optionally performs a linear fit of the amplitudes and phases [wip].
        """
        if np.any(self._harmonics):
            self._f_n[self._harmonics] = self._f_n[self._h_base[self._harmonics]] * self._h_mult[self._harmonics]

        return None

    def set_sinusoids(self, time, f_n_new, a_n_new, ph_n_new, h_base_new=None, h_mult_new=None):
        """Set the current sinusoid model with the new parameters.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sinusoids.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sinusoids.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sinusoids.
        h_base_new: numpy.ndarray[Any, dtype[int]], optional
            Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
        h_mult_new: numpy.ndarray[Any, dtype[int]], optional
            Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
        """
        # ensure 1d
        f_n_new = np.atleast_1d(f_n_new)
        a_n_new = np.atleast_1d(a_n_new)
        ph_n_new = np.atleast_1d(ph_n_new)

        n_new = len(f_n_new)

        # ensure harmonic input ok
        if h_base_new is None:
            h_base_new = -np.ones(n_new, dtype=int)
        h_base_new = np.atleast_1d(h_base_new)
        if h_mult_new is None:
            h_mult_new = np.zeros(n_new, dtype=int)
        h_mult_new = np.atleast_1d(h_mult_new)
        harmonics_new = h_base_new != -1

        # make the new model
        self._sinusoid_model = sum_sines(time, f_n_new, a_n_new, ph_n_new)

        # update the sinusoid parameters
        self._f_n = f_n_new
        self._a_n = a_n_new
        self._ph_n = ph_n_new
        self._include = np.ones(n_new, dtype=bool)

        # update harmonic parameters
        self._harmonics = harmonics_new
        self._h_base = h_base_new
        self._h_mult = h_mult_new

        # update numbers
        self.update_n()

        return None

    def add_sinusoids(self, time, f_n_new, a_n_new, ph_n_new, h_base_new=None, h_mult_new=None):
        """Add the sinusoids to the list.

        Meant for adding a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sinusoids.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sinusoids.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sinusoids.
        h_base_new: numpy.ndarray[Any, dtype[int]], optional
            Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
        h_mult_new: numpy.ndarray[Any, dtype[int]], optional
            Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
        """
        # ensure 1d
        f_n_new = np.atleast_1d(f_n_new)
        a_n_new = np.atleast_1d(a_n_new)
        ph_n_new = np.atleast_1d(ph_n_new)

        n_new = len(f_n_new)

        # ensure harmonic input ok
        if h_base_new is None:
            h_base_new = -np.ones(n_new, dtype=int)
        h_base_new = np.atleast_1d(h_base_new)
        if h_mult_new is None:
            h_mult_new = np.zeros(n_new, dtype=int)
        h_mult_new = np.atleast_1d(h_mult_new)
        harmonics_new = h_base_new != -1

        # make the new model
        new_model = sum_sines_st(time, f_n_new, a_n_new, ph_n_new)

        # update the model
        self._sinusoid_model += new_model

        # update the sinusoid parameters
        self._f_n = np.append(self._f_n, f_n_new)
        self._a_n = np.append(self._a_n, a_n_new)
        self._ph_n = np.append(self._ph_n, ph_n_new)
        self._include = np.append(self._include, np.ones(n_new, dtype=bool))

        # update harmonic parameters
        self._harmonics = np.append(self._harmonics, harmonics_new)
        self._h_base = np.append(self._h_base, h_base_new)
        self._h_mult = np.append(self._h_mult, h_mult_new)

        # update numbers
        self.update_n()

        return None

    def update_sinusoids(self, time, f_n_new, a_n_new, ph_n_new, indices, h_base_new=None, h_mult_new=None):
        """Update the current sinusoid model with changes at the given indices.

        Meant for updating a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        f_n_new: numpy.ndarray[Any, dtype[float]]
            The frequencies of a number of sinusoids.
        a_n_new: numpy.ndarray[Any, dtype[float]]
            The amplitudes of a number of sinusoids.
        ph_n_new: numpy.ndarray[Any, dtype[float]]
            The phases of a number of sinusoids.
        indices: numpy.ndarray[Any, dtype[int]]
            Indices for the sinusoids to update.
        h_base_new: numpy.ndarray[Any, dtype[int]], optional
            Indices of the base frequencies in _f_n for each of _f_n. -1 indicates a free sinusoid.
        h_mult_new: numpy.ndarray[Any, dtype[int]], optional
            Harmonic multiplier of the base frequency for each of _f_n. 0 indicates a free sinusoid.
        """
        # ensure 1d
        f_n_new = np.atleast_1d(f_n_new)
        a_n_new = np.atleast_1d(a_n_new)
        ph_n_new = np.atleast_1d(ph_n_new)
        indices = np.atleast_1d(indices)

        n_new = len(f_n_new)

        # ensure harmonic input ok
        if h_base_new is None:
            h_base_new = -np.ones(n_new, dtype=int)
        h_base_new = np.atleast_1d(h_base_new)
        if h_mult_new is None:
            h_mult_new = np.zeros(n_new, dtype=int)
        h_mult_new = np.atleast_1d(h_mult_new)
        harmonics_new = h_base_new != -1

        # if a full list of sinusoid parameters is given, pick out the ones at the indices
        if len(f_n_new) == len(self._f_n):
            f_n_new, a_n_new, ph_n_new = f_n_new[indices], a_n_new[indices], ph_n_new[indices]
            harmonics_new, h_base_new, h_mult_new = harmonics_new[indices], h_base_new[indices], h_mult_new[indices]

        # get a list of indices that are currently included in the model
        i_include = indices[self._include[indices]]

        # get the current model at the indices
        cur_model_i = sum_sines_st(time, self._f_n[i_include], self._a_n[i_include], self._ph_n[i_include])

        # make the new model
        new_model = sum_sines_st(time, f_n_new, a_n_new, ph_n_new)

        # update the model
        self._sinusoid_model = self._sinusoid_model - cur_model_i + new_model

        # update the sinusoid parameters
        self._f_n[indices] = f_n_new
        self._a_n[indices] = a_n_new
        self._ph_n[indices] = ph_n_new
        self._include[indices] = True  # we assume that the changed sinusoids need to be included

        # update harmonic parameters
        self._harmonics[indices] = harmonics_new
        self._h_base[indices] = h_base_new
        self._h_mult[indices] = h_mult_new

        # update numbers
        self.update_n()

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
        indices = np.atleast_1d(indices)

        # get a list of indices that are currently included in the model
        i_include = indices[self._include[indices]]

        # get the current model at the indices
        cur_model_i = sum_sines_st(time, self._f_n[i_include], self._a_n[i_include], self._ph_n[i_include])

        # update the model
        self._sinusoid_model = self._sinusoid_model - cur_model_i

        # update the sinusoid parameters
        self._f_n = np.delete(self._f_n, indices)
        self._a_n = np.delete(self._a_n, indices)
        self._ph_n = np.delete(self._ph_n, indices)
        self._include = np.delete(self._include, indices)

        # update harmonic parameters
        self._harmonics = np.delete(self._harmonics, indices)
        self._h_base = np.delete(self._h_base, indices)
        self._h_mult = np.delete(self._h_mult, indices)

        # if we deleted a base harmonic, also delete the harmonic series
        self._check_removed_h_base(indices)

        # update indices for the removals (needs to go after _check_removed_h_base)
        self._h_base = np.array(ut.adjust_indices_removed(self._h_base, indices))

        # update numbers
        self.update_n()

        return None

    def include_sinusoids(self, time, indices):
        """Add back the sinusoids at the provided indices to the model.

        Meant for updating a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        indices: numpy.ndarray[Any, dtype[int]]
            Indices of the sinusoids to include.
        """
        indices = np.atleast_1d(indices)

        # get a list of indices that are currently excluded from the model
        i_exclude = indices[~self._include[indices]]

        # get the current model at the indices
        cur_model_i = sum_sines_st(time, self._f_n[i_exclude], self._a_n[i_exclude], self._ph_n[i_exclude])

        # update the model
        self._sinusoid_model = self._sinusoid_model + cur_model_i

        # set their include parameter
        self._include[i_exclude] = True

        # update numbers
        self.update_n()

        return None

    def exclude_sinusoids(self, time, indices):
        """Remove the sinusoids at the provided indices from the model.

        Does not remove the sinusoids from the list yet, for index consistency.
        Meant for updating a limited number of sinusoids, less efficient for large numbers.
        For that case, see set_sinusoids.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series
        indices: numpy.ndarray[Any, dtype[int]]
            Indices of the sinusoids to exclude.
        """
        indices = np.atleast_1d(indices)

        # get a list of indices that are currently included in the model
        i_include = indices[self._include[indices]]

        # get the current model at the indices
        cur_model_i = sum_sines_st(time, self._f_n[i_include], self._a_n[i_include], self._ph_n[i_include])

        # update the model
        self._sinusoid_model = self._sinusoid_model - cur_model_i

        # set their include parameter
        self._include[i_include] = False

        # update numbers
        self.update_n()

        return None

    def remove_excluded(self):
        """Remove the sinusoids that are currently not included from the list."""
        # make removal indices out of _include
        indices = np.arange(len(self._include))[~self._include]

        # the model already doesn't include these sinusoids

        # remove the sinusoid parameters
        self._f_n = self._f_n[self._include]
        self._a_n = self._a_n[self._include]
        self._ph_n = self._ph_n[self._include]

        # remove the harmonic parameters
        self._harmonics = self._harmonics[self._include]
        self._h_base = self._h_base[self._include]
        self._h_mult = self._h_mult[self._include]

        # include itself needs to go last
        self._include = self._include[self._include]

        # check for base frequency removal
        self._check_removed_h_base(indices)

        # update indices for the removals (needs to go after _check_removed_h_base)
        self._h_base = np.array(ut.adjust_indices_removed(self._h_base, indices))

        # numbers do not change here, except if we removed a base frequency
        self.update_n()  # n_sin won't change but n_harm and n_base might

        return None

    def update_sinusoid_uncertainties(self, time, residual, flux_err, i_chunks):
        """Update the sinusoid model parameter errors using the residual flux.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        residual: numpy.ndarray[Any, dtype[float]]
            Residual flux (flux minus model).
        flux_err: numpy.ndarray[Any, dtype[float]]
            Errors in the measurement values.
        i_chunks: numpy.ndarray[Any, dtype[int]]
            Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
            the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
        """
        out = ut.formal_uncertainties_sinusoid(time, residual, flux_err, self._a_n, i_chunks)

        self._f_n_err = out[0]
        self._a_n_err = out[1]
        self._ph_n_err = out[2]

        return

    def update_sinusoid_uncertainties_harmonic(self):
        """Update the harmonic sinusoid model parameter errors.

        Note: uses the sinusoid uncertainties. These should be updated first.
        """
        self._f_h_err = ut.formal_uncertainties_harmonic(self._f_n_err, self._h_base, self._h_mult)

        return

    def update_sinusoid_passing_sigma(self):
        """Update the passing status of the sinusoids for the sigma criterion.

        Note: uses the sinusoid uncertainties. These should be updated first.
        """
        # find the insignificant frequencies
        out = frs.remove_insignificant_sigma(self._f_n, self._f_n_err, self._a_n, self._a_n_err, sigma_f=3, sigma_a=3)

        # invert removal mask
        self._passing_sigma = ~out

        return

    def update_sinusoid_passing_snr(self, time, residual, window_width=1.):
        """Update the passing status of the sinusoids for the sigma criterion.

        Parameters
        ----------
        time: numpy.ndarray[Any, dtype[float]]
            Timestamps of the time series.
        residual: numpy.ndarray[Any, dtype[float]]
            Residual flux (flux minus model).
        window_width: float
            The width of the window used to compute the noise spectrum.
        """
        # calculate the noise at each frequency
        noise_at_f = pdg.scargle_noise_at_freq(self._f_n, time, residual, window_width=window_width)

        # determine the insignificant frequencies
        out = frs.remove_insignificant_snr(time, self._a_n, noise_at_f)

        # invert removal mask
        self._passing_snr = ~out

        return

    def update_sinusoid_passing_harmonic(self, f_resolution):
        """Update the passing status of the sinusoids for the sigma criterion.

        Parameters
        ----------
        f_resolution: float
            Frequency resolution of the time series
        """
        # select candidate harmonic frequencies meeting some criteria
        self._passing_harmonic = np.zeros(self.n_sin, dtype=bool)
        for f_base in self.f_base:
            harmonics, _ = frs.select_harmonics_sigma(self._f_n, self._f_n_err, f_base, f_tol=f_resolution / 2,
                                                      sigma_f=3)
            self._passing_harmonic[harmonics] = True

        return
