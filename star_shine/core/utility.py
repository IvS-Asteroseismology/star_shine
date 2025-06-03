"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains utility functions for data processing, unit conversions.

Code written by: Luc IJspeert
"""
import datetime
import numpy as np
import numba as nb

from star_shine.config.helpers import get_config


# load configuration
config = get_config()


def datetime_formatted():
    """Return datetime string without microseconds.

    Returns
    -------
    str
        Date and time without microseconds
    """
    dt = datetime.datetime.now()
    dt_str = str(dt.date()) + ' ' + str(dt.hour) + ':' + str(dt.minute) + ':' + str(dt.second)

    return dt_str


@nb.njit(cache=True)
def decimal_figures(x, n_sf):
    """Determine the number of decimal figures to print given a target
    number of significant figures

    Parameters
    ----------
    x: float
        Value to determine the number of decimals for.
    n_sf: int
        Number of significant figures to compute.

    Returns
    -------
    int
        Number of decimal places to round to.
    """
    if x != 0:
        decimals = (n_sf - 1) - int(np.floor(np.log10(abs(x))))
    else:
        decimals = 1

    return decimals


@nb.njit(cache=True)
def float_to_str_numba(x, dec=2):
    """Convert float to string for Numba up to some decimal place
    
    Parameters
    ----------
    x: float
        Value to convert
    dec: int
        Number of decimals (be careful with large numbers here)
    
    Returns
    -------
    str
        String with the value x
    """
    x_round = np.round(x, dec)
    x_int = int(x_round)
    x_dec = int(np.abs(np.round(x_round - x_int, dec)) * 10**dec)
    s = str(x_int) + '.' + str(x_dec).zfill(dec)

    return s


def float_to_str_scientific(x, x_err, error=True, brackets=False):
    """Conversion of a number with an error margin to string in scientific notation.

    Uses two significant figures by default as no distinction is made between having a 1, 2 or higher number
    in the error value.

    Parameters
    ----------
    x: float
        Value to determine the number of decimals for.
    x_err: float
        Value to determine the number of decimals for.
    error: bool, optional
        Include the error value.
    brackets: bool, optional
        Place the error value in brackets.

    Returns
    -------
    str
        Formatted string conversion.
    """
    # determine the decimal places to round to
    rnd_x = max(decimal_figures(x_err, 2), decimal_figures(x, 2))

    # format the error value
    if brackets:
        err_str = f"(\u00B1{x_err:.{rnd_x}f})"
    else:
        err_str = f"\u00B1 {x_err:.{rnd_x}f}"

    # format the rest of the string
    if error:
        number_str = f"{x:.{rnd_x}f} {err_str}"
    else:
        number_str = f"{x:.{rnd_x}f}"

    return number_str


@nb.njit(cache=True)
def weighted_mean(x, w):
    """Weighted mean since Numba doesn't support numpy.average
    
    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the mean over
    w: numpy.ndarray[Any, dtype[float]]
        Weights corresponding to each value
    
    Returns
    -------
    float
        Mean of x weighted by w
    """
    w_mean = np.sum(x * w) / np.sum(w)

    return w_mean


@nb.njit(cache=True)
def std_unb(x, n):
    """Unbiased standard deviation

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Values to calculate the std over
    n: int
        Number of degrees of freedom

    Returns
    -------
    float
        Unbiased standard deviation
    """
    residuals = x - np.mean(x)

    # tested to be faster in numba than np.sum(x**2)
    sum_r_2 = 0
    for i in range(len(residuals)):
        sum_r_2 += residuals[i]**2

    std = np.sqrt(sum_r_2 / n)  # unbiased standard deviation of the residuals

    return std


def consecutive_subsets(x):
    """Creates all consecutive subsets of the given list.

    Parameters
    ----------
    x: list[Any], numpy.ndarray[Any, dtype[Any]]
        A list of values.

    Returns
    -------
    list
        The list of consecutive subsets ordered by size and with length two or more.
    """
    n = len(x)

    # create the subsets from largest to smallest
    subsets = [x[p1:p1 + l] for l in range(n, 1, -1) for p1 in range(n - l + 1)]

    return subsets


def adjust_indices_removed(x, removed):
    """Adjusts the indices in `x` to account for removals.

    If y_reduced=np.delete(y, removed), and x are indices indicating items in y, this function gives
    the indices x_adj that indicate the same items in y_reduced. Assumes that none of the removed items
    are indicated by indices in x.

    Parameters
    ----------
    x: iterable[int], numpy.ndarray[Any, dtype[int]]
        A list of indices that slice another array, y.
    removed: iterable[int], numpy.ndarray[Any, dtype[int]]
        A second list of indices indicating removals from y.

    Returns
    -------
    list[int]
        Adjusted indices that can be used with the reduced y.
    """
    # adjust the indices
    x_adj = [i - sum(1 for r in removed if r < i) for i in x]

    return x_adj


@nb.njit(cache=True)
def find_local_max(y):
    """Find the indeces of the local maxima in `y`.

    The array `x` must be sorted in ascending order. This function identifies local maxima in the `y` array
    and returns the corresponding indeces.

    Parameters
    ----------
    y: numpy.ndarray[Any, dtype[float]]
        1D array of y-values corresponding to the x-values.

    Returns
    -------
    numpy.ndarray[Any, dtype[int]]
        Indeces of the `x` values corresponding to the local maxima.

    Examples
    --------
    >>> import numpy as np
    >>> y = np.array([1, 3, 2, 5, 4, 6])
    >>> find_local_max(y)
    array([1, 3, 5])
    """
    # differenced y
    dy = y[1:] - y[:-1]

    # sign of difference of y
    sign_dy = np.sign(dy)
    sign_dy = np.concatenate((np.array([1]), sign_dy, np.array([-1])))  # add 1 and -1 on the ends for maximum finding

    # find the maxima in y
    extrema = sign_dy[1:] - sign_dy[:-1]  # places where the array dy changes sign
    maxima = np.arange(len(extrema))[extrema < 0]  # places where y has a maximum

    return maxima


@nb.njit(cache=True)
def uphill_local_max(x, y, x_approx):
    """Find the index of the local maximum in `y` uphill from `x_approx`.

    The array `x` must be sorted in ascending order. This function identifies local maxima in the `y` array
    and returns the index of the `x` value corresponding to the local maximum uphill from `x_approx`.

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        1D array of x-values, which must be sorted in ascending order.
    y: numpy.ndarray[Any, dtype[float]]
        1D array of y-values corresponding to the x-values.
    x_approx: numpy.ndarray[Any, dtype[float]]
        The x-value(s) around which to find the nearest local maximum.

    Returns
    -------
    numpy.ndarray[Any, dtype[int]]
        Index of the `x` value(s) corresponding to the nearest local maximum to `x_approx`.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4, 5, 6])
    >>> y = np.array([1, 3, 2, 5, 4, 6])
    >>> x_approx = np.array([2.4, 4.6])
    >>> uphill_local_max(x, y, x_approx)
    array([1, 3])
    """
    # [note] this is find_local_max(y), but since we need sign_dy we do this here
    # differenced y
    dy = y[1:] - y[:-1]

    # find the maxima in y
    sign_dy = np.sign(dy)
    sign_dy = np.concatenate((np.array([1]), sign_dy, np.array([-1])))  # add 1 and -1 on the ends for maximum finding
    extrema = sign_dy[1:] - sign_dy[:-1]  # places where the array dy changes sign
    maxima = np.arange(len(extrema))[extrema < 0]  # places where y has a maximum

    # position of x_approx in x (index is point to the right of x_approx)
    pos_x = np.searchsorted(x, x_approx)  # pos_x can be 0 and len(x) but this is ok for sign_dy

    # do we need to move left or right to get to the maximum
    left_right = sign_dy[pos_x]  # negative: left, positive: right

    # get the maximum for each x_approx (index is maximum to the right of pos_x)
    pos_max = np.searchsorted(maxima, pos_x)
    # take the maximum on the left for negative slopes
    pos_max[left_right == -1] -= 1
    x_max = maxima[pos_max]

    return x_max


@nb.njit(cache=True)
def mask_between(x, locations):
    """Mask out everything except the parts between the given timestamps

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Time series to be masked.
    locations: numpy.ndarray[Any, dtype[float]]
        Pairs of points in the time series. Everything but the space between points is masked.

    Returns
    -------
    numpy.ndarray[bool]
        Boolean mask that is True between the locations.
    """
    mask = np.zeros(len(x), dtype=np.bool_)
    for loc in locations:
        mask = mask | ((x >= loc[0]) & (x <= loc[-1]))

    return mask


@nb.njit(cache=True)
def mark_gaps(x, min_gap=1.):
    """Mark gaps in a time series.

    Parameters
    ----------
    x: numpy.ndarray[Any, dtype[float]]
        Time series with gaps.
    min_gap: float, optional
        Minimum width for a gap (in time units).

    Returns
    -------
    gaps: numpy.ndarray[Any, dtype[float]]
        Gap timestamps in pairs.
    """
    # mark the gaps
    t_sorted = np.sort(x)
    t_diff = t_sorted[1:] - t_sorted[:-1]  # np.diff(a)
    gaps = (t_diff > min_gap)

    # get the timestamps
    t_left = t_sorted[:-1][gaps]
    t_right = t_sorted[1:][gaps]
    gaps = np.column_stack((t_left, t_right))

    return gaps


@nb.njit(cache=True)
def n_parameters(n_chunks, n_sinusoids, n_harmonics):
    """Return the number of parameters of the model."""
    # equation for number of parameters
    n_param = 2 * n_chunks  # time chunk each with constant and slope
    n_param += int(n_harmonics > 0)  # one period if harmonics present
    n_param += 2 * n_harmonics  # harmonics (sinusoids with constrained frequency)
    n_param += 3 * (n_sinusoids - n_harmonics)  # sinusoids with free frequency

    return n_param


@nb.njit(cache=True)
def correct_for_crowdsap(flux, crowdsap, i_chunks):
    """Correct the flux for flux contribution of a third source
    
    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).
    
    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series corrected for
        contaminating light
    
    Notes
    -----
    Uses the parameter CROWDSAP included with some TESS data.
    flux_corrected = (flux - (1 - crowdsap)) / crowdsap
    where all quantities are median-normalised, including the result.
    This corresponds to subtracting a fraction of (1 - crowdsap) of third light
    from the (non-median-normalised) flux measurements.
    """
    cor_flux = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        cor_flux[s[0]:s[1]] = (flux[s[0]:s[1]] - 1 + crowd) / crowd

    return cor_flux


@nb.njit(cache=True)
def model_crowdsap(flux, crowdsap, i_chunks):
    """Incorporate flux contribution of a third source into the flux

    Parameters
    ----------
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    crowdsap: list[float], numpy.ndarray[Any, dtype[float]]
        Light contamination parameter (1-third_light) listed per sector
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Model of the flux incorporating light contamination

    Notes
    -----
    Does the opposite as correct_for_crowdsap, to be able to model the effect of
    third light to some degree (can only achieve an upper bound on CROWDSAP).
    """
    model = np.zeros(len(flux))
    for i, s in enumerate(i_chunks):
        crowd = min(max(0., crowdsap[i]), 1.)  # clip to avoid unphysical output
        model[s[0]:s[1]] = flux[s[0]:s[1]] * crowd + 1 - crowd

    return model


@nb.njit(cache=True)
def formal_uncertainties_linear(time, residuals, i_chunks):
    """Calculates the corrected uncorrelated (formal) uncertainties for the
    parameters constant and slope.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        sigma_const: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the constant for each sector
        sigma_slope: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the slope for each sector

    Notes
    -----
    Errors in const and slope:
    https://pages.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
    """
    n_param = 2

    # linear regression uncertainties
    sigma_const = np.zeros(len(i_chunks))
    sigma_slope = np.zeros(len(i_chunks))
    for i, s in enumerate(i_chunks):
        len_t = len(time[s[0]:s[1]])
        n_data = len(residuals[s[0]:s[1]])  # same as len_t, but just for the sake of clarity
        n_dof = n_data - n_param  # degrees of freedom

        # standard deviation of the residuals but per sector
        std = std_unb(residuals[s[0]:s[1]], n_dof)

        # some sums for the uncertainty formulae
        sum_t = 0
        for t in time[s[0]:s[1]]:
            sum_t += t
        ss_xx = 0
        for t in time[s[0]:s[1]]:
            ss_xx += (t - sum_t / len_t)**2
        sigma_const[i] = std * np.sqrt(1 / n_data + (sum_t / len_t)**2 / ss_xx)
        sigma_slope[i] = std / np.sqrt(ss_xx)

    return sigma_const, sigma_slope


@nb.njit(cache=True)
def formal_uncertainties_sinusoid(time, residuals, flux_err, a_n, i_chunks):
    """Calculates the corrected uncorrelated (formal) uncertainties for the frequencies, amplitudes, and phases.

    Harmonics are not taken into account.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sinusoids
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        sigma_f: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the frequency for each sinusoid.
        sigma_a: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the amplitude for each sinusoid (these are identical).
        sigma_ph: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the phase for each sinusoid.

    Notes
    -----
    As in Aerts 2021, https://ui.adsabs.harvard.edu/abs/2021RvMP...93a5001A/abstract
    The sigma value in the formulae is approximated by taking the maximum of the
    standard deviation of the residuals and the standard error of the minimum data error.
    """
    n_data = len(residuals)
    n_param = n_parameters(len(i_chunks), len(a_n), 0)  # number of parameters in the model
    n_dof = max(n_data - n_param, 1)  # degrees of freedom

    # calculate the standard deviation of the residuals
    std = std_unb(residuals, n_dof)

    # calculate the standard error based on the smallest data error
    ste = np.median(flux_err) / np.sqrt(n_data)

    # take the maximum of the standard deviation and standard error as sigma N
    sigma_n = max(std, ste)

    # calculate the D factor (square root of the average number of consecutive data points of the same sign)
    positive = (residuals > 0).astype(np.int_)
    indices = np.arange(n_data)
    zero_crossings = indices[1:][np.abs(positive[1:] - positive[:-1]).astype(np.bool_)]
    sss_i = np.concatenate((np.array([0]), zero_crossings, np.array([n_data])))  # same-sign sequence indices
    d_factor = np.sqrt(np.mean(np.diff(sss_i)))

    # uncertainty formulae for sinusoids
    sigma_f = d_factor * sigma_n * np.sqrt(6 / n_data) / (np.pi * a_n * np.ptp(time))
    sigma_a = d_factor * sigma_n * np.sqrt(2 / n_data)
    sigma_ph = d_factor * sigma_n * np.sqrt(2 / n_data) / a_n  # times 2 pi w.r.t. the paper

    # make an array of sigma_a (these are the same)
    sigma_a = np.full(len(a_n), sigma_a)

    return sigma_f, sigma_a, sigma_ph


def formal_uncertainties_harmonic(sigma_f, h_base, h_mult):
    """Calculates the uncertainty in the harmonic frequencies based on the unconstrained frequency
    uncertainties using a simple weighted average.

    The base frequency's uncertainty is calculated as:
    σ_h1 = (∑ n^2/σ_fn^2)^(-1/2)
    The other uncertainties follow as:
    σ_hn = n * σ_h1

    Parameters
    ----------
    sigma_f: numpy.ndarray[Any, dtype[float]]
        Uncertainty in the frequency for each sinusoid.
    h_base: numpy.ndarray[Any, dtype[int]]
        Indices of the base frequencies in f_n for each of f_n.
    h_mult: numpy.ndarray[Any, dtype[int]]
        Harmonic multiplier of the base frequency for each of f_n.

    Returns
    -------
    numpy.ndarray[Any, dtype[float]]
        Uncertainty in the harmonic frequencies.
    """
    # calculate the base frequency uncertainty
    sigma_h = np.zeros(len(sigma_f))
    for i_base in h_base[h_mult == 1]:
        h_series = h_base == i_base  # select one harmonic series at a time
        sigma_h[h_series] = np.sum(h_mult[h_series] ** 2 / sigma_f[h_series] ** 2) ** (-1 / 2)

    # multiply by the harmonic number
    sigma_h *= h_mult

    return sigma_h


@nb.njit(cache=True)
def formal_uncertainties(time, residuals, flux_err, a_n, i_chunks):
    """Calculates the corrected uncorrelated (formal) uncertainties for the extracted
    parameters (constant, slope, frequencies, amplitudes and phases).

    Harmonics are not taken into account.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    i_chunks: numpy.ndarray[Any, dtype[int]]
        Pair(s) of indices indicating time chunks within the light curve, separately handled in cases like
        the piecewise-linear curve. If only a single curve is wanted, set to np.array([[0, len(time)]]).

    Returns
    -------
    tuple
        A tuple containing the following elements:
        sigma_const: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the constant for each sector
        sigma_slope: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the slope for each sector
        sigma_f: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the frequency for each sine wave
        sigma_a: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the amplitude for each sine wave (these are identical)
        sigma_ph: numpy.ndarray[Any, dtype[float]]
            Uncertainty in the phase for each sine wave

    Notes
    -----
    As in Aerts 2021, https://ui.adsabs.harvard.edu/abs/2021RvMP...93a5001A/abstract
    The sigma value in the formulae is approximated by taking the maximum of the
    standard deviation of the residuals and the standard error of the minimum data error.
    Errors in const and slope:
    https://pages.mtu.edu/~fmorriso/cm3215/UncertaintySlopeInterceptOfLeastSquaresFit.pdf
    """
    n_data = len(residuals)
    n_param = n_parameters(len(i_chunks), len(a_n), 0)  # number of parameters in the model
    n_dof = max(n_data - n_param, 1)  # degrees of freedom

    # calculate the standard deviation of the residuals
    std = std_unb(residuals, n_dof)

    # calculate the standard error based on the smallest data error
    ste = np.median(flux_err) / np.sqrt(n_data)

    # take the maximum of the standard deviation and standard error as sigma N
    sigma_n = max(std, ste)

    # calculate the D factor (square root of the average number of consecutive data points of the same sign)
    positive = (residuals > 0).astype(np.int_)
    indices = np.arange(n_data)
    zero_crossings = indices[1:][np.abs(positive[1:] - positive[:-1]).astype(np.bool_)]
    sss_i = np.concatenate((np.array([0]), zero_crossings, np.array([n_data])))  # same-sign sequence indices
    d_factor = np.sqrt(np.mean(np.diff(sss_i)))

    # uncertainty formulae for sinusoids
    sigma_f = d_factor * sigma_n * np.sqrt(6 / n_data) / (np.pi * a_n * np.ptp(time))
    sigma_a = d_factor * sigma_n * np.sqrt(2 / n_data)
    sigma_ph = d_factor * sigma_n * np.sqrt(2 / n_data) / a_n  # times 2 pi w.r.t. the paper

    # make an array of sigma_a (these are the same)
    sigma_a = np.full(len(a_n), sigma_a)

    # linear regression uncertainties
    sigma_const, sigma_slope = formal_uncertainties_linear(time, residuals, i_chunks)

    return sigma_const, sigma_slope, sigma_f, sigma_a, sigma_ph


def linear_regression_uncertainty_ephem(time, p_orb, sigma_t=1):
    """Calculates the linear regression errors on period and t_zero

    Parameters
    ---------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series.
    p_orb: float
        Orbital period in days.
    sigma_t: float
        Error in the individual time measurements.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        p_err: float
            Error in the period.
        t_err: float
            Error in t_zero.
        p_t_corr: float
            Covariance between the period and t_zero.

    Notes
    -----
    The number of eclipses, computed from the period and
    time base, is taken to be a contiguous set.
    var_matrix:
    [[std[0]**2          , std[0]*std[1]*corr],
     [std[0]*std[1]*corr,           std[1]**2]]
    """
    # number of observed eclipses (technically contiguous)
    n = int(abs(np.ptp(time) // p_orb)) + 1

    # the arrays
    x = np.arange(n, dtype=int)  # 'time' points
    y = np.ones(n, dtype=int)  # 'positive measurement'

    # remove points in gaps
    gaps = mark_gaps(time, min_gap=1.)
    mask = mask_between(x * p_orb, gaps)  # convert x to time domain
    x = x[~mask] - n//2  # also centre the time for minimal correlation
    y = y[~mask]

    # M
    matrix = np.column_stack((x, y))

    # M^-1
    matrix_inv = np.linalg.pinv(matrix)  # inverse (of a general matrix)

    # M^-1 S M^-1^T, S unit matrix times some sigma (no covariance in the data)
    var_matrix = matrix_inv @ matrix_inv.T
    var_matrix = var_matrix * sigma_t ** 2

    # errors in the period and t_zero
    p_err = np.sqrt(var_matrix[0, 0])
    t_err = np.sqrt(var_matrix[1, 1])
    p_t_corr = var_matrix[0, 1] / (t_err * p_err)  # or [1, 0]

    return p_err, t_err, p_t_corr
