"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for goodness of fit determination.

Code written by: Luc IJspeert
"""

import numpy as np
import scipy as sp
import numba as nb

from star_shine.core import periodogram as pdg


@nb.njit(cache=True)
def calc_iid_normal_likelihood(residuals):
    """Natural logarithm of the independent and identically distributed likelihood function.

    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is estimated as σ^2 = sum((residuals)^2)/n

    Parameters
    ----------
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model

    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)
    # like = -n / 2 * (np.log(2 * np.pi * np.sum(residuals**2) / n) + 1)
    # originally un-JIT-ted function, but for loop is quicker with numba
    sum_r_2 = 0
    for i in range(n):
        sum_r_2 += residuals[i] ** 2

    like = -n / 2 * (np.log(2 * np.pi * sum_r_2 / n) + 1)

    return like


def calc_approx_did_likelihood(time, residuals):
    """Approximation for the likelihood using periodograms.

    This function approximates the dependent likelihood for correlated data.
    ln(L(θ)) =  -n ln(2 pi) - sum(ln(PSD(residuals)))

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model

    Returns
    -------
    float
        Log-likelihood approximation
    """
    n = len(time)

    # Compute the Lomb-Scargle periodogram of the data
    freqs, psd = pdg.scargle_parallel(time, residuals, f0=0, norm='psd')  # automatically mean subtracted

    # Compute the Whittle likelihood
    like = -n * np.log(2 * np.pi) - np.sum(np.log(psd))

    return like


def calc_whittle_likelihood(time, flux, model):
    """Whittle likelihood approximation using periodograms.

    This function approximates the dependent likelihood for correlated data.
    It assumes the data is identically distributed according to a normal
    distribution.
    ln(L(θ)) =  -n ln(2 pi) - sum(ln(PSD_model) + (PSD_data / PSD_model))

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    model: numpy.ndarray[Any, dtype[float]]
        Model values of the time series

    Returns
    -------
    float
        Log-likelihood approximation
    """
    n = len(time)

    # Compute the Lomb-Scargle periodogram of the data
    freqs, psd_d = pdg.scargle_parallel(time, flux, f0=0, norm='psd')  # automatically mean subtracted

    # Compute the Lomb-Scargle periodogram of the model
    freqs_m, psd_m = pdg.scargle_parallel(time, model, f0=0, norm='psd')  # automatically mean subtracted

    # Avoid division by zero in likelihood calculation
    psd_m = np.maximum(psd_m, 1e-15)  # Ensure numerical stability

    # Compute the Whittle likelihood
    like = -n * np.log(2 * np.pi) - np.sum(np.log(psd_m) + (psd_d / psd_m))

    return like


def calc_did_normal_likelihood(time, residuals):
    """Natural logarithm of the dependent and identically distributed likelihood function.

    Correlation in the data is taken into account. The data is still assumed to be
    identically distributed according to a normal distribution.

    ln(L(θ)) = -n ln(2 pi) / 2 - ln(det(∑)) / 2 - residuals @ ∑^-1 @ residuals^T / 2
    ∑ is the covariance matrix

    The covariance matrix is calculated using the power spectral density, following
    the Wiener–Khinchin theorem.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model

    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)

    # calculate the PSD, fast
    freqs, psd = pdg.scargle_parallel(time, residuals, f0=0, norm='psd')

    # calculate the autocorrelation function
    psd_ext = np.append(psd, psd[-1:0:-1])  # double the PSD domain for ifft
    acf = np.fft.ifft(psd_ext)

    # unbias the variance measure and put the array the right way around
    acf = np.real(np.append(acf[len(freqs):], acf[:len(freqs)])) * n / (n - 1)

    # calculate the acf lags
    lags = np.fft.fftfreq(len(psd_ext), d=(freqs[1] - freqs[0]))
    lags = np.append(lags[len(psd):], lags[:len(psd)])  # put them the right way around

    # make the lags matrix, but re-use the same matrix
    matrix = time - time[:, np.newaxis]  # lags_matrix, same as np.outer

    # interpolate - I need the lags at specific times
    matrix = np.interp(matrix, lags, acf)  # cov_matrix, already mean-subtracted in PSD

    # Compute the Cholesky decomposition of cov_matrix (by definition positive definite)
    matrix = sp.linalg.cho_factor(matrix, lower=False, overwrite_a=True, check_finite=False)  # cho_decomp

    # Solve M @ x = v^T using the Cholesky factorization (x = M^-1 v^T)
    x = sp.linalg.cho_solve(matrix, residuals[:, np.newaxis], check_finite=False)

    # log of the exponent - analogous to the matrix multiplication
    ln_exp = (residuals @ x)[0]  # v @ x = v @ M^-1 @ v^T

    # log of the determinant (avoids too small eigenvalues that would result in 0)
    ln_det = 2 * np.sum(np.log(np.diag(matrix[0])))

    # likelihood for multivariate normal distribution
    like = -n * np.log(2 * np.pi) / 2 - ln_det / 2 - ln_exp / 2

    return like


def calc_ddd_normal_likelihood(time, residuals, flux_err):
    """Natural logarithm of the dependent and differently distributed likelihood function.

    Only assumes that the data is distributed according to a normal distribution.
    Correlation in the data is taken into account. The measurement errors take
    precedence over the measured variance in the data. This means the distributions
    need not be identical, either.

    ln(L(θ)) = -n ln(2 pi) / 2 - ln(det(∑)) / 2 - residuals @ ∑^-1 @ residuals^T / 2
    ∑ is the covariance matrix

    The covariance matrix is calculated using the power spectral density, following
    the Wiener–Khinchin theorem.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    flux_err: numpy.ndarray[Any, dtype[float]]
        Errors in the measurement values

    Returns
    -------
    float
        Natural logarithm of the likelihood
    """
    n = len(residuals)

    # calculate the PSD, fast
    freqs, psd = pdg.scargle_parallel(time, residuals, f0=0, norm='psd')

    # calculate the autocorrelation function
    psd_ext = np.append(psd, psd[-1:0:-1])  # double the PSD domain for ifft
    acf = np.fft.ifft(psd_ext)

    # unbias the variance measure and put the array the right way around
    acf = np.real(np.append(acf[len(freqs):], acf[:len(freqs)])) * n / (n - 1)

    # calculate the acf lags
    lags = np.fft.fftfreq(len(psd_ext), d=(freqs[1] - freqs[0]))
    lags = np.append(lags[len(psd):], lags[:len(psd)])  # put them the right way around

    # make the lags matrix, but re-use the same matrix
    matrix = time - time[:, np.newaxis]  # lags_matrix, same as np.outer

    # interpolate - I need the lags at specific times
    matrix = np.interp(matrix, lags, acf)  # cov_matrix, already mean-subtracted in PSD

    # substitute individual data errors if given
    var = matrix[0, 0]  # diag elements are the same by construction
    corr_matrix = matrix / var  # divide out the variance to get correlation matrix
    err_matrix = flux_err * flux_err[:, np.newaxis]  # make matrix of measurement errors (same as np.outer)
    matrix = err_matrix * corr_matrix  # multiply to get back to covariance

    # Compute the Cholesky decomposition of cov_matrix (by definition positive definite)
    matrix = sp.linalg.cho_factor(matrix, lower=False, overwrite_a=True, check_finite=False)  # cho_decomp

    # Solve M @ x = v^T using the Cholesky factorization (x = M^-1 v^T)
    x = sp.linalg.cho_solve(matrix, residuals[:, np.newaxis], check_finite=False)

    # log of the exponent - analogous to the matrix multiplication
    ln_exp = (residuals @ x)[0]  # v @ x = v @ M^-1 @ v^T

    # log of the determinant (avoids too small eigenvalues that would result in 0)
    ln_det = 2 * np.sum(np.log(np.diag(matrix[0])))

    # likelihood for multivariate normal distribution
    like = -n * np.log(2 * np.pi) / 2 - ln_det / 2 - ln_exp / 2

    return like


@nb.njit(cache=True)
def calc_bic(residuals, n_param):
    """Bayesian Information Criterion.

    Parameters
    ----------
    residuals: numpy.ndarray[Any, dtype[float]]
        Residual is flux - model
    n_param: int
        Number of free parameters in the model

    Returns
    -------
    float
        Bayesian Information Criterion

    Notes
    -----
    BIC = −2 ln(L(θ)) + k ln(n)
    where L is the likelihood as function of the parameters θ, n the number of data points
    and k the number of free parameters.

    Under the assumption that the errors are independent and identically distributed
    according to a normal distribution, the likelihood becomes:
    ln(L(θ)) = -n/2 (ln(2 pi σ^2) + 1)
    and σ^2 is the error variance estimated as σ^2 = sum((residuals)^2)/n
    (residuals being data - model).

    Combining this gives:
    BIC = n ln(2 pi σ^2) + n + k ln(n)
    """
    n = len(residuals)

    sum_r_2 = 0
    for i in range(n):
        sum_r_2 += residuals[i]**2

    bic = n * np.log(2 * np.pi * sum_r_2 / n) + n + n_param * np.log(n)

    return bic
