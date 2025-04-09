"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains functions for data analysis;
specifically for the fitting of stellar oscillations and harmonic sinusoids.

Code written by: Luc IJspeert
"""

import numpy as np
import numba as nb
import itertools as itt

from . import utility as ut


@nb.njit(cache=True)
def f_within_rayleigh(i, f_n, rayleigh):
    """Selects a chain of frequencies within the Rayleigh criterion from each other
    around the chosen frequency.
    
    Parameters
    ----------
    i: int
        Index of the frequency around which to search
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    rayleigh: float
        The appropriate frequency resolution (usually 1.5/T)
    
    Returns
    -------
    numpy.ndarray[Any, dtype[int]]
        Indices of close frequencies in the chain
    """
    indices = np.arange((len(f_n)))
    sorter = np.argsort(f_n)  # first sort by frequency
    f_diff = np.diff(f_n[sorter])  # spaces between frequencies
    sorted_pos = indices[sorter == i][0]  # position of i in the sorted array
    if np.all(f_diff > rayleigh):
        # none of the frequencies are close
        i_close = np.zeros(0, dtype=np.int_)
    elif np.all(f_diff < rayleigh):
        # all the frequencies are close
        i_close = indices
    else:
        # the frequency to investigate is somewhere inbetween other frequencies
        right_not_close = indices[sorted_pos + 1:][f_diff[sorted_pos:] > rayleigh]
        left_not_close = indices[:sorted_pos][f_diff[:sorted_pos] > rayleigh]
        # if any freqs to left or right are not close, take the first index where this happens
        if len(right_not_close) > 0:
            i_right_nc = right_not_close[0]
        else:
            i_right_nc = len(f_n)  # else take the right edge of the array
        if len(left_not_close) > 0:
            i_left_nc = left_not_close[-1]
        else:
            i_left_nc = -1  # else take the left edge of the array (minus one)
        # now select the frequencies close to f_n[i] by using the found boundaries
        i_close = indices[i_left_nc + 1:i_right_nc]
    # convert back to unsorted indices
    i_close_unsorted = sorter[i_close]
    return i_close_unsorted


@nb.njit(cache=True)
def chains_within_rayleigh(f_n, rayleigh):
    """Find all chains of frequencies within each other's Rayleigh criterion.
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    rayleigh: float
        The appropriate frequency resolution (usually 1.5/T)
    
    Returns
    -------
    list[numpy.ndarray[Any, dtype[int]]]
        Indices of close frequencies in all found chains
    
    See Also
    --------
    f_within_rayleigh
    """
    indices = np.arange(len(f_n))
    used = []
    groups = []
    for i in indices:
        if i not in used:
            i_close = f_within_rayleigh(i, f_n, rayleigh)
            if len(i_close) > 1:
                used.extend(i_close)
                groups.append(i_close)
    return groups


@nb.njit(cache=True)
def remove_insignificant_sigma(f_n, f_n_err, a_n, a_n_err, sigma_a=3., sigma_f=1.):
    """Removes insufficiently significant frequencies in terms of error margins.
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors in the frequencies
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    a_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors in the amplitudes
    sigma_a: float
        Number of times the error to use for check of significant amplitude
    sigma_f: float
        Number of times the error to use for check of significant
        frequency separation
    
    Returns
    -------
    numpy.ndarray[Any, dtype[int]]
        Indices of frequencies deemed insignificant
    
    Notes
    -----
    Frequencies with an amplitude less than sigma times the error are removed,
    as well as those that have an overlapping frequency error and are lower amplitude
    than any of the overlapped frequencies.
    """
    # amplitude not significant enough
    a_insig = (a_n / a_n_err < sigma_a)
    # frequency error overlaps with neighbour
    f_insig = np.zeros(len(f_n), dtype=np.bool_)
    for i in range(len(f_n)):
        overlap = (f_n[i] + sigma_f * f_n_err[i] > f_n) & (f_n[i] - sigma_f * f_n_err[i] < f_n)
        # if any of the overlap is higher in amplitude, throw this one out
        if np.any((a_n[overlap] > a_n[i]) & (f_n[overlap] != f_n[i])):
            f_insig[i] = True
    remove = np.arange(len(f_n))[a_insig | f_insig]
    return remove


@nb.njit(cache=True)
def remove_insignificant_snr(a_n, noise_at_f, n_points):
    """Removes insufficiently significant frequencies in terms of S/N.
    
    Parameters
    ----------
    a_n: numpy.ndarray[Any, dtype[float]]
        The amplitudes of a number of sine waves
    noise_at_f: numpy.ndarray[Any, dtype[float]]
        The noise level at each frequency
    n_points: int
        Number of data points
    
    Returns
    -------
    numpy.ndarray[Any, dtype[int]]
        Indices of frequencies deemed insignificant
    
    Notes
    -----
    Frequencies with an amplitude less than the S/N threshold are removed,
    using a threshold appropriate for TESS as function of the number of
    data points.
    
    The noise_at_f here captures the amount of noise on fitting a
    sinusoid of a certain frequency to all data points.
    Not to be confused with the noise on the individual data points of the
    time series.
    """
    snr_threshold = ut.signal_to_noise_threshold(n_points)

    # signal-to-noise below threshold
    a_insig_1 = (a_n / noise_at_f < snr_threshold)
    remove = np.arange(len(a_n))[a_insig_1]

    return remove


@nb.njit(cache=True)
def find_harmonics(f_n, f_n_err, p_orb, sigma=1.):
    """Find the orbital harmonics from a set of frequencies, given the orbital period.
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors in the frequencies
    p_orb: float
        The orbital period
    sigma: float
        Number of times the error to use for check of significance
    
    Returns
    -------
    numpy.ndarray[bool]
        Indices of frequencies that are harmonics of p_orb
    
    Notes
    -----
    Only includes those frequencies that are within sigma * error of an orbital harmonic.
    If multiple frequencies correspond to one harmonic, only the closest is kept.
    """
    # the frequencies divided by the orbital frequency gives integers for harmonics
    test_int = f_n * p_orb
    is_harmonic = ((test_int % 1) > 1 - sigma * f_n_err * p_orb) | ((test_int % 1) < sigma * f_n_err * p_orb)
    # look for harmonics that have multiple close frequencies
    harmonic_f = f_n[is_harmonic]
    sorter = np.argsort(harmonic_f)
    harmonic_n = np.round(test_int[is_harmonic], 0, np.zeros(np.sum(is_harmonic)))  # third arg needed for numba
    n_diff = np.diff(harmonic_n[sorter])
    # only keep the closest frequencies
    if np.any(n_diff == 0):
        n_dup = np.unique(harmonic_n[sorter][:-1][n_diff == 0])
        for n in n_dup:
            is_harmonic[np.round(test_int, 0, np.zeros(len(test_int))) == n] = False
            is_harmonic[np.argmin(np.abs(test_int - n))] = True
    i_harmonic = np.arange(len(f_n))[is_harmonic]
    return i_harmonic


@nb.njit(cache=True)
def construct_harmonic_range(f_0, domain):
    """create a range of harmonic frequencies given the base frequency.
    
    Parameters
    ----------
    f_0: float
        Base frequency in the range, from where the rest of the pattern is built.
    domain: list[float], numpy.ndarray[2, dtype[float]]
        Two values that give the borders of the range.
        Sensible values could be the Rayleigh criterion and the Nyquist frequency
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        harmonics: numpy.ndarray[Any, dtype[float]]
            Frequencies of the harmonic series in the domain
        n_range: numpy.ndarray[Any, dtype[int]]
            Corresponding harmonic numbers (base frequency is 1)
    """
    # determine where the range of harmonics starts and ends
    n_start = np.ceil(domain[0] / f_0)
    n_end = np.floor(domain[1] / f_0)
    n_range = np.arange(max(1, n_start), n_end + 1).astype(np.int_)
    harmonics = f_0 * n_range
    return harmonics, n_range


@nb.njit(cache=True)
def find_harmonics_from_pattern(f_n, p_orb, f_tol=1e-9):
    """Get the indices of the frequencies matching closest to the harmonics.
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        harmonics: numpy.ndarray[Any, dtype[int]]
            Indices of the frequencies in f_n that are deemed harmonics
        harmonic_n: numpy.ndarray[Any, dtype[int]]
            Corresponding harmonic numbers (base frequency is 1)
    
    Notes
    -----
    A frequency is only accepted as harmonic if it is within 1e-9 of the pattern
    (by default). This can now be user defined for more flexibility.
    """
    # guard against zero period or empty list
    if (p_orb == 0) | (len(f_n) == 0):
        harmonics = np.zeros(0, dtype=np.int_)
        harmonic_n = np.zeros(0, dtype=np.int_)
        return harmonics, harmonic_n
        
    # make the pattern of harmonics
    domain = [0, np.max(f_n) + 0.5 / p_orb]
    harmonic_pattern, harmonic_n = construct_harmonic_range(1 / p_orb, domain)
    # sort the frequencies
    sorter = np.argsort(f_n)
    f_n = f_n[sorter]
    # get nearest neighbour in harmonics for each f_n by looking to the left and right of the sorted position
    i_nn = np.searchsorted(f_n, harmonic_pattern)
    i_nn[i_nn == len(f_n)] = len(f_n) - 1
    closest = np.abs(f_n[i_nn] - harmonic_pattern) < np.abs(harmonic_pattern - f_n[i_nn - 1])
    i_nn = i_nn * closest + (i_nn - 1) * np.invert(closest)
    # get the distances to nearest neighbours
    d_nn = np.abs(f_n[i_nn] - harmonic_pattern)
    # check that the closest neighbours are reasonably close to the harmonic
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance must be smaller than tolerance (and never larger than 1/2P)
    # keep the ones left over
    harmonics = sorter[i_nn[m_cn]]
    harmonic_n = harmonic_n[m_cn]
    return harmonics, harmonic_n


@nb.njit(cache=True)
def find_harmonics_tolerance(f_n, p_orb, f_tol):
    """Get the indices of the frequencies matching within a tolerance to the harmonics.
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        harmonics: numpy.ndarray[Any, dtype[int]]
            Indices of the frequencies in f_n that are deemed harmonics
        harmonic_n: numpy.ndarray[Any, dtype[int]]
            Corresponding harmonic numbers (base frequency is 1)
    
    Notes
    -----
    A frequency is only accepted as harmonic if it is within some relative error.
    This can be user defined for flexibility.
    """
    harmonic_n = np.zeros(len(f_n))
    harmonic_n = np.round(f_n * p_orb, 0, harmonic_n)  # closest harmonic (out argument needed in numba atm)
    harmonic_n[harmonic_n == 0] = 1  # avoid zeros resulting from large f_tol
    # get the distances to the nearest pattern frequency
    d_nn = np.abs(f_n - harmonic_n / p_orb)
    # check that the closest neighbours are reasonably close to the harmonic
    m_cn = (d_nn < min(f_tol, 1 / (2 * p_orb)))  # distance smaller than tolerance (or half the harmonic spacing)
    # produce indices and make the right selection
    harmonics = np.arange(len(f_n))[m_cn]
    harmonic_n = harmonic_n[m_cn].astype(np.int_)
    return harmonics, harmonic_n


@nb.njit(cache=True)
def select_harmonics_sigma(f_n, f_n_err, p_orb, f_tol, sigma_f=3):
    """Selects only those frequencies that are probably harmonics

    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors in the frequencies
    p_orb: float
        The orbital period
    f_tol: float
        Tolerance in the frequency for accepting harmonics
    sigma_f: float
        Number of times the error to use for check of significant
        frequency separation
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
         harmonics_passed: numpy.ndarray[Any, dtype[int]]
            Indices of the frequencies in f_n that are harmonics
         harmonic_n: numpy.ndarray[Any, dtype[int]]
            Corresponding harmonic numbers (base frequency is 1)

    Notes
    -----
    A frequency is only accepted as harmonic if it is within the
    frequency resolution of the pattern, and if it is within <sigma_f> sigma
    of the frequency uncertainty
    """
    # get the harmonics within f_tol
    harmonics, harm_n = find_harmonics_from_pattern(f_n, p_orb, f_tol=f_tol)
    # frequency error overlaps with theoretical harmonic
    passed_h = np.zeros(len(harmonics), dtype=np.bool_)
    for i, (h, n) in enumerate(zip(harmonics, harm_n)):
        f_theo = n / p_orb
        margin = sigma_f * f_n_err[h]
        overlap_h = (f_n[h] + margin > f_theo) & (f_n[h] - margin < f_theo)
        if overlap_h:
            passed_h[i] = True
    harmonics_passed = harmonics[passed_h]
    harmonic_n = harm_n[passed_h]
    return harmonics_passed, harmonic_n


# @nb.njit()  # won't work due to itertools
def find_combinations(f_n, f_n_err, sigma=1.):
    """Find linear combinations from a set of frequencies.
    
    Parameters
    ----------
    f_n: list[float], numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors on the frequencies
    sigma: float
        Number of times the error to use for check of significance
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        final_o2: dict[int]
            Dictionary containing the indices of combinations of order 2
        final_o3: dict[int]
            Dictionary containing the indices of combinations of order 3

    Notes
    -----
    Only includes those frequencies that are within sigma * error of a linear combination.
    Does 2nd and 3rd order combinations. The number of sigma tolerance can be specified.
    """
    indices = np.arange(len(f_n))
    # make all combinations
    comb_order_2 = np.array(list(itt.combinations_with_replacement(indices, 2)))  # combinations of order 2
    comb_order_3 = np.array(list(itt.combinations_with_replacement(indices, 3)))  # combinations of order 2
    comb_freqs_o2 = np.sum(f_n[comb_order_2], axis=1)
    comb_freqs_o3 = np.sum(f_n[comb_order_3], axis=1)
    # check if any of the frequencies is a combination within error
    final_o2 = {}
    final_o3 = {}
    for i in indices:
        match_o2 = (f_n[i] > comb_freqs_o2 - sigma * f_n_err[i]) & (f_n[i] < comb_freqs_o2 + sigma * f_n_err[i])
        match_o3 = (f_n[i] > comb_freqs_o3 - sigma * f_n_err[i]) & (f_n[i] < comb_freqs_o3 + sigma * f_n_err[i])
        if np.any(match_o2):
            final_o2[i] = comb_order_2[match_o2]
        if np.any(match_o3):
            final_o3[i] = comb_order_3[match_o3]
    return final_o2, final_o3


def find_unknown_harmonics(f_n, f_n_err, sigma=1., n_max=5, f_tol=None):
    """Try to find harmonic series of unknown base frequency
    
    Parameters
    ----------
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    f_n_err: numpy.ndarray[Any, dtype[float]]
        Formal errors on the frequencies
    sigma: float
        Number of times the error to use for check of significance
    n_max: int
        Maximum divisor for each frequency in search of a base harmonic
    f_tol: None, float
        Tolerance in the frequency for accepting harmonics
        If None, use sigma matching instead of pattern matching
    
    Returns
    -------
    dict[int]
        Dictionary containing dictionaries with the indices of harmonic series
    
    Notes
    -----
    The first layer of the dictionary has the indices of frequencies as keys,
    the second layer uses n as keys and values are the indices of the harmonics
    stored in an array.
    n denotes the integer by which the frequency in question (index of the
    first layer) is divided to get the base frequency of the series.
    """
    indices = np.arange(len(f_n))
    n_harm = np.arange(1, n_max + 1)  # range of harmonic number that is tried for each frequency
    # test all frequencies for being the n-th harmonic in a series of harmonics
    candidate_h = {}
    for i in indices:
        for n in n_harm:
            p_base = n / f_n[i]  # 1/(f_n[i]/n)
            if f_tol is not None:
                i_harmonic, _ = find_harmonics_from_pattern(f_n, p_base, f_tol=f_tol)
            else:
                i_harmonic = find_harmonics(f_n, f_n_err, p_base, sigma=sigma)  # harmonic indices
            if len(i_harmonic) > 1:
                # don't allow any gaps of more than 20 + the number of preceding harmonics
                set_i = np.arange(len(i_harmonic))
                set_sorter = np.argsort(f_n[i_harmonic])
                harm_n = np.rint(np.sort(f_n[i_harmonic][set_sorter]) / (f_n[i] / n))  # harmonic integer n
                large_gaps = (np.diff(harm_n) > 20 + set_i[:-1])
                if np.any(large_gaps):
                    cut_off = set_i[1:][large_gaps][0]
                    i_harmonic = i_harmonic[set_sorter][:cut_off]
                # only take sets that don't have frequencies lower than f_n[i]
                cond_1 = np.all(f_n[i_harmonic] >= f_n[i])
                cond_2 = (len(i_harmonic) > 1)  # check this again after cutting off gap
                if cond_1 & cond_2 & (i in candidate_h.keys()):
                    candidate_h[i][n] = i_harmonic
                elif cond_1 & cond_2:
                    candidate_h[i] = {n: i_harmonic}
    # check for conditions that only look at the set itself
    i_n_remove = []
    for i in candidate_h.keys():
        for n in candidate_h[i].keys():
            set_len = len(candidate_h[i][n])
            # determine the harmonic integer n of each frequency in the set
            harm_n = np.rint(np.sort(f_n[candidate_h[i][n]]) / (f_n[i] / n))
            # remove sets of two where n is larger than two
            cond_1 = (set_len == 2) & (n > 2)
            # remove sets of three where n is larger than three
            cond_2 = (set_len == 3) & (n > 3)
            # remove sets where all gaps are larger than three
            cond_3 = np.all(np.diff(harm_n) > 3)
            # remove sets with large gap between the first and second frequency
            cond_4 = (np.diff(harm_n)[0] > 7)
            # also remove any sets with n>1 that are not longer than the one with n=1
            if cond_1 | cond_2 | cond_3 | cond_4:
                if [i, n] not in i_n_remove:
                    i_n_remove.append([i, n])
    # remove entries
    for i, n in i_n_remove:
        candidate_h[i].pop(n, None)
        if len(candidate_h[i]) == 0:
            candidate_h.pop(i, None)
    # check whether a series is fully contained in another (and other criteria involving other sets)
    i_n_redundant = []
    for i in candidate_h.keys():
        for n in candidate_h[i].keys():
            # sets of keys to compare current (i, n) to
            compare = np.array([[j, k] for j in candidate_h.keys() for k in candidate_h[j].keys()
                                if (j != i) | (k != n)])
            # check whether this set is fully contained in another
            this_contained = np.array([np.all(np.in1d(candidate_h[i][n], candidate_h[j][k])) for j, k in compare])
            # check whether another set is fully contained in this one
            other_contained = np.array([np.all(np.in1d(candidate_h[j][k], candidate_h[i][n])) for j, k in compare])
            # check for equal length ones
            equal_length = np.array([len(candidate_h[i][n]) == len(candidate_h[j][k]) for j, k in compare])
            # those that are fully contained and same length are equal
            equal = (equal_length & this_contained)
            # remove equals from contained list
            this_contained = (this_contained & np.invert(equal))
            other_contained = (other_contained & np.invert(equal))
            # check for sets with the same starting f (or multiple)
            f_i, e_i = f_n[i] / n, f_n_err[i] / n  # the base frequency and corresponding error
            same_start = np.array([np.any((f_i + e_i * sigma > f_n[candidate_h[j][k]])
                                          & (f_i - e_i * sigma < f_n[candidate_h[j][k]])) for j, k in compare])
            # if this set is contained in another (larger) set and n is larger or equal, it is redundant
            for j, k in compare[this_contained]:
                if (k <= n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if another set is contained in this (larger) one and n is larger, it is redundant
            for j, k in compare[other_contained]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if this set is equal to another set but has higher n, it is redundant (we keep lowest n)
            for j, k in compare[equal]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
            # if this set starts with the same base frequency as another, but has higher n, it is redundant
            for j, k in compare[same_start]:
                if (k < n) & ([i, n] not in i_n_redundant):
                    i_n_redundant.append([i, n])
    # remove redundant entries
    for i, n in i_n_redundant:
        candidate_h[i].pop(n, None)
        if len(candidate_h[i]) == 0:
            candidate_h.pop(i, None)
    return candidate_h


@nb.njit(cache=True)
def harmonic_series_length(f_test, f_n, freq_res, f_nyquist):
    """Find the number of harmonics that a set of frequencies has
    
    Parameters
    ----------
    f_test: numpy.ndarray[Any, dtype[float]]
        Frequencies to test at
    f_n: numpy.ndarray[Any, dtype[float]]
        The frequencies of a number of sine waves
    freq_res: float
        Frequency resolution
    f_nyquist: float
        Nyquist frequency
    
    Returns
    -------
    tuple
        A tuple containing the following elements:
        n_harm: numpy.ndarray[Any, dtype[float]]
            Number of harmonics per pattern
        completeness: numpy.ndarray[Any, dtype[float]]
            Completeness factor of each pattern
        distance: numpy.ndarray[Any, dtype[float]]
            Sum of squared distances between harmonics
    """
    n_harm = np.zeros(len(f_test))
    completeness = np.zeros(len(f_test))
    distance = np.zeros(len(f_test))
    for i, f in enumerate(f_test):
        harmonics, harmonic_n = find_harmonics_from_pattern(f_n, 1 / f, f_tol=freq_res / 2)
        n_harm[i] = len(harmonics)
        if n_harm[i] == 0:
            completeness[i] = 1
            distance[i] = 0
        else:
            completeness[i] = n_harm[i] / (f_nyquist // f)
            distance[i] = np.sum((f_n[harmonics] - harmonic_n * f)**2)
    return n_harm, completeness, distance


def linear_regression_uncertainty(p_orb, t_tot, sigma_t=1):
    """Calculates the linear regression errors on period and t_zero

    Parameters
    ---------
    p_orb: float
        Orbital period of the eclipsing binary in days
    t_tot: float
        Total time base of observations
    sigma_t: float
        Error in the individual time measurements

    Returns
    -------
    tuple
        A tuple containing the following elements:
        p_err: float
            Error in the period
        t_err: float
            Error in t_zero
        p_t_cov: float
            Covariance between the period and t_zero

    Notes
    -----
    The number of eclipses, computed from the period and
    time base, is taken to be a contiguous set.
    var_matrix:
    [[std[0]**2          , std[0]*std[1]*corr],
     [std[0]*std[1]*corr,           std[1]**2]]
    """
    # number of observed eclipses (technically contiguous)
    n = int(abs(t_tot // p_orb)) + 1
    # M
    matrix = np.column_stack((np.ones(n, dtype=int), np.arange(n, dtype=int)))
    # M^-1
    matrix_inv = np.linalg.pinv(matrix)  # inverse (of a general matrix)
    # M^-1 S M^-1^T, S unit matrix times some sigma (no covariance in the data)
    var_matrix = matrix_inv @ matrix_inv.T
    var_matrix = var_matrix * sigma_t ** 2
    # errors in the period and t_zero
    t_err = np.sqrt(var_matrix[0, 0])
    p_err = np.sqrt(var_matrix[1, 1])
    p_t_corr = var_matrix[0, 1] / (t_err * p_err)  # or [1, 0]
    return p_err, t_err, p_t_corr
