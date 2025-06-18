"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains algorithms for data analysis.

Code written by: Luc IJspeert
"""
import numpy as np

from star_shine.core import time_series as tms, periodogram as pdg, fitting as fit
from star_shine.core import frequency_sets as frs, utility as ut
from star_shine.config.helpers import get_config


# load configuration
config = get_config()


def extract_single(time, flux, f0=-1, fn=-1, select='a'):
    """Extract a single sinusoid from a time series.

    The extracted frequency is based on the highest amplitude or signal-to-noise in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float, optional
        Lowest allowed frequency for extraction.
        If left -1, default is f0 = 1/(100*T)
    fn: float, optional
        Highest allowed frequency for extraction.
        If left -1, default is fn = 1/(2*np.min(np.diff(time))) = Nyquist frequency
    select: str, optional
        Select the next frequency based on amplitude 'a' or signal-to-noise 'sn'

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid

    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # full LS periodogram
    freqs, ampls = pdg.scargle_parallel(time, flux, f0=f0, fn=fn, df=df)

    # selection step based on flux to noise (refine step keeps using ampl)
    if select == 'sn':
        noise_spectrum = pdg.scargle_noise_spectrum_redux(freqs, ampls, window_width=1.0)
        ampls = ampls / noise_spectrum

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df/100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_local(time, flux, f0, fn):
    """Extract a single sinusoid from a time series at a predefined frequency interval.

    The extracted frequency is based on the highest amplitude in the periodogram.
    The highest peak is oversampled by a factor 100 to get a precise measurement.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f0: float
        Lowest allowed frequency for extraction.
    fn: float
        Highest allowed frequency for extraction.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid

    See Also
    --------
    scargle, scargle_phase_single
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # partial LS periodogram
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # cut off the ends of the frequency range if they are rising
    i_f_min_edges = ut.uphill_local_max(freqs, -ampls, freqs[np.array([0, -1])])
    freqs = freqs[i_f_min_edges[0]:i_f_min_edges[1] + 1]
    ampls = ampls[i_f_min_edges[0]:i_f_min_edges[1] + 1]

    # select highest amplitude
    i_f_max = np.argmax(ampls)

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)  # may not get too low
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase (and make sure it stays within + and - pi)
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def extract_approx(time, flux, f_approx):
    """Extract a single sinusoid from a time series at an approximate location.

    Follows the periodogram upwards to the nearest peak. The periodogram is oversampled for a more precise result.

    Parameters
    ----------
    time: numpy.ndarray[Any, dtype[float]]
        Timestamps of the time series
    flux: numpy.ndarray[Any, dtype[float]]
        Measurement values of the time series
    f_approx: float
        Approximate location of the frequency of maximum amplitude.

    Returns
    -------
    tuple
        A tuple containing the following elements:
        f_final: float
            Frequency of the extracted sinusoid
        a_final: float
            Amplitude of the extracted sinusoid
        ph_final: float
            Phase of the extracted sinusoid
    """
    df = 0.1 / np.ptp(time)  # default frequency sampling is about 1/10 of frequency resolution

    # LS periodogram around the approximate location (2.5 times freq res)
    f0 = max(f_approx - 25 * df, df / 10)
    fn = f_approx + 25 * df
    freqs, ampls = pdg.scargle(time, flux, f0=f0, fn=fn, df=df)

    # get the index of the frequency of the maximum amplitude
    i_f_max = ut.uphill_local_max(freqs, ampls, np.array([f_approx]))[0]

    # refine frequency by increasing the frequency resolution x100
    f_left = max(freqs[i_f_max] - df, df / 10)
    f_right = freqs[i_f_max] + df
    f_refine, a_refine = pdg.scargle(time, flux, f0=f_left, fn=f_right, df=df / 100)

    # select refined highest amplitude
    i_f_max = np.argmax(a_refine)
    f_final = f_refine[i_f_max]
    a_final = a_refine[i_f_max]

    # finally, compute the phase
    _, ph_final = pdg.scargle_ampl_phase_single(time, flux, f_final)

    return f_final, a_final, ph_final


def refine_subset(ts_model, close_f, logger=None):
    """Refine a subset of frequencies that are within the Rayleigh criterion of each other,
    taking into account (and not changing the frequencies of) harmonics if present.

    Intended as a sub-loop within another extraction routine (extract_sinusoids), can work standalone too.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    close_f: numpy.ndarray[Any, dtype[int]]
        Indices of the subset of frequencies to be refined
    logger: logging.Logger, optional
        Instance of the logging library.

    See Also
    --------
    extract_sinusoids
    """
    # get all the harmonics (regardless of base)
    i_harmonics = np.arange(len(ts_model.sinusoid.harmonics))[ts_model.sinusoid.harmonics]

    # determine initial bic
    bic_prev = ts_model.bic()

    # stop the loop when the BIC increases
    condition_1 = True
    while condition_1:
        # remember the current sinusoids
        f_c, a_c, ph_c = ts_model.sinusoid.get_sinusoid_parameters(exclude=False)

        # remove each frequency one at a time to then re-extract them
        for j in close_f:
            # exclude the sinusoid
            ts_model.exclude_sinusoids(j)
            # update the linear model for good measure
            ts_model.update_linear_model()

            # improve sinusoid j by re-extracting its parameters
            f_j = ts_model.sinusoid.f_n[j]
            if j in i_harmonics:
                # if f is a harmonic, don't shift the frequency
                a_j, ph_j = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_j)
            else:
                f_j, a_j, ph_j = extract_approx(ts_model.time, ts_model.residual(), f_j)

            # update the model
            ts_model.update_sinusoids(f_j, a_j, ph_j, j)

        # as a last model-refining step, redetermine the constant and slope
        ts_model.update_linear_model()

        # calculate BIC before moving to the next iteration
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # stop the loop when the BIC increases
        condition_1 = np.round(d_bic, 2) > 0

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            # update the sinusoids back to their original values
            ts_model.update_sinusoids(f_c, a_c, ph_c, close_f)
            ts_model.update_linear_model()

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - N_refined= {len(close_f)}",
                     extra={'update': True})

    return None

def replace_subset(ts_model, close_f, final_remove=True, logger=None):
    """Attempt the replacement of frequencies within the Rayleigh criterion of each other by a single one,
    taking into account (and not changing the frequencies of) harmonics if present.

    Intended as a sub-loop within another extraction routine (extract_sinusoids), can work standalone too.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    close_f: numpy.ndarray[Any, dtype[int]]
        Indices of the subset of frequencies to be (subdivided and) replaced.
    final_remove: bool, optional
        Remove the excluded frequencies at the end.
    logger: logging.Logger, optional
        Instance of the logging library.

    See Also
    --------
    extract_sinusoids
    """
    # standard frequency resolution (not the user defined one)
    freq_res = 1 / ts_model.t_tot
    # get all the harmonics (regardless of base)
    i_harmonics = np.arange(len(ts_model.sinusoid.harmonics))[ts_model.sinusoid.harmonics]

    # make all combinations of consecutive frequencies in close_f (longer sets first)
    close_f_sets = ut.consecutive_subsets(close_f)

    # determine initial quantities
    n_sin_tot_init = len(ts_model.sinusoid.f_n)
    n_excluded_init = np.sum(~ts_model.sinusoid.include)
    bic_prev = ts_model.bic()

    # loop over all subsets:
    for set_i in close_f_sets:
        # if set_i contains removed sinusoids, skip (order of sets matters)
        if np.any(~ts_model.sinusoid.include[set_i]):
            continue

        # exclude the next set of sinusoids
        ts_model.exclude_sinusoids(set_i)
        # update the linear model for good measure
        ts_model.update_linear_model()

        # check for harmonics
        harm_i = [h for h in set_i if h in i_harmonics]

        # remove all frequencies in the set and re-extract one
        f_c = ts_model.sinusoid.f_n  # current frequencies
        if len(harm_i) > 0:
            # if f is a harmonic, don't shift the frequency
            f_i = f_c[harm_i]  # can be more than one harmonic
            a_i, ph_i = pdg.scargle_ampl_phase(ts_model.time, ts_model.residual(), f_i)
        else:
            f0 = min(f_c[set_i]) - freq_res
            fn = max(f_c[set_i]) + freq_res
            f_i, a_i, ph_i = extract_local(ts_model.time, ts_model.residual(), f0=f0, fn=fn)

        # add sinusoid to the model
        ts_model.add_sinusoids(f_i, a_i, ph_i)
        # as a last model-refining step, redetermine the constant and slope
        ts_model.update_linear_model()

        # calculate BIC before moving to the next iteration
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # acceptance condition for replacement
        condition_1 = np.round(d_bic, 2) > 0

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the changes
            bic_prev = bic
        else:
            # remove the added sinusoid(s)
            ts_model.remove_sinusoids(np.arange(len(f_c), len(f_c) + len(np.atleast_1d(f_i))))

            # include the excluded sinusoids
            ts_model.include_sinusoids(set_i)
            ts_model.update_linear_model()

    # determine number of excluded
    n_excluded = np.sum(~ts_model.sinusoid.include[:n_sin_tot_init])
    n_replaced = n_excluded - n_excluded_init

    if final_remove:
        ts_model.remove_excluded()

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - N_replaced= {n_replaced}",
                     extra={'update': True})

    return None


def extract_sinusoids(ts_model, bic_thr=2, snr_thr=0, stop_crit='bic', select='hybrid', n_extract=0,
                      fit_each_step=False, replace_each_step=True, logger=None):
    """Extract all the frequencies from a periodic flux.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    bic_thr: float, optional
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    snr_thr: float, optional
        Threshold for signal-to-noise ratio for a signal to be considered significant.
    stop_crit: str, optional
        Use the BIC as stopping criterion or the SNR, choose from 'bic', or 'snr'
    select: str, optional
        Select the next frequency based on amplitude ('a'),
        signal-to-noise ('sn'), or hybrid ('hybrid') (first a then sn).
    n_extract: int, optional
        Maximum number of frequencies to extract. The stop criterion is still leading. Zero means as many as possible.
    fit_each_step: bool
        If set to True, a non-linear least-squares fit of all extracted sinusoids in groups is performed at each
        iteration. While this increases the quality of the extracted signals, it drastically slows down the code.
        Generally gives a better quality of the extraction than only doing this all the way at the end.
    replace_each_step: bool
        If set to True, close frequecies are attempted to be replaced by a single sinusoid at each iteration.
        May increase the quality of the extraction more than only doing this all the way at the end.
    logger: logging.Logger, optional
        Instance of the logging library.

    Notes
    -----
    Spits out frequencies and amplitudes in the same units as the input,
    and phases that are measured with respect to the first time point.
    Also determines the flux average, so this does not have to be subtracted
    before input into this function.
    Note: does not perform a non-linear least-squares fit at the end,
    which is highly recommended! (In fact, no fitting is done at all).

    The function optionally takes a pre-existing frequency list to append
    additional frequencies to. Set these to np.array([]) to start from scratch.

    i_chunks is a 2D array with start and end indices of each (half) sector.
    This is used to model a piecewise-linear trend in the data.
    If you have no sectors like the TESS mission does, set
    i_chunks = np.array([[0, len(time)]])

    Exclusively uses the Lomb-Scargle periodogram (and an iterative parameter
    improvement scheme) to extract the frequencies.
    Uses a delta BIC > bic_thr stopping criterion.

    [Author's note] Although it is my belief that doing a non-linear
    multi-sinusoid fit at each iteration of the prewhitening is the
    ideal approach, it is also a very (very!) time-consuming one and this
    algorithm aims to be fast while approaching the optimal solution.

    [Another author's note] I added an option to do the non-linear multi-
    sinusoid fit at each iteration.
    """
    if n_extract == 0:
        n_extract = 10**6  # 'a lot'

    # initial number of sinusoids
    n_sin_init = ts_model.sinusoid.n_sin

    # set up selection process
    if select == 'hybrid':
        switch = True  # when we would normally end, we switch strategy
        select = 'a'  # start with amplitude extraction
    else:
        switch = False

    # determine the initial bic
    bic_prev = ts_model.bic()  # initialise current BIC to the mean (and slope) subtracted flux

    # log a message
    if logger is not None:
        logger.extra(f"N_f= {n_sin_init}, BIC= {bic_prev:1.2f} - Iterative prewhitening")

    # stop the loop when the BIC decreases by less than 2 (or increases)
    condition_1 = True
    condition_2 = True
    while (condition_1 | switch) & condition_2:
        # switch selection method when extraction would normally stop
        if switch and not condition_1:
            select = 'sn'
            switch = False
            logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Switch selection from a to sn.")

        # remember the current sinusoids
        f_c, a_c, ph_c = ts_model.sinusoid.get_sinusoid_parameters(exclude=False)
        _, h_base, h_mult = ts_model.sinusoid.get_harmonic_parameters(exclude=False)

        # attempt to extract the next frequency
        f_i, a_i, ph_i = extract_single(ts_model.time, ts_model.residual(), f0=ts_model.pd_f0, fn=ts_model.pd_fn,
                                        select=select)
        ts_model.add_sinusoids(f_i, a_i, ph_i)

        # update the linear pars
        ts_model.update_linear_model()

        # improve sinusoids with some strategy
        if fit_each_step:
            # fit all sinusoids for best improvement
            fit.fit_multi_sinusoid_grouped(ts_model, logger=logger)
        else:
            # select only close frequencies for iteration
            close_f = frs.f_within_rayleigh(ts_model.sinusoid.n_sin - 1, ts_model.sinusoid.f_n, ts_model.f_resolution)

            if len(close_f) > 1:
                # iterate over (re-extract) close frequencies (around f_i) a number of times to improve them
                refine_subset(ts_model, close_f, logger=logger)

        # possibly replace close frequencies
        if replace_each_step:
            close_f = frs.f_within_rayleigh(ts_model.sinusoid.n_sin - 1, ts_model.sinusoid.f_n, ts_model.f_resolution)
            if len(close_f) > 1:
                replace_subset(ts_model, close_f, logger=logger)

        # calculate BIC
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # acceptance condition
        if stop_crit == 'snr':
            # calculate SNR in a 1 c/d window around the extracted frequency
            noise = pdg.scargle_noise_at_freq(np.array([f_i]), ts_model.time, ts_model.residual(), window_width=1.0)
            snr = a_i / noise
            # stop the loop if snr threshold not met
            condition_1 = snr > snr_thr
        else:
            # stop the loop when the BIC decreases by less than bic_thr (or increases)
            condition_1 = np.round(d_bic, 2) > bic_thr

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            ts_model.set_sinusoids(f_c, a_c, ph_c, h_base_new=h_base, h_mult_new=h_mult)
            ts_model.update_linear_model()

        # stop the loop if n_sin reaches limit
        condition_2 = ts_model.sinusoid.n_sin - n_sin_init < n_extract

        if logger is not None and condition_1:
            logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - Extracted: f= {f_i:1.6f}, a= {a_i:1.6f}",
                         extra={'update': True})

    if logger is not None:
        n_sin = ts_model.sinusoid.n_sin
        logger.extra(f"N_f= {n_sin}, BIC= {bic_prev:1.2f} - N_extracted= {n_sin - n_sin_init}.")

    return None


def couple_harmonics(ts_model, f_base, logger=None):
    """Finds matching harmonics in the sinusoids and couples their frequencies.

    Surrounding frequencies are also removed (within the frequency resolution).

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    f_base: float
        Base frequency to couple the harmonics to.
    logger: logging.Logger, optional
        Instance of the logging library.
    """
    # find the harmonic candidates using the period
    harmonics, h_mult = frs.find_harmonics_tolerance(ts_model.sinusoid.f_n, f_base, f_tol=ts_model.f_resolution / 2)

    if len(harmonics) == 0:
        if logger is not None:
            logger.warning("No harmonic frequencies found")

        return None

    # the index of f_base is len(f_n), because the base harmonic is the first frequency to be added at the end
    i_base = len(ts_model.sinusoid.f_n)

    # if the base frequency is not yet present, add it
    if 1 not in h_mult:
        a_1, ph_1 = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_base)
        ts_model.add_sinusoids(f_base, a_1, ph_1, h_base_new=i_base, h_mult_new=1)

    # determine initial quantities
    n_harm_init = len(harmonics)

    # go through the harmonics by harmonic number and re-extract them (n==1 must come first, if present)
    for n in np.unique(h_mult):
        # get the indices to exclude, disregarding included state
        n_sin_tot = len(ts_model.sinusoid.f_n)
        remove = np.arange(n_sin_tot)[harmonics][h_mult == n]

        # exclude the neighbouring harmonic candidates and update linear model
        ts_model.exclude_sinusoids(remove)
        ts_model.update_linear_model()

        # extract the harmonic amplitude and phase
        f_i = n * f_base
        a_i, ph_i = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_i)

        # add harmonic candidate and redetermine the constant and slope
        ts_model.add_sinusoids(f_i, a_i, ph_i, h_base_new=i_base, h_mult_new=n)

    # lastly re-determine slope and const and remove the excluded frequencies
    ts_model.remove_excluded()
    ts_model.update_linear_model()

    if logger is not None:
        bic = ts_model.bic()
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - N_coupled= {ts_model.sinusoid.n_harm}, "
                     f"N_harmonics_init= {n_harm_init}", extra={'update': True})

    return None


def extract_harmonics(ts_model, bic_thr=2, logger=None):
    """Tries to extract more harmonics from the flux

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    bic_thr: float
        The minimum decrease in BIC by fitting a sinusoid for the signal to be considered significant.
    logger: logging.Logger, optional
        Instance of the logging library.

    See Also
    --------
    fix_harmonic_frequency

    Notes
    -----
    Looks for missing harmonics and checks whether adding them decreases the BIC sufficiently (by more than 2).
    Assumes the harmonics are already fixed multiples of f_base as can be achieved with fix_harmonic_frequency.
    """
    # make lists of not-present possible harmonics paired with their base frequency
    i_base_all = []
    f_base_all = []
    h_candidates_n = []
    for i_base in np.unique(ts_model.sinusoid.h_base[ts_model.sinusoid.harmonics]):
        # the range of harmonic multipliers below the Nyquist frequency
        harmonics_i = np.arange(1, ts_model.pd_fn / ts_model.sinusoid.f_n[i_base], dtype=int)

        # h_mult minus one is the position for existing harmonics
        harmonics_i = np.delete(harmonics_i, ts_model.sinusoid.h_mult[ts_model.sinusoid.h_base == i_base] - 1)
        h_candidates_n.extend(harmonics_i)

        # add to the bae frequency lists
        i_base_all.extend([i_base for _ in range(len(harmonics_i))])
        f_base_all.extend([ts_model.sinusoid.f_n[i_base] for _ in range(len(harmonics_i))])

    # determine initial quantities
    n_sin_init = ts_model.sinusoid.n_sin
    bic_prev = ts_model.bic()  # initialise current BIC to the mean (and slope) subtracted flux

    if logger is not None:
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic_prev:1.2f} - Extract harmonics")

    # loop over candidates and try to extract (BIC decreases by 2 or more)
    for i in range(len(i_base_all)):
        f_i = h_candidates_n[i] * f_base_all[i]
        a_i, ph_i = pdg.scargle_ampl_phase_single(ts_model.time, ts_model.residual(), f_i)

        # add harmonic candidate and redetermine the constant and slope
        ts_model.add_sinusoids(f_i, a_i, ph_i, h_base_new=i_base_all[i], h_mult_new=h_candidates_n[i])
        ts_model.update_linear_model()

        # determine new BIC and whether it improved
        bic = ts_model.bic()
        d_bic = bic_prev - bic

        # stop the loop when the BIC decreases by less than bic_thr (or increases)
        condition_1 = np.round(d_bic, 2) > bic_thr

        # check acceptance condition before moving to the next iteration
        if condition_1:
            # accept the new frequency
            bic_prev = bic
        else:
            # h_c is rejected, revert to previous model
            ts_model.remove_sinusoids(len(ts_model.sinusoid.f_n) - 1)  # remove last added
            ts_model.update_linear_model()

        if logger is not None and condition_1:
            logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - Extracted: "
                         f"f_base= {f_base_all[i]:1.2f}, h= {h_candidates_n[i]}, f= {f_i:1.6f}, a= {a_i:1.6f}",
                         extra={'update': True})

    if logger is not None:
        n_sin = ts_model.sinusoid.n_sin
        logger.extra(f"N_f= {n_sin}, BIC= {bic_prev:1.2f} - N_h_extracted= {n_sin - n_sin_init}")

    return None


def replace_close_sinusoid_pair(ts_model):
    """Replace pairs of sinusoids that have gotten too close.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    """
    f_n = ts_model.sinusoid.f_n
    harmonics = ts_model.sinusoid.harmonics

    # get the pairs of close frequencies
    i_close = frs.f_close_pairs(f_n, df=ts_model.pd_df)

    # replace each pair, taking harmonics into account
    for pair in i_close:
        # exclude the pair
        ts_model.exclude_sinusoids(pair)

        # check for harmonic
        if np.any(harmonics[pair]):
            # if f is a harmonic, don't shift the frequency
            f_i = f_n[pair][harmonics[pair]]  # select the harmonic
            a_i, ph_i = pdg.scargle_ampl_phase(ts_model.time, ts_model.residual(), f_i)
        else:
            f0 = f_n[pair[0]] - ts_model.pd_df  # pairs are ordered
            fn = f_n[pair[1]] + ts_model.pd_df  # pairs are ordered
            f_i, a_i, ph_i = extract_local(ts_model.time, ts_model.residual(), f0=f0, fn=fn)

        # add sinusoid to the model
        ts_model.add_sinusoids(f_i, a_i, ph_i)
        # as a last model-refining step, redetermine the constant and slope
        ts_model.update_linear_model()

    # remove the excluded sinusoids
    ts_model.remove_excluded()

    return None


def remove_sinusoids_single(ts_model, logger=None):
    """Attempt the removal of individual sinusoids.

    Checks whether the BIC can be improved by removing a sinusoid. Harmonics are taken into account.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.
    """
    # determine initial quantities
    n_sin_init = len(ts_model.sinusoid.f_n)
    bic_prev = ts_model.bic()

    # while frequencies are added to the exclude list, continue loop
    n_sin = np.sum(ts_model.sinusoid.include)
    n_prev = n_sin + 1
    while n_sin < n_prev:
        n_prev = n_sin
        for i in range(n_sin_init):
            # continue if sinusoid is already excluded, or when it is a base harmonic
            if not ts_model.sinusoid.include[i] or ts_model.sinusoid.h_mult[i] == 1:
                continue

            # exclude sinusoid and redetermine the constant and slope
            ts_model.exclude_sinusoids(i)
            ts_model.update_linear_model()

            # determine new BIC and whether it improved
            bic = ts_model.bic()
            d_bic = bic_prev - bic

            # only remove sinusoid if it increases the BIC
            condition_1 = np.round(d_bic, 2) > 0

            # check acceptance condition before moving to the next iteration
            if condition_1:
                # accept the removal
                bic_prev = bic
                n_sin = np.sum(ts_model.sinusoid.include)
            else:
                # removal is rejected, revert to previous model
                ts_model.include_sinusoids(i)

    # lastly re-determine slope and const and remove the excluded frequencies
    ts_model.remove_excluded()
    ts_model.update_linear_model()

    if logger is not None:
        n_sin = ts_model.sinusoid.n_sin
        logger.extra(f"N_f= {n_sin}, BIC= {bic_prev:1.2f} - N_removed= {n_sin_init - n_sin}.", extra={'update': True})

    return None


def replace_sinusoid_groups(ts_model, logger=None):
    """Attempt the replacement of groups of frequencies by a single one.

    Checks whether the BIC can be improved by replacing a group of frequencies by only one.
    Harmonics are never removed.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.
    """
    # make an array of sets of frequencies to be investigated for replacement
    close_f_groups = frs.chains_within_rayleigh(ts_model.sinusoid.f_n, ts_model.f_resolution)

    # determine initial quantities
    n_sin_tot_init = len(ts_model.sinusoid.f_n)
    n_excluded_init = np.sum(~ts_model.sinusoid.include)

    # while frequencies are added to the exclude list, continue loop
    n_sin = np.sum(ts_model.sinusoid.include)
    n_prev = n_sin + 1
    while n_sin < n_prev:
        n_prev = n_sin
        for i, close_f in enumerate(close_f_groups):
            # continue if full sinusoid set is already excluded
            if np.all(~ts_model.sinusoid.include[close_f]):
                continue

            # use the replace_subset function to handle the details
            replace_subset(ts_model, close_f, final_remove=False, logger=None)

            # update number of sinusoids after replacement
            n_sin = np.sum(ts_model.sinusoid.include)

    # determine number of excluded and added
    n_excluded = np.sum(~ts_model.sinusoid.include[:n_sin_tot_init])
    n_replaced = n_excluded - n_excluded_init
    n_new = len(ts_model.sinusoid.f_n) - n_sin_tot_init

    # lastly re-determine slope and const and remove the excluded frequencies
    ts_model.remove_excluded()
    ts_model.update_linear_model()

    if logger is not None:
        bic = ts_model.bic()
        logger.extra(f"N_f= {ts_model.sinusoid.n_sin}, BIC= {bic:1.2f} - N_replaced= {n_replaced}, N_kept= {n_new}.",
                     extra={'update': True})

    return None


def reduce_sinusoids(ts_model, logger=None):
    """Attempt to reduce the number of frequencies taking into account any harmonics if present.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.

    Notes
    -----
    Checks whether the BIC can be improved by removing a frequency. Special attention is given to frequencies
    that are within the Rayleigh criterion of each other. It is attempted to replace these by a single frequency.
    """
    # first check if any frequency can be left out (after the fit, this may be possible)
    remove_sinusoids_single(ts_model, logger=logger)

    # Now go on to trying to replace sets of frequencies that are close together
    replace_sinusoid_groups(ts_model, logger=logger)

    return None


def select_sinusoids(ts_model, logger=None):
    """Selects the credible frequencies from the given set

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.
    logger: logging.Logger, optional
        Instance of the logging library.

    Notes
    -----
    Harmonic frequencies that are said to be passing the criteria are in fact passing the criteria for
    individual frequencies, not those for a set of harmonics (which would be a looser constraint).
    """
    n_sin = ts_model.sinusoid.n_sin

    # update the sinusoid errors in the model
    ts_model.update_sinusoid_uncertainties()
    ts_model.update_sinusoid_uncertainties_harmonic()

    # find the insignificant frequencies
    ts_model.update_sinusoid_passing_sigma()

    # apply the signal-to-noise threshold
    ts_model.update_sinusoid_passing_snr(window_width=config.window_width)

    # candidate harmonic frequencies passing criteria
    ts_model.update_sinusoid_passing_harmonic()

    if logger is not None:
        passed_all = (ts_model.sinusoid.passing_sigma & ts_model.sinusoid.passing_snr)
        n_pass_all = np.sum(passed_all)
        n_harm = np.sum(ts_model.sinusoid.passing_harmonic)
        n_harm_pass_all = np.sum(passed_all & ts_model.sinusoid.passing_harmonic)
        logger.extra(f"Sinusoids passing criteria: {n_pass_all} of {n_sin}. "
                     f"Harmonics passing criteria: {n_harm_pass_all} of {n_harm}.")

    return None


def refine_harmonic_base_frequency(f_base, ts_model):
    """Refine the base frequency for a harmonic sinusoid model.

    Parameters
    ----------
    f_base: float
        Base frequency of the harmonic series to refine.
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    Returns
    -------
    float
        Base frequency of the harmonic series.

    Notes
    -----
    Uses the sum of distances between the harmonics and their
    theoretical positions to refine the orbital period.

    Period precision is 0.00001, but accuracy is a bit lower.

    Same refine algorithm as used in find_orbital_period
    """
    freq_res = ts_model.f_resolution
    f_nyquist = ts_model.pd_fn
    f_n = ts_model.sinusoid.f_n

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 * f_base, 1.01 * f_base, 0.00001 * f_base)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    f_base = f_refine[mask_peak][i_min_dist]

    return f_base


def find_harmonic_base_frequency(ts_model):
    """Find the most likely base frequency for a harmonic sinusoid model.

    Parameters
    ----------
    ts_model: tms.TimeSeriesModel
        Instance of TimeSeriesModel containing the time series and model parameters.

    Returns
    -------
    float
        Base frequency of the harmonic series.

    Notes
    -----
    Uses a combination of phase dispersion minimisation and Lomb-Scargle periodogram (see Saha & Vivas 2017),
    and some refining steps to get the best frequency.

    Also tests various multiples of the period. Precision is 0.00001 (one part in one-hundred-thousand).
    (accuracy might be slightly lower)
    """
    freq_res = ts_model.f_resolution
    f_nyquist = ts_model.pd_fn
    f_n = ts_model.sinusoid.f_n

    # first to get a global minimum do combined PDM and LS, at select frequencies
    periods, phase_disp = pdg.phase_dispersion_minimisation(ts_model.time, ts_model.flux, f_n, local=False)
    freqs = 1 / periods
    ampls, _ = pdg.scargle_ampl_phase(ts_model.time, ts_model.flux, freqs)
    psi_measure = ampls / phase_disp

    # also check the number of harmonics at each period and include into best f
    n_harm, completeness, distance = frs.harmonic_series_length(freqs, f_n, freq_res, f_nyquist)
    psi_h_measure = psi_measure * n_harm * completeness

    # select the best period, refine it and check double P
    f_base = freqs[np.argmax(psi_h_measure)]

    # refine by using a dense sampling and the harmonic distances
    f_refine = np.arange(0.99 * f_base, 1.01 * f_base, 0.00001 * f_base)
    n_harm_r, completeness_r, distance_r = frs.harmonic_series_length(f_refine, f_n, freq_res, f_nyquist)
    h_measure = n_harm_r * completeness_r  # compute h_measure for constraining a domain
    mask_peak = (h_measure > np.max(h_measure) / 1.5)  # constrain the domain of the search
    i_min_dist = np.argmin(distance_r[mask_peak])
    f_base = f_refine[mask_peak][i_min_dist]

    # reduce the search space by taking limits in the distance metric
    f_left = f_refine[mask_peak][:i_min_dist]
    f_right = f_refine[mask_peak][i_min_dist:]
    d_left = distance_r[mask_peak][:i_min_dist]
    d_right = distance_r[mask_peak][i_min_dist:]
    d_max = np.max(distance_r)
    if np.any(d_left > d_max / 2):
        f_l_bound = f_left[d_left > d_max / 2][-1]
    else:
        f_l_bound = f_refine[mask_peak][0]
    if np.any(d_right > d_max / 2):
        f_r_bound = f_right[d_right > d_max / 2][0]
    else:
        f_r_bound = f_refine[mask_peak][-1]
    bound_interval = f_r_bound - f_l_bound

    # decide on the multiple of the period
    harmonics, harmonic_n = frs.find_harmonics_from_pattern(f_n, f_base, f_tol=freq_res / 2)
    completeness_p = (len(harmonics) / (f_nyquist // f_base))
    completeness_p_l = (len(harmonics[harmonic_n <= 15]) / (f_nyquist // f_base))

    # check these (commonly missed) fractions
    n_multiply = np.array([1/2, 2, 3, 4, 5])
    f_fracs = f_base / n_multiply
    n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(f_fracs, f_n, freq_res, f_nyquist)
    h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # if there are very high numbers, add double that fraction for testing
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    if np.any(test_frac[2:] > 3):
        n_multiply = np.append(n_multiply, [2 * n_multiply[2:][test_frac[2:] > 3]])
        f_fracs = f_base / n_multiply
        n_harm_r_m, completeness_r_m, distance_r_m = frs.harmonic_series_length(f_fracs, f_n, freq_res, f_nyquist)
        h_measure_m = n_harm_r_m * completeness_r_m  # compute h_measure for constraining a domain

    # compute diagnostic fractions that need to meet some threshold
    test_frac = h_measure_m / h_measure[mask_peak][i_min_dist]
    compl_frac = completeness_r_m / completeness_p

    # doubling the period may be done if the harmonic filling factor below f_16 is very high
    f_cut = np.max(f_n[harmonics][harmonic_n <= 15])
    f_n_c = f_n[f_n <= f_cut]
    n_harm_r_2, completeness_r_2, distance_r_2 = frs.harmonic_series_length(f_fracs, f_n_c, freq_res, f_nyquist)
    compl_frac_2 = completeness_r_2[1] / completeness_p_l

    # empirically determined thresholds for the various measures
    minimal_frac = 1.1
    minimal_compl_frac = 0.85
    minimal_frac_low = 0.95
    minimal_compl_frac_low = 0.95

    # test conditions
    test_condition = (test_frac > minimal_frac)
    compl_condition = (compl_frac > minimal_compl_frac)
    test_condition_2 = (test_frac[1] > minimal_frac_low)
    compl_condition_2 = (compl_frac_2 > minimal_compl_frac_low)
    if np.any(test_condition & compl_condition) | (test_condition_2 & compl_condition_2):
        if np.any(test_condition & compl_condition):
            i_best = np.argmax(test_frac[compl_condition])
            f_base = f_fracs[compl_condition][i_best]
        else:
            f_base = f_base / 2

        # make new bounds for refining
        f_left_b = f_base - (bound_interval / 2)
        f_right_b = f_base + (bound_interval / 2)

        # refine by using a dense sampling and the harmonic distances
        f_refine_2 = np.arange(f_left_b, f_right_b, 0.00001 * f_base)
        n_harm_r2, completeness_r2, distance_r2 = frs.harmonic_series_length(f_refine_2, f_n, freq_res, f_nyquist)
        h_measure_2 = n_harm_r2 * completeness_r2  # compute h_measure for constraining a domain
        mask_peak = (h_measure_2 > np.max(h_measure_2) / 1.5)  # constrain the domain of the search
        i_min_dist = np.argmin(distance_r2[mask_peak])
        f_base = f_refine_2[mask_peak][i_min_dist]

    return f_base
