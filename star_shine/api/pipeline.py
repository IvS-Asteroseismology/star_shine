"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the pipeline class that defines the analysis pipeline.

Code written by: Luc IJspeert
"""
import os
import time as systime
import numpy as np

from star_shine.api.data import Data
from star_shine.api.result import Result

from star_shine.core import analysis as ana, time_series as tms
from star_shine.core import frequency_sets as frs, periodogram as pdg, fitting as fit
from star_shine.core import goodness_of_fit as gof, mcmc as mcf, utility as ut
from star_shine.config.helpers import get_config, get_custom_logger


# load configuration
config = get_config()


class Pipeline:
    """A class to analyze light curve data.

    Handles the full analysis pipeline of Star Shine.

    Attributes
    ----------
    data: Data object
        Instance of the Data class holding the light curve data.
    result: Result object
        Instance of the Result class holding the parameters of the result.
    save_dir: str
        Root directory where the result files will be stored.
    save_subdir: str
        Sub-directory that is made to contain the save files.
    logger: Logger object
        Instance of the logging library.
    """

    def __init__(self, data, save_dir='', logger=None):
        """Initialises the Pipeline object.

        Parameters
        ----------
        data: Data object
            Instance of the Data class with the data to be analysed.
        save_dir: str, optional
            Root directory where result files will be stored. Added to the file name.
            If empty, it is loaded from config.

        Notes
        -----
        Creates a directory where all the analysis result files will be stored.
        """
        # set the data and result objects
        self.data = data  # the data to be analysed
        self.result = Result()  # an empty result instance

        # the files will be stored here
        if save_dir == '':
            save_dir = config.save_dir
        self.save_dir = save_dir
        self.save_subdir = f"{self.data.target_id}_analysis"

        # for saving, make a folder if not there yet
        full_dir = os.path.join(self.save_dir, self.save_subdir)
        if not os.path.isdir(full_dir):
            os.mkdir(full_dir)  # create the subdir

        # initialise custom logger
        self.logger = logger or get_custom_logger(self.data.target_id, full_dir, config.verbose)

        # check the input data
        if not isinstance(data, Data):
            self.logger.warning("Input `data` should be a Data object.")
        elif len(data.time_series.time) == 0:
            self.logger.warning("Data object does not contain time series data.")

        return

    def model_linear(self):
        """A piece-wise linear curve for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The model time series of a (set of) straight line(s)
        """
        curve = self.result.model_linear(self.data.time_series.time, self.data.time_series.i_chunks)

        return curve

    def model_sinusoid(self):
        """A sum of sine waves for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model time series of a sum of sine waves. Varies around 0.
        """
        curve = self.result.model_sinusoid(self.data.time_series.time)

        return curve

    def model(self):
        """The full model of the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model of the time series.
        """
        full_model = self.model_linear() + self.model_sinusoid()

        return full_model

    def residual(self):
        """The residuals of the full model of the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Residual of the time series.
        """
        resid = self.data.time_series.flux - self.model()

        return resid

    def periodogram(self, subtract_model=True):
        """Compute the Lomb-Scargle periodogram of the time series

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
            resid = self.data.time_series.flux - self.model()
            f0, fn, df = self.data.time_series.pd_f0, self.data.time_series.pd_fn, self.data.time_series.pd_df
            f, a = pdg.scargle_parallel(self.data.time_series.time, resid, f0=f0, fn=fn, df=df, norm='amplitude')
        else:
            f, a = self.data.periodogram()

        return f, a

    def reduce_sinusoids(self):
        """Remove any frequencies that end up not making the statistical cut"""
        # remove any frequencies that end up not making the statistical cut
        out = ana.reduce_sinusoids(self.data.time_series.time, self.data.time_series.flux, self.result.p_orb,
                                   self.result.const,
                                   self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                   self.data.time_series.i_chunks, logger=self.logger)

        self.result.setter(const=out[0], slope=out[1], f_n=out[2], a_n=out[3], ph_n=out[4])

        return None

    def select_sinusoids(self):
        """Select frequencies based on some significance criteria."""
        out = ana.select_sinusoids(self.data.time_series.time, self.data.time_series.flux,
                                   self.data.time_series.flux_err, self.result.p_orb,
                                   self.result.const, self.result.slope, self.result.f_n, self.result.a_n,
                                   self.result.ph_n, self.data.time_series.i_chunks, logger=self.logger)

        self.result.setter(passed_sigma=out[0], passed_snr=out[1], passed_both=out[2], passed_harmonic=out[3])

        return None

    def update_stats(self):
        """Updates the model statistics and formal uncertainties."""
        # calculate the number of parameters
        self.result.update_n_param()

        # model residual
        resid = self.residual()

        # set number of parameters, BIC and noise level
        bic = gof.calc_bic(resid, self.result.n_param)
        noise_level = ut.std_unb(resid, len(self.data.time_series.time) - self.result.n_param)
        self.result.setter(bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out = ut.formal_uncertainties(self.data.time_series.time, resid, self.data.time_series.flux_err,
                                      self.result.a_n, self.data.time_series.i_chunks)
        self.result.setter(c_err=out[0], sl_err=out[1], f_n_err=out[2], a_n_err=out[3], ph_n_err=out[4])

        # period uncertainty
        if self.result.p_orb > 0:
            p_err, _, _ = ut.linear_regression_uncertainty_ephem(self.data.time_series.time, self.result.p_orb,
                                                                 sigma_t=self.data.time_series.t_step / 2)
            self.result.setter(p_err=p_err)

        return None

    def extract_approx(self, f_approx):
        """Extract a sinusoid from the time series at an approximate frequency.

        Parameters
        ----------
        f_approx: float
            Approximate location of the frequency of maximum amplitude.
        """
        f, a, ph = ana.extract_approx(self.data.time_series.time, self.residual(), f_approx)

        # if identical frequency exists, assume this was unintentional
        if len(self.result.f_n) > 0 and np.min(np.abs(f - self.result.f_n)) < self.data.time_series.pd_df:
            self.logger.warning("Existing identical frequency found.")
            return None

        # append the sinusoid to the result
        f_n, a_n, ph_n = self.result.f_n, self.result.a_n, self.result.ph_n
        f_n, a_n, ph_n = np.append(f_n, f), np.append(a_n, a), np.append(ph_n, ph)
        self.result.setter(f_n=f_n, a_n=a_n, ph_n=ph_n)
        self.logger.info(f"Appended f: {f:1.2f}")

        self.reduce_sinusoids()  # remove any frequencies that end up not making the statistical cut
        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Manual extraction.')

        # print some useful info
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        return None

    def remove_approx(self, f_approx):
        """Remove a sinusoid from the list at an approximate frequency.

        Parameters
        ----------
        f_approx: float
            Approximate location of the frequency to be removed.
        """
        # guard against empty frequency array
        if len(self.result.f_n) == 0:
            return None

        index = np.argmin(np.abs(f_approx - self.result.f_n))
        f_to_remove = self.result.f_n[index]

        # if too far away, assume this was unintentional
        if abs(f_to_remove - f_approx) > 3 * self.data.f_resolution:
            self.logger.warning("No close frequency to remove.")
            return None

        # remove the sinusoid
        self.result.remove_sinusoids(np.array([index]))
        self.logger.info(f"Removed f: {f_to_remove:1.2f}")

        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Manual extraction.')

        # print some useful info
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        return None

    def iterative_prewhitening(self, n_extract=0):
        """Iterative prewhitening of the input flux time series in the form of sine waves and a piece-wise linear curve.

        After extraction, a final check is done to see whether some frequencies are better removed or groups of
        frequencies are better replaced by one frequency.

        Continues from last results if frequency list is not empty.

        Parameters
        ----------
        n_extract: int, optional
            Maximum number of frequencies to extract. The stop criterion is still leading. Zero means as many as possible.

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        n_f_init = len(self.result.f_n)
        self.logger.info(f"{n_f_init} frequencies. Looking for more...")

        # start by looking for more harmonics
        if self.result.p_orb != 0:
            out_a = ana.extract_harmonics(self.data.time_series.time, self.data.time_series.flux, self.result.p_orb,
                                          self.data.time_series.i_chunks,
                                          config.bic_thr, self.result.f_n, self.result.a_n, self.result.ph_n,
                                          logger=self.logger)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        # extract all frequencies with the iterative scheme
        ts_model = tms.TimeSeriesModel(self.data.time_series.time, self.data.time_series.flux,
                                       self.data.time_series.flux_err, self.data.time_series.i_chunks)
        ts_model.set_sinusoids(self.result.f_n, self.result.a_n, self.result.ph_n)
        ts_model.update_linear_model()
        ts_model = ana.extract_sinusoids(ts_model, bic_thr=config.bic_thr, snr_thr=config.snr_thr,
                                         stop_crit=config.stop_criterion, select=config.select_next,
                                         n_extract=n_extract, fit_each_step=config.optimise_step,
                                         replace_each_step=config.replace_step,
                                         logger=self.logger)
        out_b = ts_model.get_parameters()
        self.result.setter(const=out_b[0], slope=out_b[1], f_n=out_b[2], a_n=out_b[3], ph_n=out_b[4])

        self.reduce_sinusoids()  # remove any frequencies that end up not making the statistical cut
        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Iterative prewhitening results.')

        # print some useful info
        t_b = systime.time()
        self.logger.info(f"Extraction of sinusoids complete. Time taken: {t_b - t_a:1.1f}.")
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        return None

    def optimise_sinusoid(self):
        """Optimise the parameters of the sinusoid and linear model

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        self.logger.info("Starting multi-sinusoid NL-LS optimisation.")

        # use the chosen optimisation method
        if config.optimise_method == 'fitter':
            par_mean = fit.fit_multi_sinusoid_per_group(self.data.time_series.time, self.data.time_series.flux,
                                                        self.result.const,
                                                        self.result.slope, self.result.f_n, self.result.a_n,
                                                        self.result.ph_n, self.data.time_series.i_chunks,
                                                        logger=self.logger)
        else:
            # make model including everything to calculate noise level
            resid = self.data.time_series.flux - self.model_linear() - self.model_sinusoid()
            n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
            noise_level = ut.std_unb(resid, len(self.data.time_series.time) - n_param)

            # formal linear and sinusoid parameter errors
            out_a = ut.formal_uncertainties(self.data.time_series.time, resid, self.data.time_series.flux_err,
                                            self.result.a_n, self.data.time_series.i_chunks)
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = out_a

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            out_b = mcf.sample_sinusoid(self.data.time_series.time, self.data.time_series.flux, self.result.const,
                                        self.result.slope, f_n, a_n, ph_n, self.result.c_err, self.result.sl_err,
                                        f_n_err, a_n_err, ph_n_err, noise_level, self.data.time_series.i_chunks,
                                        logger=self.logger)
            inf_data, par_mean, par_hdi = out_b
            self.result.setter(c_hdi=par_hdi[0], sl_hdi=par_hdi[1], f_n_hdi=par_hdi[2], a_n_hdi=par_hdi[3],
                               ph_n_hdi=par_hdi[4])

        self.result.setter(const=par_mean[0], slope=par_mean[1], f_n=par_mean[2], a_n=par_mean[3], ph_n=par_mean[4])

        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Multi-sinusoid NL-LS optimisation results.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        self.logger.info(f"Optimisation of sinusoids complete. Time taken: {t_b - t_a:1.1f}s.")
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        return None

    def couple_harmonics(self):
        """Find the orbital period and couple harmonic frequencies to the orbital period

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results

        Notes
        -----
        Performs a global period search, if the period is unknown.
        If a period is given, it is locally refined for better performance.

        Removes theoretical harmonic candidate frequencies within the frequency
        resolution, then extracts a single harmonic at the theoretical location.

        Removes any frequencies that end up not making the statistical cut.
        """
        t_a = systime.time()
        self.logger.info("Coupling the harmonic frequencies to the orbital frequency...")

        # if given, the input p_orb is refined locally, otherwise the period is searched for globally
        if self.data.p_orb == 0:
            p_orb = ana.find_orbital_period(self.data.time_series.time, self.data.time_series.flux, self.result.f_n)
        else:
            p_orb = ana.refine_orbital_period(self.data.p_orb, self.data.time_series.time, self.result.f_n)
        self.result.setter(p_orb=p_orb)

        # if time series too short, or no harmonics found, log and warn and maybe cut off the analysis
        freq_res = 1.5 / self.data.t_tot  # Rayleigh criterion
        harmonics, harmonic_n = frs.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=freq_res / 2)

        if (self.data.t_tot / self.result.p_orb > 1.1) & (len(harmonics) > 1):
            # couple the harmonics to the period. likely removes more frequencies that need re-extracting
            out_a = ana.fix_harmonic_frequency(self.data.time_series.time, self.data.time_series.flux,
                                               self.result.p_orb, self.result.const,
                                               self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                               self.data.i_chunks, logger=self.logger)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        self.reduce_sinusoids()  # remove any frequencies that end up not making the statistical cut
        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Harmonic frequencies coupled to the orbital period.')

        # print some useful info
        t_b = systime.time()
        p_orb_formatted = ut.float_to_str_scientific(self.result.p_orb, self.result.p_err, error=True, brackets=True)
        self.logger.info(f"Orbital harmonic frequencies coupled. P_orb: {p_orb_formatted}. "
                         f"Time taken: {t_b - t_a:1.1f}s.")
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        # log if short time span or few harmonics
        if self.data.t_tot / self.result.p_orb < 1.1:
            self.logger.warning(f"Period over time-base is less than two: {self.data.t_tot / self.result.p_orb}; "
                                f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")
        elif len(harmonics) < 2:
            self.logger.warning(f"Not enough harmonics found: {len(harmonics)}; "
                                f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")

        return None

    def optimise_sinusoid_h(self):
        """Optimise the parameters of the sinusoid and linear model with coupled harmonics

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        self.logger.info("Starting multi-sine NL-LS optimisation with harmonics.")

        # use the chosen optimisation method
        if config.optimise_method == 'fitter':
            par_mean = fit.fit_multi_sinusoid_harmonics_per_group(self.data.time_series.time, self.data.time_series.flux, self.result.p_orb,
                                                                  self.result.const, self.result.slope,
                                                                  self.result.f_n, self.result.a_n, self.result.ph_n,
                                                                  self.data.i_chunks, logger=self.logger)
        else:
            # make model including everything to calculate noise level
            resid = self.data.time_series.flux - self.model_linear() - self.model_sinusoid()
            harmonics, harmonic_n = frs.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
            n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
            noise_level = ut.std_unb(resid, len(self.data.time_series.time) - n_param)

            # formal linear and sinusoid parameter errors
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = ut.formal_uncertainties(self.data.time.time_series, resid,
                                                                                self.data.time_series.flux_err, self.result.a_n,
                                                                                self.data.i_chunks)
            p_err, _, _ = ut.linear_regression_uncertainty_ephem(self.data.time.time_series, self.result.p_orb,
                                                                                      sigma_t=self.data.t_step / 2)

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            output = mcf.sample_sinusoid_h(self.data.time_series.time, self.data.time_series.flux, self.result.p_orb, self.result.const,
                                           self.result.slope, f_n, a_n, ph_n, self.result.p_err, self.result.c_err,
                                           self.result.sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                                           self.data.i_chunks, logger=self.logger)
            inf_data, par_mean, par_hdi = output
            self.result.setter(p_hdi=par_hdi[0], c_hdi=par_hdi[1], sl_hdi=par_hdi[2], f_n_hdi=par_hdi[3],
                               a_n_hdi=par_hdi[4], ph_n_hdi=par_hdi[5])

        self.result.setter(p_orb=par_mean[0], const=par_mean[1], slope=par_mean[2], f_n=par_mean[3], a_n=par_mean[4],
                           ph_n=par_mean[5])

        self.select_sinusoids()  # select frequencies based on some significance criteria
        self.update_stats()  # update the stats

        # set the result identifiers and description
        self.result.setter(target_id=self.data.target_id, data_id=self.data.data_id)
        self.result.setter(description='Multi-sine NL-LS optimisation results with coupled harmonics.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        p_orb_formatted = ut.float_to_str_scientific(self.result.p_orb, self.result.p_err, error=True, brackets=True)
        self.logger.info(f"Optimisation with coupled harmonics complete. P_orb: {p_orb_formatted}."
                         f"Time taken: {t_b - t_a:1.1f}s.")
        self.logger.extra(f"N_f: {len(self.result.f_n)}, N_p: {self.result.n_param}, BIC: {self.result.bic:1.2f}.")

        return None

    def run(self):
        """Run the analysis pipeline on the given data.

        Runs a predefined sequence of analysis steps. Stops at the stage defined in the configuration file.
        Files are saved automatically at the end of each stage, taking into account the desired behaviour for
        overwriting.

        The followed recipe is:

        1) Extract all frequencies
            We start by extracting the frequency with the highest amplitude one by one,
            directly from the Lomb-Scargle periodogram until the BIC does not significantly
            improve anymore. This step involves a final cleanup of the frequencies.

        2) Multi-sinusoid NL-LS optimisation
            The sinusoid parameters are optimised with a non-linear least-squares method,
            using groups of frequencies to limit the number of free parameters.

        3) Measure the orbital period and couple the harmonic frequencies
            Global search done with combined phase dispersion, Lomb-Scargle and length/
            filling factor of the harmonic series in the list of frequencies.
            Then sets the frequencies of the harmonics to their new values, coupling them
            to the orbital period. This step involves a final cleanup of the frequencies.
            [Note: it is possible to provide a fixed period if it is already known.
            It will still be optimised]

        4) Attempt to extract additional frequencies
            The decreased number of free parameters (2 vs. 3), the BIC, which punishes for free
            parameters, may allow the extraction of more harmonics.
            It is also attempted to extract more frequencies like in step 1 again, now taking
            into account the presence of harmonics.
            This step involves a final cleanup of the frequencies.

        5) Multi-sinusoid NL-LS optimisation with coupled harmonics
            Optimise all sinusoid parameters with a non-linear least-squares method,
            using groups of frequencies to limit the number of free parameters
            and including the orbital period and the coupled harmonics.

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        # this list determines which analysis steps are taken and in what order
        step_names = ['iterative_prewhitening', 'optimise_sinusoid', 'couple_harmonics', 'iterative_prewhitening',
                      'optimise_sinusoid_h']

        # if we have predefined periods, repeat the harmonic steps
        harmonic_step_names = ['couple_harmonics', 'iterative_prewhitening', 'optimise_sinusoid_h']

        # run steps until config number
        if config.stop_at_stage != 0:
            step_names = step_names[:config.stop_at_stage]

        # tag the start of the analysis
        t_a = systime.time()
        self.logger.info("Start of analysis")

        # run this sequence for each analysis step of the pipeline
        for step in range(len(step_names)):
            file_name = os.path.join(self.save_dir, self.save_subdir, f"{self.data.target_id}_result_{step + 1}.hdf5")

            # Load existing result from this step if not overwriting (returns empty Result if no file)
            print(step, file_name)
            self.result = Result.load_conditional(file_name, logger=self.logger)

            # if existing result was loaded, go to the next step
            if self.result.target_id != '':
                continue

            # Load result from previous step (returns empty Result if no file)
            self.result = Result.load(file_name.replace(f'result_{step + 1}', f'result_{step}'),
                                      logger=self.logger)

            # do the analysis step
            analysis_step = getattr(self, step_names[step])
            analysis_step()

            # save the results if conditions are met
            print(step, file_name)
            self.result.save_conditional(file_name)

        # final message and timing
        t_b = systime.time()
        self.logger.info(f"End of analysis. Total time elapsed: {t_b - t_a:1.1f}s.")  # info to save to log

        return None
