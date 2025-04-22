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

from star_shine.core import timeseries as tsf
from star_shine.core import fitting as fit
from star_shine.core import analysis as anf
from star_shine.core import mcmc as mcf
from star_shine.core import utility as ut
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
        self.logger = logger or get_custom_logger(full_dir, self.data.target_id, config.verbose)

        # check the input data
        if not isinstance(data, Data):
            self.logger.warning("Input `data` should be a Data object.")
        elif len(data.time) == 0:
            self.logger.warning("Data object does not contain time series data.")

        return

    def model_linear(self):
        """Returns a piece-wise linear curve for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            The model time series of a (set of) straight line(s)
        """
        curve = self.result.model_linear(self.data.time, self.data.i_chunks)

        return curve

    def model_sinusoid(self):
        """Returns a sum of sine waves for the time series with the current parameters.

        Returns
        -------
        numpy.ndarray[Any, dtype[float]]
            Model time series of a sum of sine waves. Varies around 0.
        """
        curve = self.result.model_sinusoid(self.data.time)

        return curve

    def iterative_prewhitening(self):
        """Iterative prewhitening of the input flux time series in the form of sine waves and a piece-wise linear curve.

        After extraction, a final check is done to see whether some frequencies are better removed or groups of
        frequencies are better replaced by one frequency.

        Continues from last results if frequency list is not empty.

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        n_f_init = len(self.result.f_n)
        if config.verbose:
            self.logger.info(f"{n_f_init} frequencies. Looking for more...")

        # start by looking for more harmonics
        if self.result.p_orb != 0:
            out_a = tsf.extract_harmonics(self.data.time, self.data.flux, self.result.p_orb, self.data.i_chunks,
                                          config.bic_thr, self.result.f_n, self.result.a_n, self.result.ph_n,
                                          verbose=config.verbose)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        # extract all frequencies with the iterative scheme
        out_b = tsf.extract_sinusoids(self.data.time, self.data.flux, self.data.i_chunks, self.result.p_orb,
                                      self.result.f_n, self.result.a_n, self.result.ph_n, bic_thr=config.bic_thr,
                                      snr_thr=config.snr_thr, stop_crit=config.stop_criterion, select=config.select_next,
                                      f0=0, fn=self.data.f_nyquist, fit_each_step=config.optimise_step,
                                      verbose=config.verbose)
        self.result.setter(const=out_b[0], slope=out_b[1], f_n=out_b[2], a_n=out_b[3], ph_n=out_b[4])

        # remove any frequencies that end up not making the statistical cut
        out_c = tsf.reduce_sinusoids(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(const=out_c[0], slope=out_c[1], f_n=out_c[2], a_n=out_c[3], ph_n=out_c[4])

        # select frequencies based on some significance criteria
        out_d = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope,
                                     self.result.f_n, self.result.a_n, self.result.ph_n, self.data.i_chunks,
                                     verbose=config.verbose)
        self.result.setter(passed_sigma=out_d[0], passed_snr=out_d[1], passed_both=out_d[2], passed_harmonic=out_d[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3], ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Iterative prewhitening results.')

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            self.logger.info("Extraction of sinusoids complete.")
            self.logger.extra(f"{len(self.result.f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. "
                              f"Time taken: {t_b - t_a:1.1f}")

        return self.result

    def optimise_sinusoid(self):
        """Optimise the parameters of the sinusoid and linear model

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        if config.verbose:
            self.logger.info("Starting multi-sinusoid NL-LS optimisation.")

        # use the chosen optimisation method
        if config.optimise_method == 'fitter':
            par_mean = fit.fit_multi_sinusoid_per_group(self.data.time, self.data.flux, self.result.const,
                                                        self.result.slope, self.result.f_n, self.result.a_n,
                                                        self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        else:
            # make model including everything to calculate noise level
            resid = self.data.flux - self.model_linear() - self.model_sinusoid()
            n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
            noise_level = ut.std_unb(resid, len(self.data.time) - n_param)

            # formal linear and sinusoid parameter errors
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(self.data.time, resid,
                                                                                 self.data.flux_err, self.result.a_n,
                                                                                 self.data.i_chunks)

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            output = mcf.sample_sinusoid(self.data.time, self.data.flux, self.result.const, self.result.slope,
                                         f_n, a_n, ph_n,  self.result.c_err, self.result.sl_err, f_n_err, a_n_err,
                                         ph_n_err, noise_level, self.data.i_chunks, verbose=config.verbose)
            inf_data, par_mean, par_hdi = output
            self.result.setter(c_hdi=par_hdi[0], sl_hdi=par_hdi[1], f_n_hdi=par_hdi[2], a_n_hdi=par_hdi[3],
                               ph_n_hdi=par_hdi[4])

        self.result.setter(const=par_mean[0], slope=par_mean[1], f_n=par_mean[2], a_n=par_mean[3], ph_n=par_mean[4])

        # select frequencies based on some significance criteria
        out_b = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, 0, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_b[0], passed_snr=out_b[1], passed_both=out_b[2], passed_harmonic=out_b[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3], ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Multi-sinusoid NL-LS optimisation results.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            self.logger.info("Optimisation of sinusoids complete.")
            self.logger.extra(f"{len(self.result.f_n)} frequencies, {self.result.n_param} free parameters, "
                              f"BIC: {self.result.bic:1.2f}. Time taken: {t_b - t_a:1.1f}s")

        return self.result

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
        if config.verbose:
            self.logger.info("Coupling the harmonic frequencies to the orbital frequency...")

        # if given, the input p_orb is refined locally, otherwise the period is searched for globally
        if self.data.p_orb == 0:
            self.result.p_orb = tsf.find_orbital_period(self.data.time, self.data.flux, self.result.f_n)
        else:
            self.result.p_orb = tsf.refine_orbital_period(self.data.p_orb, self.data.time, self.result.f_n)

        # if time series too short, or no harmonics found, log and warn and maybe cut off the analysis
        freq_res = 1.5 / self.data.t_tot  # Rayleigh criterion
        harmonics, harmonic_n = anf.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=freq_res / 2)
        if (self.data.t_tot / self.result.p_orb > 1.1) & (len(harmonics) > 1):
            # couple the harmonics to the period. likely removes more frequencies that need re-extracting
            out_a = tsf.fix_harmonic_frequency(self.data.time, self.data.flux, self.result.p_orb,  self.result.const,
                                               self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                               self.data.i_chunks, verbose=config.verbose)
            self.result.setter(const=out_a[0], slope=out_a[1], f_n=out_a[2], a_n=out_a[3], ph_n=out_a[4])

        # remove any frequencies that end up not making the statistical cut
        out_b = tsf.reduce_sinusoids(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                     self.result.slope, self.result.f_n, self.result.a_n, self.result.ph_n,
                                     self.data.i_chunks, verbose=config.verbose)
        self.result.setter(const=out_b[0], slope=out_b[1], f_n=out_b[2], a_n=out_b[3], ph_n=out_b[4])

        # select frequencies based on some significance criteria
        out_c = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope, self.result.f_n, self.result.a_n,
                                     self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_c[0], passed_snr=out_c[1], passed_both=out_c[2], passed_harmonic=out_c[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        harmonics, harmonic_n = anf.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
        n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_d = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        self.result.setter(c_err=out_d[0], sl_err=out_d[1], f_n_err=out_d[2], a_n_err=out_d[3], ph_n_err=out_d[4])
        p_err, _, _ = tsf.linear_regression_uncertainty_ephem(self.data.time, self.result.p_orb,
                                                              sigma_t=self.data.t_step / 2)
        self.result.setter(p_orb=np.array([self.result.p_orb, p_err, 0, 0]))

        # set the result description
        self.result.setter(description='Harmonic frequencies coupled to the orbital period.')

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            rnd_p_orb = max(ut.decimal_figures(p_err, 2), ut.decimal_figures(self.result.p_orb, 2))
            self.logger.info("Orbital harmonic frequencies coupled.")
            self.logger.extra(f"p_orb: {self.result.p_orb:.{rnd_p_orb}f} (+-{p_err:.{rnd_p_orb}f}), "
                              f"{len(self.result.f_n)} frequencies, {n_param} free parameters, BIC: {bic:1.2f}. "
                              f"Time taken: {t_b - t_a:1.1f}s")

        # log if short time span or few harmonics
        if self.data.t_tot / self.result.p_orb < 1.1:
            self.logger.warning(f"Period over time-base is less than two: {self.data.t_tot / self.result.p_orb}; "
                                f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")
        elif len(harmonics) < 2:
            self.logger.warning(f"Not enough harmonics found: {len(harmonics)}; "
                                f"period (days): {self.result.p_orb}; time-base (days): {self.data.t_tot}")

        return self.result

    def optimise_sinusoid_h(self):
        """Optimise the parameters of the sinusoid and linear model with coupled harmonics

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results
        """
        t_a = systime.time()
        if config.verbose:
            self.logger.info("Starting multi-sine NL-LS optimisation with harmonics.")

        # use the chosen optimisation method
        if config.optimise_method == 'fitter':
            par_mean = fit.fit_multi_sinusoid_harmonics_per_group(self.data.time, self.data.flux, self.result.p_orb,
                                                                  self.result.const, self.result.slope,
                                                                  self.result.f_n, self.result.a_n, self.result.ph_n,
                                                                  self.data.i_chunks, verbose=config.verbose)
        else:
            # make model including everything to calculate noise level
            resid = self.data.flux - self.model_linear() - self.model_sinusoid()
            harmonics, harmonic_n = anf.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
            n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
            noise_level = ut.std_unb(resid, len(self.data.time) - n_param)

            # formal linear and sinusoid parameter errors
            c_err, sl_err, f_n_err, a_n_err, ph_n_err = tsf.formal_uncertainties(self.data.time, resid,
                                                                                 self.data.flux_err, self.result.a_n,
                                                                                 self.data.i_chunks)
            p_err, _, _ = tsf.linear_regression_uncertainty_ephem(self.data.time, self.result.p_orb,
                                                                  sigma_t=self.data.t_step / 2)

            # do not include those frequencies that have too big uncertainty
            include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]

            # Monte Carlo sampling of the model
            output = mcf.sample_sinusoid_h(self.data.time, self.data.flux, self.result.p_orb, self.result.const,
                                           self.result.slope, f_n, a_n, ph_n, self.result.p_err, self.result.c_err,
                                           self.result.sl_err, f_n_err, a_n_err, ph_n_err, noise_level,
                                           self.data.i_chunks, verbose=config.verbose)
            inf_data, par_mean, par_hdi = output
            self.result.setter(p_hdi=par_hdi[0], c_hdi=par_hdi[1], sl_hdi=par_hdi[2], f_n_hdi=par_hdi[3],
                               a_n_hdi=par_hdi[4], ph_n_hdi=par_hdi[5])

        self.result.setter(p_orb=par_mean[0], const=par_mean[1], slope=par_mean[2], f_n=par_mean[3], a_n=par_mean[4],
                           ph_n=par_mean[5])

        # select frequencies based on some significance criteria
        out_b = tsf.select_sinusoids(self.data.time, self.data.flux, self.data.flux_err, self.result.p_orb,
                                     self.result.const, self.result.slope, self.result.f_n, self.result.a_n,
                                     self.result.ph_n, self.data.i_chunks, verbose=config.verbose)
        self.result.setter(passed_sigma=out_b[0], passed_snr=out_b[1], passed_both=out_b[2], passed_harmonic=out_b[3])

        # main function done, calculate the rest of the stats
        resid = self.data.flux - self.model_linear() - self.model_sinusoid()
        harmonics, harmonic_n = anf.find_harmonics_from_pattern(self.result.f_n, self.result.p_orb, f_tol=1e-9)
        n_param = 2 * len(self.result.const) + 1 + 2 * len(harmonics) + 3 * (len(self.result.f_n) - len(harmonics))
        bic = tsf.calc_bic(resid, n_param)
        noise_level = ut.std_unb(resid, len(self.data.time) - n_param)
        self.result.setter(n_param=n_param, bic=bic, noise_level=noise_level)

        # calculate formal uncertainties
        out_e = tsf.formal_uncertainties(self.data.time, resid, self.data.flux_err, self.result.a_n, self.data.i_chunks)
        p_err, _, _ = tsf.linear_regression_uncertainty_ephem(self.data.time, self.result.p_orb,
                                                              sigma_t=self.data.t_step / 2)
        self.result.setter(p_err=p_err, c_err=out_e[0], sl_err=out_e[1], f_n_err=out_e[2], a_n_err=out_e[3],
                           ph_n_err=out_e[4])

        # set the result description
        self.result.setter(description='Multi-sine NL-LS optimisation results with coupled harmonics.')
        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        if config.verbose:
            rnd_p_orb = max(ut.decimal_figures(self.result.p_err, 2),
                            ut.decimal_figures(self.result.p_orb, 2))
            self.logger.info("Optimisation with coupled harmonics complete.")
            self.logger.extra(f"p_orb: {self.result.p_orb:.{rnd_p_orb}f} (+-{self.result.p_err:.{rnd_p_orb}f}), "
                              f"{len(self.result.f_n)} frequencies, {self.result.n_param} free parameters, "
                              f"BIC: {self.result.bic:1.2f}. Time taken: {t_b - t_a:1.1f}s")

        return self.result

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

        # run steps until config number
        if config.stop_at_stage != 0:
            step_names = step_names[:config.stop_at_stage]

        # tag the start of the analysis
        t_a = systime.time()
        self.logger.info("Start of analysis")

        # run this sequence for each analysis step of the pipeline
        for step in range(len(step_names)):
            file_name = os.path.join(self.save_dir, self.save_subdir, f"{self.data.target_id}_result_{step + 1}.hdf5")

            # Load existing file if not overwriting
            self.result = Result.load_conditional(file_name)  # returns empty Result if no file

            # if existing results were loaded, go to the next step
            if self.result.target_id != '':
                continue

            # do the analysis step
            analysis_step = getattr(self, step_names[step])
            analysis_step()

            # save the results if conditions are met
            self.result.save_conditional(file_name)

        # final message and timing
        t_b = systime.time()
        self.logger.info(f"End of analysis. Total time elapsed: {t_b - t_a:1.1f}s.")  # info to save to log

        return self.result
