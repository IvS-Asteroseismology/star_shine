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
    data: Data
        Instance of the Data class holding the light curve data.
    result: Result
        Instance of the Result class holding the parameters of the result.
    ts_model: TimeSeriesModel
        Instance of the TimeSeriesModel class holding the time series model.
    save_dir: str
        Root directory where the result files will be stored.
    save_subdir: str
        Sub-directory that is made to contain the save files.
    logger: Logger
        Instance of the logging library.
    """

    def __init__(self, data, save_dir='', logger=None):
        """Initialises the Pipeline object.

        Parameters
        ----------
        data: Data
            Instance of the Data class with the data to be analysed.
        save_dir: str, optional
            Root directory where result files will be stored. Added to the file name.
            If empty, it is loaded from config.

        Notes
        -----
        Creates a directory where all the analysis result files will be stored.
        """
        # check the input data
        if not isinstance(data, Data):
            raise ValueError("Input `data` should be a Data object.")

        # set the data and result objects
        self.data = data  # the data to be analysed
        self.result = Result()  # an empty result instance

        # set the time series object with the time series from data
        self.ts_model = tms.TimeSeriesModel.from_time_series(self.data.time_series)

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

        return

    def extract_approx(self, f_approx):
        """Extract a sinusoid from the time series at an approximate frequency.

        Parameters
        ----------
        f_approx: float
            Approximate location of the frequency of maximum amplitude.
        """
        f, a, ph = ana.extract_approx(self.ts_model.time, self.ts_model.residual(), f_approx)

        # if identical frequency exists, assume this was unintentional
        if self.ts_model.sinusoid.n_sin > 0 and np.min(np.abs(f - self.ts_model.sinusoid.f_n)) < self.ts_model.pd_df:
            self.logger.warning("Existing identical frequency found.")
            return None

        # add the sinusoid
        self.ts_model.add_sinusoids(f, a, ph)
        self.ts_model.update_linear_model()

        # remove any frequencies that end up not making the statistical cut
        ana.reduce_sinusoids(self.ts_model, logger=self.logger)

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # print some useful info
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Extracted f= {f:1.2f}", extra={'update': True})

        return None

    def remove_approx(self, f_approx):
        """Remove a sinusoid from the list at an approximate frequency.

        Parameters
        ----------
        f_approx: float
            Approximate location of the frequency to be removed.
        """
        # guard against empty frequency array
        if len(self.ts_model.sinusoid.f_n) == 0:
            return None

        index = np.argmin(np.abs(f_approx - self.ts_model.sinusoid.f_n))
        f_to_remove = self.ts_model.sinusoid.f_n[index]

        # if too far away, assume this was unintentional
        if abs(f_to_remove - f_approx) > 3 * self.ts_model.f_resolution:
            self.logger.warning("No close frequency to remove.")
            return None

        # remove the sinusoid
        self.ts_model.remove_sinusoids(index)

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # print some useful info
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Removed f= {f_to_remove:1.2f}", extra={'update': True})

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
        self.logger.info(f"Iterative prewhitening starting.")

        # start by looking for more harmonics - only do this when n_extract set to 0
        if self.ts_model.sinusoid.n_base != 0 and n_extract == 0:
            ana.extract_harmonics(self.ts_model, config.bic_thr, logger=self.logger)

        # extract all frequencies with the iterative scheme
        ana.extract_sinusoids(self.ts_model, bic_thr=config.bic_thr, snr_thr=config.snr_thr,
                              stop_crit=config.stop_criterion, select=config.select_next,  n_extract=n_extract,
                              fit_each_step=config.optimise_step, replace_each_step=config.replace_step,
                              logger=self.logger)

        # remove any frequencies that end up not making the statistical cut
        ana.reduce_sinusoids(self.ts_model, logger=self.logger)

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # print some useful info
        t_b = systime.time()
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Iterative prewhitening finished. "
                         f"Time taken= {t_b - t_a:1.1f}s.")

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

        # optimisation
        fit.fit_multi_sinusoid_grouped(self.ts_model, logger=self.logger)
            # # make model including everything to calculate noise level
            # resid = self.data.time_series.flux - self.ts_model.model_linear() - self.ts_model.model_sinusoid()
            # n_param = 2 * len(self.result.const) + 3 * len(self.result.f_n)
            # noise_level = ut.std_unb(resid, len(self.data.time_series.time) - n_param)
            #
            # # formal linear and sinusoid parameter errors
            # out_a = ut.formal_uncertainties(self.data.time_series.time, resid, self.data.time_series.flux_err,
            #                                 self.result.a_n, self.data.time_series.i_chunks)
            # c_err, sl_err, f_n_err, a_n_err, ph_n_err = out_a
            #
            # # do not include those frequencies that have too big uncertainty
            # include = (ph_n_err < 1 / np.sqrt(6))  # circular distribution for ph_n cannot handle these
            # f_n, a_n, ph_n = self.result.f_n[include], self.result.a_n[include], self.result.ph_n[include]
            # f_n_err, a_n_err, ph_n_err = f_n_err[include], a_n_err[include], ph_n_err[include]
            #
            # # Monte Carlo sampling of the model
            # out_b = mcf.sample_sinusoid(self.data.time_series.time, self.data.time_series.flux, self.result.const,
            #                             self.result.slope, f_n, a_n, ph_n, self.result.c_err, self.result.sl_err,
            #                             f_n_err, a_n_err, ph_n_err, noise_level, self.data.time_series.i_chunks,
            #                             logger=self.logger)
            # inf_data, par_mean, par_hdi = out_b
            # self.result.from_dict(c_hdi=par_hdi[0], sl_hdi=par_hdi[1], f_n_hdi=par_hdi[2], a_n_hdi=par_hdi[3],
            #                       ph_n_hdi=par_hdi[4])

        # remove any frequencies that end up not making the statistical cut
        ana.reduce_sinusoids(self.ts_model, logger=self.logger)

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # ut.save_inference_data(file_name, inf_data)  # currently not saved

        # print some useful info
        t_b = systime.time()
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Optimisation complete. Time taken: {t_b - t_a:1.1f}s.")

        return None

    def add_base_harmonic(self, f_base):
        """Add a base harmonic frequency to the model."""
        # get amplitude and phase
        a, ph = pdg.scargle_ampl_phase_single(self.ts_model.time, self.ts_model.residual(), f_base)

        # add the sinusoid
        index = len(self.ts_model.sinusoid.f_n)
        self.ts_model.add_sinusoids(f_base, a, ph, h_base_new=index, h_mult_new=1)
        self.ts_model.update_linear_model()

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # print some useful info
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Added f= {f_base:1.2f}", extra={'update': True})

        return None

    def couple_harmonics(self):
        """Find the orbital period and couple harmonic frequencies to the orbital period

        Returns
        -------
        Result
            Instance of the Result class containing the analysis results

        Notes
        -----
        Performs a global search for the base frequency, if unknown. If a frequency is given, it is locally
        refined for better performance. Needs an existing list of extracted sinusoids.

        Removes theoretical harmonic candidate frequencies within the frequency resolution, then extracts
        a single harmonic at the theoretical location.

        Removes any frequencies that end up not making the statistical cut.
        """
        t_a = systime.time()
        self.logger.info("Coupling the harmonic frequencies to the base frequency.")

        # check for non-coupled f_base
        h_base_unique, h_base_map = self.ts_model.sinusoid.get_h_base_map()
        for i_h in h_base_unique:
            # take the first f_base without a series
            if len(h_base_map[h_base_map == i_h]) > 1:
                f_base = self.ts_model.sinusoid.f_n[i_h]
                f_base = ana.refine_harmonic_base_frequency(f_base, self.ts_model)
                break
        else:
            # if no break encountered (no non-coupled f_base found), an f_base is searched for globally
            f_base = ana.find_harmonic_base_frequency(self.ts_model)

        if (t_over_p := self.ts_model.t_tot * f_base) > 1.1:
            # couple the harmonics to the period. likely removes more frequencies that need re-extracting
            ana.couple_harmonics(self.ts_model, f_base, logger=self.logger)

        # remove any frequencies that end up not making the statistical cut
        ana.reduce_sinusoids(self.ts_model, logger=self.logger)

        # update the TimeSeriesModel passing masks and uncertainties
        ana.select_sinusoids(self.ts_model, logger=self.logger)

        # print some useful info
        t_b = systime.time()
        f_base_err = self.ts_model.sinusoid.f_h_err[self.ts_model.sinusoid.get_f_index(f_base)]
        f_base_formatted = ut.float_to_str_scientific(f_base, f_base_err, error=True, brackets=True)
        self.logger.info(f"N_f= {self.ts_model.sinusoid.n_sin}, BIC= {self.ts_model.bic():1.2f}, "
                         f"N_p= {self.ts_model.n_param} - Harmonic frequencies coupled. f_base= {f_base_formatted}, "
                         f"1/f_base= {1/f_base:1.2f}. Time taken: {t_b - t_a:1.1f}s")

        # log if short time span or few harmonics
        if t_over_p < 1.1:
            self.logger.warning(f"Time-base over period is less than two: {t_over_p}.")
        elif self.ts_model.sinusoid.n_harm < 2:
            self.logger.warning(f"Not enough harmonics found: {self.ts_model.sinusoid.n_harm}.")

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
                      'optimise_sinusoid']

        # if we have predefined periods, repeat the harmonic steps
        harmonic_step_names = ['couple_harmonics', 'iterative_prewhitening', 'optimise_sinusoid']

        # run steps until config number
        if config.stop_at_stage != 0:
            step_names = step_names[:config.stop_at_stage]

        # tag the start of the analysis
        t_a = systime.time()
        self.logger.info("Start of analysis")

        # reset the time series object with the time series from data
        self.ts_model = tms.TimeSeriesModel.from_time_series(self.data.time_series)

        # run this sequence for each analysis step of the pipeline
        for step in range(len(step_names)):
            file_name = os.path.join(self.save_dir, self.save_subdir, f'{self.data.target_id}_result_{step + 1}.hdf5')

            # skip to next step if result exists and we were not planning to overwrite it
            if os.path.isfile(file_name) and not config.overwrite:
                self.load_result(file_name)  # load result for next step

                continue

            # do the analysis step
            analysis_step = getattr(self, step_names[step])
            analysis_step()

            # save the results if conditions are met
            if not os.path.isfile(file_name) or config.overwrite:
                self.save_result(file_name)

        # final message and timing
        t_b = systime.time()
        self.logger.info(f"End of analysis. Total time elapsed: {t_b - t_a:1.1f}s.")

        return None

    def load_result(self, file_name):
        """Load a result file and update the model.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        """
        # load result from file
        self.result = Result.load(file_name, logger=self.logger)

        # set the TimeSeriesModel
        self.ts_model = self.result.to_time_series_model(self.ts_model)

        return None

    def save_result(self, file_name):
        """Update the result form model and save.

        Parameters
        ----------
        file_name: str
            File name to load the results from
        """
        # update the result instance and set the identifiers
        self.result.from_time_series_model(self.ts_model, target_id=self.data.target_id, data_id=self.data.data_id)

        # save result to file
        self.result.save(file_name)

        return None
