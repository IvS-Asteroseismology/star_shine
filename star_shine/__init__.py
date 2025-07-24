"""
# What is Star Shine?

The name STAR SHINE is an acronym, and stands for:
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

Yes... but what IS it?

Star Shine is a Python application that is aimed at facilitating the analysis of variable light curves.
It is broadly applicable to variable sources like pulsators, eclipsing binaries, and spotted stars.
To this end, it implements the iterative prewhitening scheme common in asteroseismology, multi-sinusoid non-linear
fitting, full integration of harmonic sinusoid series, and more.
It features a high degree of automation, offering a fully hands-off operation mode.
Alternatively, each sinusoid can be extracted manually, and there are many customisation options to fine tune the
methodology to specific needs.
The code has been written with efficiency in mind, incorporating many computational optimisations and low-level
parallelisation by default, resulting in very fast operation.
The Graphical User Interface (GUI) provides a simple interface to directly see what is happening during the analysis
of your favourite target, while the Application Programming Interface (API) allows flexible access to the methods for
processing batches of targets.

Star Shine's idea originated from the code Star Shadow, which was made for the analysis of (pulsating) eclipsing binary
stars.
That code can be found here: [github.com/LucIJspeert/star_shadow](https://github.com/LucIJspeert/star_shadow).
The first, general part of that algorithm was taken as a starting point, and completely rewritten for ease of use,
customizability, and multithreaded speed.


# Getting started

Star Shine is indexed on PyPI, so the simplest way to install it is to use pip:

    pip install star_shine

One can then import the package from the python environment it was installed in.
It is recommended to install into a separate, clean virtual Python environment.
Of course one can always manually download any desired release or make a fork on GitHub.
The included `pyproject.toml` file can then be used to install the project using a tool like Poetry.

The GUI is optional functionality, and its dependencies can be included when installing the package using:

    pip install star_shine[gui]

If there are issues launching the GUI, not all Pyside6 dependencies may have been installed (by e.g. pip).
In that case try installing Pyside6 separately through conda.

### Notes on library versions

The GUI uses the PySide6 library, which is tightly bound to specific Python versions.
Each PySide6 version only supports a small range of Python versions, as may be found in their documentation.
The standard Star Shine dependencies do not have this limitation.

**Star Shine has only been tested in Python 3.11 and 3.12**.
Using older versions could result in unexpected errors, however, any Python version >=3.6 is expected to work,
and Python versions >=3.8 are expected to work with the GUI.
An upper limit of Python 3.12 is set specifically for the GUI dependency pyside6, although later versions of this
package support higher Python versions.

**Package dependencies:** The following package versions have been used in the development of this code,
meaning older versions can in principle work, but this is not guaranteed.
NumPy 1.20.3, SciPy 1.7.3, Numba 0.55.1, h5py 3.7.0, Astropy 4.3.1, Pandas 1.2.3, Matplotlib 3.5.3, pyyaml 6.0.2,
pyside6 6.6.0 (optional),
pymc 5.24.0 (optional), Arviz 0.22.0 (optional), fastprogress 1.0.3 (optional).

Newer versions are expected to work, and it is considered a bug if this is not the case.
That statement does not extend to PySide6, because of its strong dependency on Python version.

> [!WARNING]
> Star Shine makes use of just-in-time compilation and caching of the compiled functions for fast operation.
> Before first use, it is recommended to run the script `run_first_use.py`. This runs a very short time-series
> (sim_000_lc.dat included in the data folder) and will make sure that the just-in-time compiler can do its magic to
> make everything run as fast as it can. Just-in-time compilation can result in more optimised machine code than
> ahead-of-time compilation. Note that compilation takes time, but this only applies to the first time each function is
> used. In short: first time use is not indicative of the final runtime.


## Example use

Since the main feature of Star Shine is its fully automated operation, taking advantage of its functionality is
as simple as running one function.
First, set up a `Data` object:

```python
import star_shine as sts
data = sts.Data.load_data(file_list)
```

The `file_list` is a list of file path strings.
Then, use the `Data` object to initialise the `Pipeline`, and simply run it:

```python
pipeline = sts.Pipeline(data)
pipeline.run()
```

A provided light curve file is expected to contain a time column, flux measurements (non-negative), and flux
measurement errors.
Astronomical data products in FITS format can also be read by the Data class, provided the correct column names are
configured (by default these use standard TESS data product names).
The normalisation of the time series handled automatically per time chunk (each separate file is considered a
time chunk - e.g. a sector in TESS jargon).
The stage parameter can be set to indicate which parts of the analysis will be performed, see the configuration
documentation for options.

If a save_dir is given, the outputs are saved in that directory with either the given target identifier or the file
name of the first data file as identifier.
If not given, files are saved in a subdirectory of where the data is.
The 'overwrite' argument can be used to either overwrite old data, or to continue from a previous save file.
The pipeline can print useful progress information if verbose is set to True in the configuration.
In the case of harmonic signals such as in eclipsing binaries, and if an orbital period is known beforehand,
this information can be used to find orbital harmonics in the prewhitened frequencies.
If not provided, a period is found using a combination of phase dispersion minimisation, Lomb-Scargle periodogram
amplitude, and the extracted frequencies.

The GUI can be started by running

```python
sts.launch_gui()
```


# Working Principles

What follows is an extremely detailed description of the inner workings of Star Shine.
This includes top-level descriptions, as well as motivations for the choices of specific algorithms [WIP].

In broad lines, when running the fully automated pipeline, the following recipe is followed regardless of configuration:

1. Extract all sinusoids.
    We start by extracting the frequency with the highest amplitude (or signal-to-noise ratio, see settings) one by one,
    directly from the Lomb-Scargle periodogram until the BIC (or signal-to-noise ratio, see settings) does not
    significantly (see settings) improve anymore.

2. Multi-sinusoid NL-LS optimisation.
    The sinusoid parameters are optimised with a non-linear least-squares method, using groups of sinusoids to limit
    the number of free parameters.

3. Measure the harmonic base frequency and couple the harmonic sinusoids.
    Primarily meant for the search of an eclipsing binary orbital period, the global search algorithm can find
    prominent series of harmonic sinusoids. It uses phase dispersion, Lomb-Scargle amplitude and length/filling factor
    of the harmonic series in the list of extracted sinusoids.
    Knowing the base frequency, it then sets the frequencies of the harmonics to their integer multiple values to
    couple them.
    It is possible to provide a base frequency if it is already known.
    This base frequency will still be optimised in consequent optimisation steps.

4. Attempt to extract additional sinusoids.
    The decreased number of free parameters (2 vs. 3) may allow the extraction of more harmonics, since the BIC
    punishes for free parameters.
    It is also attempted to extract more sinusoids like in step 1 again, now taking into account the presence of
    harmonics.

5. Multi-sinusoid NL-LS optimisation with coupled harmonics.
    The sinusoid parameters are optimised with a non-linear least-squares method, using groups of sinusoids to limit
    the number of free parameters.
    Base frequencies and their respective coupled harmonics are optimised simultaneously with every sinusoid group.

All steps include a final cleanup of the sinusoids.
This means that two things are attempted.
The first is to try to remove individual sinusoids, and checking whether the BIC improves by doing so.
Secondly, it is tested whether groups of closely spaced sinusoids can be replaced by a single sinusoid while improving
the BIC.
With closely spaced is meant chains of sinusoids within the frequency resolution of each other.


# Configuration

In general, settings are always saved in the current session, only.
At the start of a session, the configuration file config.yaml is read in from the default location (which is the
star_shine/config directory).
If it is not found in that location, or another error occurs, default values are used.

The configuration can be changed through the API using the function `sts.update_config(file_name='', settings=None)`.
Only one of `file_name` or `settings` can be used at a time.
The use of `file_name` is fairly self-explanatory: simply supply a file path to a valid Star Shine config file
(yaml format).
The `settings` keyword argument expects a dictionary with keys that are valid setting names.
This offers a convenient way to change only one or a few settings at a time.

To change settings in the GUI, go to File > Settings, change any fields and click Apply.
A valid config file may also be used by going to File > Import Settings.

The configuration can be saved to file using the API function `sts.save_config(file_name='')`.
If no file name is supplied, this overwrites the config file in the default location.

In the GUI, the settings can be saved by clicking Save in the File > Settings dialog.
To save a copy of the config file under a different name or directory, choose File > Export Settings.

All settings are explained in more detail below.

## General settings

`verbose`: bool, default=True

Print information during runtime.
This includes detailed progress updates that are not normally logged.
Some status messages are always logged to a file in the same directory as the analysis results.
This setting is always active in the GUI.

`stop_at_stage`: int, default=0

Run the analysis up to and including this stage; 0 means all stages are run.
When running the full pipeline, a predefined sequence of analysis steps is performed.
The followed recipe is: 'iterative_prewhitening', 'optimise_sinusoid', 'couple_harmonics', 'iterative_prewhitening',
'optimise_sinusoid'.
For more detail, see "Working Principles".

## Extraction settings

`select_next`: str, default='hybrid'

Select the next sinusoid in iterative extraction based on amplitude ('amp'), signal-to-noise ratio ('snr'), or first
amplitude and then signal-to-noise ratio ('hybrid').
The latter option is an attempt at avoiding red noise when there is still high-frequency signal present at lower
amplitudes.

`optimise_step`: bool, default=True

Optimise with a multi-sinusoid non-linear fit at every step (True) or only at the end (False).
In case of False, the fit at the end is only automatically performed when running the full pipeline.
When running the iterative prewhitening separately, the fit at the end has to be performed manually.

`replace_step`: bool, default=True

Attempt to replace closely spaced sinusoids by one sinusoid at every step (True) or only at the end (False).
Contrary to the optimise_step setting, this is always included automatically at the end of iterative prewhitening.

`stop_criterion`: str, default='bic'

Stop criterion for the iterative extraction of sinusoids will be based on BIC ('bic'), or signal-to-noise ratio ('snr').
The BIC provides a more objective measure of the information content of the time series.
It is recommended to use the BIC here, in tandem with a signal-to-noise ratio cut before physical interpretation.

`bic_thr`: float, default=2.0

Delta-BIC threshold for the acceptance of sinusoids, if stop_criterion is set to 'bic'.
Values of 2 to 10 are rule-of-thumb values that approximately correspond to certainty levels 'weak evidence' to
'strong evidence'.

`snr_thr`: float, default=-1.0

Signal-to-noise threshold for the acceptance of sinusoids, uses a built-in method if set to -1.
This is both for if stop_criterion is set to 'snr', and for use in the built-in method for making signal-to-noise ratio
cuts after the time series analysis is done.
The built-in method is specific to TESS, see the documentation for more information.

`nyquist_factor`: float, default=1.0

The simple Nyquist frequency approximation (1/(2 delta_t_min)) is multiplied by this factor.
The Nyquist frequency is defined as the highest frequency that can be found in a time series dataset of fixed time
interval.
For irregular time intervals, it can be approximated by using the minimum time interval that occurs in the time series.
However, a more rigorous approach shows that the actual Nyquist frequency can be a large integer times higher than this
estimate.
Therefore, this factor allows the user to increase the highest frequency of sinusoid extraction.

Note: harmonic sinusoids are always extracted up to twice the limiting frequency value, due to their additional
constraint.

`resolution_factor`: float, default=1.5

The frequency resolution (1/T) is multiplied by this factor.
The frequency resolution defines the separation between two periodogram peaks that is needed to distinguish them from
each other.
Due to complex window functions and subsequent spectral windows, it may be desirable to increase this value.
By default, a factor of one and a half is already applied, as is common practice in asteroseismology.

Note: the frequency resolution is not used as an exclusion zone around extracted frequencies, and this is not a
supported feature of Star Shine.
Experience teaches that this leads to worse models.

`window_width`: float, default=1.0

Periodogram spectral noise is calculated over this window width.
Only influences the extraction in the signal-to-noise ratio mode of picking the next sinusoid.

## Optimisation settings

`min_group`: int, default=45

Minimum group size for the multi-sinusoid non-linear fit.
See `max_group` below for more information.

`max_group`: int, default=50

Maximum group size for the multi-sinusoid non-linear fit (max_group > min_group).
Simultaneously optimising a larger number of sinusoids is beneficial for the quality of the fit, but may drastically
increase the time it takes to complete.
Make sure max_group is always larger than min_group.
The final group sizes are determined during runtime by the largest amplitude jump within the group size limits.

## Data and File settings

`overwrite`: bool, default=False

Overwrite existing result files.
When running the full pipeline, save files are automatically generated.
Any pre-existing save files of the same name will be overwritten if this setting is set to True.
Otherwise, existing save files of the same name will be loaded and their results used to skip ahead in the pipeline.

`data_dir`: str, default=''

Root directory where the data files to be analysed are located; if empty will use current dir.
Appended to the front of the file name(s).
Mainly for use in automated scripts, not relevant in the GUI.

`save_dir`: str, default=''

Root directory where analysis results will be saved; if empty will use current dir.
Appended to the front of the file name(s).
A subdirectory is automatically made with the target identifier to store the results and log file.

`save_ascii`: bool, default=False

Save ascii variants of the HDF5 result files.
Saves several CSV files per HDF5 file, due to format constraints.

## Tabulated File settings

`cn_time`: str, default='time'

Column name for the time stamps.
When using CSV data, tries to read in time data from this column name.

`cn_flux`: str, default='flux'

Column name for the flux measurements.
When using CSV data, tries to read in flux data from this column name.

`cn_flux_err`: str, default='flux_err'

Column name for the flux measurement errors.
When using CSV data, tries to read in flux error data from this column name.

## FITS File settings

`cf_time`: str, default='TIME'

Column name for the time stamps.
When using FITS data, tries to read in time data from this column name.

`cf_flux`: str, default='SAP_FLUX'

Column name for the flux.
When using FITS data, tries to read in flux data from this column name.
Examples: SAP_FLUX, PDCSAP_FLUX, KSPSAP_FLUX.

`cf_flux_err`: str, default='SAP_FLUX_ERR'

Column name for the flux errors.
When using FITS data, tries to read in flux error data from this column name.
Examples: SAP_FLUX_ERR, PDCSAP_FLUX_ERR, KSPSAP_FLUX_ERR.

`cf_quality`: str, default='QUALITY'

Column name for the flux quality flags.
When using FITS data, tries to read in quality flags from this column name.

`apply_q_flags`: bool, default=True

Apply the quality flags supplied by the data source.
Quality flags are assumed to follow the convention that zero means a good flux measurement.

`halve_chunks`: bool, default=False

Cut the time chunks in half (TESS data often has a discontinuity mid-sector).
This is a convenience feature, mainly for users of TESS data, that makes sure the linear model is able to fit
residual trends well in the case of flux discontinuities in the middle of a time chunk.

## GUI settings

`dark_mode`: bool, default=False

Dark mode. [WIP]
If this interest you, make a feature request (or react to it if exists) on GitHub.

`h_size_frac`: float, default=0.8

Horizontal window size as a fraction of the screen width.

`v_size_frac`: float, default=0.8

Vertical window size as a fraction of the screen height.


# Credits

The code was written and documented by: Luc IJspeert, PhD in astronomy and astrophysics.

Contributions were made to setting options and ideation by: Mathijs Vanrespaille.

"""

from .api.main import *
from .api.data import Data
from .api.result import Result
from .api.pipeline import Pipeline

try:
    # GUI
    from .gui.gui_app import launch_gui
except ImportError as e:
    print(e)
    print('GUI unavailable, likely missing dependency PySide6.')
    pass

__all__ = ['gui', 'api', 'core', 'config', 'data']
