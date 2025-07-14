# STAR SHINE
### Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction


![Language Badge](https://img.shields.io/badge/Language-Python-blue.svg)
<a href="./LICENCE.md"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg" alt="License Badge"/></a>
[![Tests](https://github.com/LucIJspeert/star_shine/actions/workflows/tests.yaml/badge.svg)](https://github.com/LucIJspeert/star_shine/actions/workflows/tests.yaml)

[//]: # (<a href="https://github.com/LucIJspeert/star_shine/blob/master/LICENCE.md"><img src="https://img.shields.io/github/license/LucIJspeert/star_shine" alt="License Badge"/></a>)

[//]: # (make the badges dynamic...)

## What is STAR SHINE?
STAR SHINE is a Python application that is aimed at facilitating the analysis of variable light curves. It is broadly 
applicable to variable sources like pulsators, eclipsing binaries, and spotted stars. To this end, it implements the 
iterative prewhitening scheme common in asteroseismology, multi-sinusoid non-linear fitting, full integration of harmonic 
sinusoids, and more. It features a high degree of automation, offering a fully hands-off operation mode. Alternatively, each 
sinusoid can be extracted manually, and there are many customisation options to fine tune the methodology to specific needs. 
The code has been written with efficiency in mind, incorporating many computational optimisations and low-level 
parallelisation by default, resulting in very fast operation. The GUI provides a simple interface to directly see what is 
happening during the analysis of your favourite target, while the API allows flexible access to the methods for processing 
batches of targets.

![Star Shine logo](star_shine/data/images/Star_Shine_transparent.png?raw=true)

### Reference Material

* The algorithm that this application stems from has been documented, tested and applied in the publication: [Automated eccentricity measurement from raw eclipsing binary light curves with intrinsic variability](https://ui.adsabs.harvard.edu/abs/2024arXiv240206084I/abstract)

## Getting started

The easiest way to install STAR SHINE is to use pip:

    pip install star_shine

One can then import the package from the python environment it was installed in. 
Of course one can always still manually download it or make a fork on GitHub. 
It is recommended to get the latest release from the GitHub page. 

The GUI is optional functionality, and its dependencies can be included when installing the package using:

    pip install star_shine[gui]

**STAR SHINE has only been tested in Python 3.11**. Using older versions could result in unexpected errors, 
although any Python version >=3.6 is expected to work.

**Package dependencies:** The following package versions have been used in the development of this code, 
meaning older versions can in principle work, but this is not guaranteed. NumPy 1.20.3, SciPy 1.7.3, Numba 0.55.1, 
h5py 3.7.0, Astropy 4.3.1, Pandas 1.2.3, Matplotlib 3.5.3, pyyaml 6.0.2, pyside6 6.0.0 (optional), 
pymc3 3.11.4 (optional), theano 1.1.2 (optional), Arviz 0.11.4 (optional), fastprogress 1.0.0 (optional).

Newer versions are expected to work, and it is considered a bug if this is not the case.

Before first use, it is recommended to run one very short time-series (for example sim_000_lc.dat included in the data 
folder). This will make sure that the just-in-time compiler can do its magic and make everything run as fast as it can. 
See the script run_first_use.py.


### Example use

Since the main feature of STAR SHINE is its fully automated operation, taking advantage of its functionality is 
as simple as running one function. First, set up a `Data` object:

    import star_shine as sts
    data = sts.Data.load_data(file_list)

Then, use the `Data` object to initialise the `Pipeline`, and simply run it:

    pipeline = sts.Pipeline(data)
    pipeline.run()

A provided light curve file is expected to contain a time column, flux measurements (non-negative), and flux 
measurement errors. Astronomical data products in FITS format can also be read by the Data class, provided the 
correct column names are configured (by default these use standard TESS data product names). The normalisation of the 
time series handled automatically per time chunk (each separate file is considered a time chunk - e.g. a sector in 
TESS jargon). The stage parameter can be set to indicate which parts of the analysis will be performed, see the 
documentation for options.

If a save_dir is given, the outputs are saved in that directory with either the given target identifier or the file 
name of the first data file as identifier. If not given, files are saved in a subdirectory of where the data is.
The 'overwrite' argument can be used to either overwrite old data, or to continue from a previous save file. 
The pipeline can print useful progress information if verbose is set to True in the configuration. In the case of 
harmonic signals such as in eclipsing binaries, and if an orbital period is known beforehand, this information can be 
used to find orbital harmonics in the prewhitened frequencies. If not provided, a period is found using a 
combination of phase dispersion minimisation, Lomb-Scargle periodogram amplitude, and the extracted frequencies.


### Explanation of output

Results are saved mainly in hdf5 files, in addition to csv and sometimes nc4 files. A plain text log file keeps track 
of the start and end time of the analysis and can contain important messages about the operation of the algorithm, 
like a reason for early termination.

Currently, there are a total of 5 analysis steps. Normal operation can terminate at several intermediate stages. 
A log entry is made when this happens containing further information. The analysis can stop if for example no 
frequencies were extracted, or not enough (orbital) harmonics are found.

Each step produces at least an .hdf5 file with all the model parameters from that stage of the analysis. 
The io module contains a function for reading these files, `load_result_hdf5`, which outputs a convenient 
dictionary. However, the main way to interact with the results is through the Result object, provided by the Pipeline.
With the Result object, it is also possible to save ascii variants of the results (multiple files are generated due
to the less efficient manner of data storage).

The .nc4 files (a wrapper for hdf5) contain pymc3 sampling chains.


### Diagnostic plots

There are several plotting functions available in the visualisation module. A simpler way to visualise the analysis
is to use the GUI, which shows the data, periodogram, and the state of the model as the analysis progresses.


## Bugs and Issues

Despite all the testing, I am certain that there are still bugs in this code, or will be created in future versions. 

If you happen to come across any bugs or issues, *please* contact me. Only known bugs can be resolved.
This can be done through opening an issue on the STAR SHINE GitHub page: 
[LucIJspeert/star_shine/issues](https://github.com/LucIJspeert/star_shineshine/issues).

If you are (going to be) working on new or improved features, I would love to hear from you and see if it can be 
implemented in the source code.


## Contact

For questions and suggestions, please contact:

**Main developer:** Luc IJspeert, PhD in astronomy and astrophysics
