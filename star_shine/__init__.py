"""What is Star Shine?

The name STAR SHINE is an acronym, and stands for:
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

Star Shine is a Python application that is aimed at facilitating the analysis of variable light curves.
It is broadly applicable to variable sources like pulsators, eclipsing binaries, and spotted stars.
To this end, it implements the iterative prewhitening scheme common in asteroseismology, multi-sinusoid non-linear
fitting, full integration of harmonic sinusoids, and more.
It features a high degree of automation, offering a fully hands-off operation mode.
Alternatively, each sinusoid can be extracted manually, and there are many customisation options to fine tune the
methodology to specific needs.
The code has been written with efficiency in mind, incorporating many computational optimisations and low-level
parallelisation by default, resulting in very fast operation.
The GUI provides a simple interface to directly see what is happening during the analysis of your favourite target,
while the API allows flexible access to the methods for processing batches of targets.

Star Shine's idea originated from the code Star Shadow, which was made for the analysis of eclipsing binary stars.
That code can be found here: [github.com/LucIJspeert/star_shadow](https://github.com/LucIJspeert/star_shadow).
The first, general part of that algorithm was taken as a starting point, and completely rewritten for ease of use,
customizability, and multithreaded speed.

The code was written by: Luc IJspeert


# Configuration

In general, settings are always saved in the current session, only.
At the start of a session, the configuration file config.yaml is read in from the default location (which is the
star_shine/config directory).
If it is not found in that location, or another error occurs, default values are used.

The configuration can be changed through the API using the function `sts.update_config(file_name='', settings=None)`.
Only one of `file_name` or `settings` can be used at a time.
The use of `file_name` is fairly self-explanatory: simply supply a file path to a valid STAR SHINE config file
(yaml format).
The `settings` keyword argument expects a dictionary with keys that are valid setting names.
This offers a convenient way to change only one or a few settings.

To change settings in the GUI, go to File > Settings, change any fields and click Apply.
A valid config file may also be used by going to File > Import Settings.

The configuration can be saved to file using the API function `sts.save_config(file_name='')`.
If no file name is supplied, this overwrites the config file in the default location.

In the GUI, the settings can be saved by clicking Save in the File > Settings dialog.
To save a copy of the config file under a different name, choose File > Export Settings.

## Individual settings

All settings are explained in more detail below.

### General settings

`verbose`: bool, default=True
Print information during runtime

`stop_at_stage`: int, default=0
Run the analysis up to and including this stage; 0 means all stages are run

### Extraction settings

`select_next`: str, default='hybrid'
Select the next frequency in iterative extraction based on 'amp', 'snr', or 'hybrid' (first amp then snr)

`optimise_step`: bool, default=True
Optimise with a non-linear multi-sinusoid fit at every step (T) or only at the end (F)

`replace_step`: bool, default=True
Attempt to replace closely spaced sinusoids by one sinusoid at every step (T) or only at the end (F)

`stop_criterion`: str, default='bic'
Stop criterion for the iterative extraction of sinusoids will be based on 'bic', or 'snr'

`bic_thr`: float, default=2.0
Delta-BIC threshold for the acceptance of sinusoids

`snr_thr`: float, default=-1.0
Signal-to-noise threshold for the acceptance of sinusoids, uses a built-in method if set to -1

`nyquist_factor`: float, default=1.0
The simple Nyquist frequency approximation (1/(2 delta_t_min)) is multiplied by this factor

`resolution_factor`: float, default=1.5
The frequency resolution (1/T) is multiplied by this factor

`window_width`: float, default=1.0
Periodogram spectral noise is calculated over this window width

### Optimisation settings

`min_group`: int, default=45
Minimum group size for the multi-sinusoid non-linear fit

`max_group`: int, default=50
Maximum group size for the multi-sinusoid non-linear fit (max_group > min_group)

### Data and File settings

`overwrite`: bool, default=False
Overwrite existing result files

`data_dir`: str, default=''
Root directory where the data files to be analysed are located; if empty will use current dir

`save_dir`: str, default=''
Root directory where analysis results will be saved; if empty will use current dir

`save_ascii`: bool, default=False
Save ascii variants of the HDF5 result files

### Tabulated File settings

`cn_time`: str, default='time'
Column name for the time stamps

`cn_flux`: str, default='flux'
Column name for the flux measurements

`cn_flux_err`: str, default='flux_err'
Column name for the flux measurement errors

### FITS File settings

`cf_time`: str, default='TIME'
Column name for the time stamps

`cf_flux`: str, default='SAP_FLUX'
Column name for the flux [examples: SAP_FLUX, PDCSAP_FLUX, KSPSAP_FLUX]

`cf_flux_err`: str, default='SAP_FLUX_ERR'
Column name for the flux errors [examples: SAP_FLUX_ERR, PDCSAP_FLUX_ERR, KSPSAP_FLUX_ERR]

`cf_quality`: str, default='QUALITY'
Column name for the flux quality flags

`apply_q_flags`: bool, default=True
Apply the quality flags supplied by the data source

`halve_chunks`: bool, default=False
Cut the time chunks in half (TESS data often has a discontinuity mid-sector)

### GUI settings

`dark_mode`: bool, default=False
Dark mode. [WIP]

`h_size_frac`: float, default=0.8
Horizontal window size as a fraction of the screen width

`v_size_frac`: float, default=0.8
Vertical window size as a fraction of the screen height

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
