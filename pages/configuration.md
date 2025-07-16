## STAR SHINE Configuration

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

All settings are explained in more detail below.

### General settings 
verbose description:
Print information during runtime
verbose: True

stop_at_stage description:
Run the analysis up to and including this stage; 0 means all stages are run
stop_at_stage: 0

### Extraction settings 
select_next description:
Select the next frequency in iterative extraction based on 'amp', 'snr', or 'hybrid' (first amp then snr)
select_next: 'hybrid'

optimise_step description:
Optimise with a non-linear multi-sinusoid fit at every step (T) or only at the end (F)
optimise_step: True

replace_step description:
Attempt to replace closely spaced sinusoids by one sinusoid at every step (T) or only at the end (F)
replace_step: True

stop_criterion description:
Stop criterion for the iterative extraction of sinusoids will be based on 'bic', or 'snr'
stop_criterion: 'bic'

bic_thr description:
Delta-BIC threshold for the acceptance of sinusoids
bic_thr: 2.0

snr_thr description:
Signal-to-noise threshold for the acceptance of sinusoids, uses a built-in method if set to -1
snr_thr: -1.0

nyquist_factor description:
The simple Nyquist frequency approximation (1/(2 delta_t_min)) is multiplied by this factor
nyquist_factor: 1.0

resolution_factor description:
The frequency resolution (1/T) is multiplied by this factor
resolution_factor: 1.5

window_width description:
Periodogram spectral noise is calculated over this window width
window_width: 1.0

### Optimisation settings 
min_group description:
Minimum group size for the multi-sinusoid non-linear fit
min_group: 45

max_group description:
Maximum group size for the multi-sinusoid non-linear fit (max_group > min_group)
max_group: 50

### Data and File settings 
overwrite description:
Overwrite existing result files
overwrite: False

data_dir description:
Root directory where the data files to be analysed are located; if empty will use current dir
data_dir: ''

save_dir description:
Root directory where analysis results will be saved; if empty will use current dir
save_dir: ''

save_ascii description:
Save ascii variants of the HDF5 result files
save_ascii: False

### Tabulated File settings 
cn_time description:
Column name for the time stamps
cn_time: 'time'

cn_flux description:
Column name for the flux measurements
cn_flux: 'flux'

cn_flux_err description:
Column name for the flux measurement errors
cn_flux_err: 'flux_err'

### FITS File settings 
cf_time description:
Column name for the time stamps
cf_time: 'TIME'

cf_flux description:
Column name for the flux [examples: SAP_FLUX, PDCSAP_FLUX, KSPSAP_FLUX]
cf_flux: 'SAP_FLUX'

cf_flux_err description:
Column name for the flux errors [examples: SAP_FLUX_ERR, PDCSAP_FLUX_ERR, KSPSAP_FLUX_ERR]
cf_flux_err: 'SAP_FLUX_ERR'

cf_quality description:
Column name for the flux quality flags
cf_quality: 'QUALITY'

apply_q_flags description:
Apply the quality flags supplied by the data source
apply_q_flags: True

halve_chunks description:
Cut the time chunks in half (TESS data often has a discontinuity mid-sector)
halve_chunks: False

### GUI settings
dark_mode description:
Dark mode. [WIP]
dark_mode: False

h_size_frac description:
Horizontal window size as a fraction of the screen width
h_size_frac: 0.8

v_size_frac description:
Vertical window size as a fraction of the screen height
v_size_frac: 0.8
