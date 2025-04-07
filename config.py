########################################################################################################################
# Star Shine settings file
########################################################################################################################
# general
OVERWRITE = False                      # Overwrite existing result files
VERBOSE = False                        # Print information during runtime
STOP_AT_STAGE = 0                      # Run the analysis up to and including this stage, 0 means all stages are run
# extraction
SELECT = 'hybrid'                      # What the next frequency selected in the extraction is based on (see docs)
# optimisation
OPTIMISE = 'fitter'                    # 'fitter' or 'sampler' method is used for model optimisation
OPTIMISE_STEP = True                   # Optimise at every step (T) or only at the end (F)
########################################################################################################################
# data and file defaults
DATA_DIR = ''                          # Root directory of the data, if empty will use current dir
SAVE_DIR = ''                          # Save directory for analysis results, if empty will use current dir
SAVE_ASCII = False                     # Save ascii variants of the HDF5 result files
# tabulated data
CN_TIME = 'time'                       # Column name for the time stamps
CN_FLUX = 'flux'                       # Column name for the flux measurements
CN_FLUX_ERR = 'flux_err'               # Column name for the flux measurement errors
# fits data
CF_TIME = 'TIME'                       # Column name for the time stamps
CF_FLUX = 'SAP_FLUX'                   # Column name for the flux [examples: SAP_FLUX, PDCSAP_FLUX, KSPSAP_FLUX]
CF_FLUX_ERR = 'SAP_FLUX_ERR'           # Column name for the flux errors [examples: SAP_FLUX_ERR, PDCSAP_FLUX_ERR]
CF_QUALITY = 'QUALITY'                 # Column name for the flux errors
APPLY_Q_FLAGS = True                   # Apply the quality flags supplied by the data source
HALVE_CHUNKS = False                   # Cut the time chunks in half (TESS data often has a discontinuity mid-sector)
########################################################################################################################
