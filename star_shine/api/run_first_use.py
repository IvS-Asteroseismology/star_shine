"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python script is meant to be run before first use, it ensures that the Just-In-Time compiler has done its job.
If your own use case involves time series longer than a few thousand data points, this is strongly recommended.
If not, this is less important, but do keep in mind that the first run will be slower.

Code written by: Luc IJspeert
"""

import os
import importlib.resources
import star_shine as sts


# get the path to the test light curve
data_path = str(importlib.resources.files('star_shine.data'))
file = os.path.join(data_path, 'sim_000_lc.dat')

# execute the code
sts.analyse_lc_from_file(file, p_orb=0, i_sectors=None, stage='all', method='fitter', data_id='', save_dir=None,
                         overwrite=True, verbose=True)
