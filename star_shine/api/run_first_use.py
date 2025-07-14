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
data_path_traversable = importlib.resources.files('star_shine.data')
target_id = 'sim_000_lc'
file = data_path_traversable.joinpath(target_id + '.dat').as_posix()
data_path = os.path.split(file)[0]
file_list = [file]

# initialise the data and pipeline
data = sts.Data.load_data(file_list, data_dir='', target_id=target_id, data_id='', logger=None)
pipeline = sts.Pipeline(data, save_dir=data_path, logger=None)
pipeline.run()
