"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python script is meant to be run before first use, it ensures that the Just-In-Time compiler has done its job
and cache the compiled functions. If your own use case involves time series longer than a few thousand data points,
this is strongly recommended. If not, this is less important, but do keep in mind that the first run will be slower.
"""

import os
import importlib.resources
import star_shine as sts


def first_use_script():
    # get the path to the test light curve
    data_path_traversable = importlib.resources.files('star_shine.data')
    target_id = 'sim_000_lc'
    file = data_path_traversable.joinpath(target_id + '.dat').as_posix()
    data_path = os.path.split(file)[0]
    file_list = [file]

    # initialise the data and pipeline
    data = sts.Data.load_data(file_list, data_dir='', target_id=target_id, data_id='', logger=None)
    pipeline = sts.Pipeline(data, save_dir=data_path, logger=None)

    # first only extract 10 sinusoids
    pipeline.iterative_prewhitening(n_extract=10)
    pipeline.optimise_sinusoid()

    # start new pipeline for the second part
    pipeline = sts.Pipeline(data, save_dir=data_path, logger=None)

    # do a full run without fitting every step and lower group size
    sts.update_config(settings={'optimise_step': False, 'min_group': 15, 'max_group': 20})
    pipeline.run()


if __name__ == '__main__':
    first_use_script()
