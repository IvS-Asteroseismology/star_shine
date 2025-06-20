"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the main functions that link together all functionality.

Code written by: Luc IJspeert
"""
from star_shine.config.helpers import get_config, get_config_path


# load configuration
config = get_config()
# todo: modelling red-noise with autoregressive model
# todo: amplitude factor for each sector (linear model)
# todo: consider twice nyquist for harmonics
# todo: enforce_harmonics do linear fit


def update_config(file_name='', settings=None):
    """Update the configuration using a file and/or a dictionary.

    First loads the file, then updates settings, so both could be used simultaneously.
    This alters the state of the current configuration, not the configuration file.

    Parameters
    ----------
    file_name: str, optional
        Path to the yaml configuration file.
    settings: dict, optional
        Dictionary to update specific configuration settings.

    Returns
    -------
    None
    """
    # load from file
    if file_name != '':
        config.update_from_file(file_name)

    # update individual settings
    if settings is not None:
        config.update_from_dict(settings)

    return None

def save_config(file_name=''):
    """Save the configuration to a file."""
    # if no file name is supplied, overwrite it in the default place
    if file_name == '':
        file_name = get_config_path()

    config.save_to_file(file_name)

    return None
