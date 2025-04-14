"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains some getter functions for configuration purposes.

Code written by: Luc IJspeert
"""

import os
import logging
import tomllib
import importlib.metadata
import importlib.resources

from star_shine.config import config as cnfg


def get_version():
    """Get the version of the code from the metadata, or the project file if that fails.

    Returns
    -------
    str
        The version of the code

    Raises
    ------
    FileNotFoundError
        In case the package metadata is not set and subsequently the pyproject.toml was not found
    """
    try:
        version = importlib.metadata.version('star_shine')

    except importlib.metadata.PackageNotFoundError:
        # Fallback to reading from pyproject.toml if not installed
        project_root = os.path.dirname(os.path.abspath(__file__))
        pyproject_path = os.path.join(project_root, '../../pyproject.toml')  # Adjust the path as needed

        try:
            with open(pyproject_path, 'rb') as f:
                pyproject_data = tomllib.load(f)

        except (FileNotFoundError, KeyError) as e:
            raise FileNotFoundError("Could not find or parse version in pyproject.toml")

        version = pyproject_data['project']['version']

    return version


def get_config():
    """Use this function to get the configuration

    Returns
    -------
    Config
        The singleton instance of Config.
    """
    return cnfg.get_config()


def get_config_path():
    """Get the path to the configuration file

    Returns
    -------
    str
        Path to the config file
    """
    # Use importlib.resources to find the path
    config_path = cnfg.get_config_path()

    return config_path


def get_mpl_stylesheet_path():
    """Get the path to the matplotlib stylesheet

    Returns
    -------
    str
        Path to the matplotlib stylesheet
    """
    # Use importlib.resources to find the path
    stylesheet_path = str(importlib.resources.files('star_shine.config').joinpath('mpl_stylesheet.dat'))

    return stylesheet_path


def get_custom_logger(save_dir, target_id, verbose):
    """Create a custom logger for logging to file and to stdout

    Parameters
    ----------
    save_dir: str
        folder to save the log file
    target_id: str
        Identifier to use for the log file
    verbose: bool
        If set to True, information will be printed by the logger

    Returns
    -------
     : None
    """
    # customize the logger
    logger = logging.getLogger(__name__)  # make an instance of the logging library
    logger.setLevel(logging.INFO)  # set base activation level for logger

    # make formatters for the handlers
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    f_format = logging.Formatter(fmt='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
                                 datefmt='%Y-%m-%d %H:%M:%S')

    # remove existing handlers to avoid duplicate messages
    if logger.hasHandlers():
        logger.handlers.clear()

    # make stream handler
    if verbose:
        s_handler = logging.StreamHandler()  # for printing
        s_handler.setLevel(logging.INFO)  # print everything with level 20 or above
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    # file handler
    logname = os.path.join(save_dir, f'{target_id}_analysis', f'{target_id}.log')
    f_handler = logging.FileHandler(logname, mode='a')  # for saving
    f_handler.setLevel(logging.INFO)  # save everything with level 20 or above
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return None
