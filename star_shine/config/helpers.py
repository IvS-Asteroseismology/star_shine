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

        except (FileNotFoundError, KeyError, tomllib.TOMLDecodeError) as e:
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


def get_images_path():
    """Get the path to the data/images folder

    Returns
    -------
    str
        Path to the data/images folder
    """
    # Use importlib.resources to find the path
    images_path = str(importlib.resources.files('star_shine.data').joinpath('images'))

    return images_path


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


def add_logging_level(level_name, level_num, method_name=None, verbose=False):
    """Comprehensively adds a new logging level to the `logging` module and the
    currently configured logging class.

    `levelName` becomes an attribute of the `logging` module with the value
    `levelNum`. `methodName` becomes a convenience method for both `logging`
    itself and the class returned by `logging.getLoggerClass()` (usually just
    `logging.Logger`). If `methodName` is not specified, `levelName.lower()` is
    used.

    To avoid accidental clobberings of existing attributes, this method will
    raise an `AttributeError` if the level name is already an attribute of the
    `logging` module or if the method name is already present

    http://stackoverflow.com/q/2183233/2988730

    Example
    -------
    >>> addLoggingLevel('TRACE', logging.DEBUG - 5)
    >>> logging.getLogger(__name__).setLevel("TRACE")
    >>> logging.getLogger(__name__).trace('that worked')
    >>> logging.trace('so did this')
    >>> logging.TRACE
    5

    """
    if not method_name:
        method_name = level_name.lower()

    if hasattr(logging, level_name):
        if verbose:
            print(f'{level_name} already defined in logging module')
        return None
    if hasattr(logging, method_name):
        if verbose:
            print(f'{method_name} already defined in logging module')
        return None
    if hasattr(logging.getLoggerClass(), method_name):
        if verbose:
            print(f'{method_name} already defined in logger class')
        return None

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(level_num):
            self._log(level_num, message, args, **kwargs)
    def logToRoot(message, *args, **kwargs):
        logging.log(level_num, message, *args, **kwargs)

    logging.addLevelName(level_num, level_name)
    setattr(logging, level_name, level_num)
    setattr(logging.getLoggerClass(), method_name, logForLevel)
    setattr(logging, method_name, logToRoot)

    return None


def get_custom_logger(target_id, save_dir, verbose):
    """Create a custom logger for logging to file and to stdout

    Parameters
    ----------
    target_id: str
        Identifier to use for the log file.
    save_dir: str
        Folder to save the log file. If empty, no saving happens.
    verbose: bool
        If set to True, information will be printed by the logger.

    Returns
    -------
    logging.Logger
        Customised logger object
    """
    # define a custom logging level
    add_logging_level('EXTRA', logging.INFO - 5, method_name=None)

    # customize the logger
    logger = logging.getLogger(target_id)  # make an instance of the logging library
    logger.setLevel(logging.EXTRA)  # set base activation level for logger

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
        s_handler.setLevel(logging.EXTRA)  # print everything with level 15 or above
        s_handler.setFormatter(s_format)
        logger.addHandler(s_handler)

    # file handler
    if save_dir != '':
        log_name = os.path.join(save_dir, f'{target_id}.log')
        f_handler = logging.FileHandler(log_name, mode='a')  # for saving
        f_handler.setLevel(logging.INFO)  # save everything with level 20 or above
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)

    return logger
