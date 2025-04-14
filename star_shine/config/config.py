"""STAR SHINE

Code written by: Luc IJspeert
"""
import os
import yaml
import importlib.resources


class Config:
    """A singleton class that manages application configuration settings.

    Attributes
    ----------
    _instance: Config object, None
        The single instance of the Config class, or None.
    _config_path: str
        Path to the configuration file.

    Methods
    -------
    __new__(cls)
        Ensures only one instance of the Config class is created.
    _load_config(cls)
        Loads and validates the configuration from a file.
    _validate_config(cls, config_data)
        Validates the loaded configuration data.
    _initialize_default_settings(cls)
        Initializes default settings if the configuration file is not found or invalid.
    """
    _instance = None
    _config_path = os.path.join(os.path.dirname(__file__), 'data', 'config.yaml')

    # default settings below
    # General settings
    verbose: bool = False
    stop_at_stage: int = 0

    # Extraction settings
    select: str = 'hybrid'
    stop_crit: str = 'bic'
    bic_thr: float = 2.
    snr_thr: float = -1.
    f_max: float = -1.

    # Optimisation settings
    optimise: str = 'fitter'
    optimise_step: bool = True

    # Data and file settings
    overwrite: bool = False
    data_dir: str = ''
    save_dir: str = ''
    save_ascii: bool = False

    # Tabulated data settings
    cn_time: str = 'time'
    cn_flux: str = 'flux'
    cn_flux_err: str = 'flux_err'

    # Fits data settings
    cf_time: str = 'TIME'
    cf_flux: str = 'SAP_FLUX'
    cf_flux_err: str = 'SAP_FLUX_ERR'
    cf_quality: str = 'QUALITY'
    apply_q_flags: bool = True
    halve_chunks: bool = False

    def __new__(cls):
        """Ensures that only one instance of the Config class is created.

        Returns
        -------
        Config
            The single instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()

        return cls._instance

    @classmethod
    def _validate_config(cls, config_data):
        """Validates the loaded configuration data.

        Parameters
        ----------
        config_data: dict
            The loaded configuration data.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the configuration data is invalid.
        """
        # make sets out of the keywords
        expected_keys = set(key for key in dir(cls) if not callable(getattr(cls, key)) and not key.startswith('_'))
        config_keys = set(config_data.keys())

        # test whether config_keys is missing items
        missing_keys = expected_keys - config_keys
        if len(missing_keys) != 0:
            raise ValueError(f"Missing keys in configuration file: {missing_keys}")

        # test whether config_keys has excess items
        excess_keys = config_keys - expected_keys
        if len(excess_keys) != 0:
            raise ValueError(f"Excess keys in configuration file: {excess_keys}")

        return None

    @classmethod
    def _load_config(cls):
        """Loads and validates the configuration from a file.

        Returns
        -------
        None

        Notes
        -----
        - `FileNotFoundError`: Catches the error if the configuration file is not found.
        - `YAMLError`: Catches errors related to parsing YAML files.
        - `ValueError`: Catches errors related to the validation of the configuration.
        In these cases, the default settings are loaded.
        """
        # try to open the config file
        try:
            with open(cls._config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                cls._validate_config(config_data)
                for key, value in config_data.items():
                    setattr(cls._instance, key, value)

        except FileNotFoundError:
            print(f"Configuration file {cls._config_path} not found. Using default settings.")

        except yaml.YAMLError as e:
            print(f"Error parsing YAML from {cls._config_path}: {e}. Using default settings.")

        except ValueError as e:
            print(f"Error validating configuration from {cls._config_path}: {e}. Your config file may be out of date. "
                  f"Using default settings.")

        return None

    def update_from_file(self, new_config_path):
        """Updates the settings with a user defined configuration file.

        Parameters
        ----------
        new_config_path: str
            Path to a valid configuration file.

        Returns
        -------
        None
        """
        self._config_path = new_config_path
        self._load_config()

        return None

    def update_from_dict(self, settings):
        """Updates the settings with user defined keyword arguments.

        Parameters
        ----------
        settings: dict
            Configuration settings.

        Returns
        -------
        None
        """
        # remember invalid items
        invalid = dict()

        # set the valid attributes
        for key, value in settings.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                invalid[key] = value

        # Print a message about items that were invalid
        if len(invalid) > 0:
            print(f"Invalid items that were not updated: {invalid}")

        return None


def get_config():
    """Use this function to get the configuration

    Returns
    -------
    Config
        The singleton instance of Config.
    """
    return Config()


def get_config_path():
    """Get the path to the configuration file

    Returns
    -------
    str
        Path to the config file
    """
    # Use importlib.resources to find the path
    config_path = str(importlib.resources.files('star_shine.config').joinpath('config.yaml'))

    return config_path
