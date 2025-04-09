"""STAR SHINE

Code written by: Luc IJspeert
"""
import os
import yaml


default_settings = {
    'verbose': False,
    'stop_at_stage': 0,

    'select': 'hybrid',
    'stop_crit': 'bic',
    'bic_thr': 2,
    'snr_thr': 4,

    'optimise': 'fitter',
    'optimise_step': True,

    'overwrite': False,
    'data_dir': '',
    'save_dir': '',
    'save_ascii': False,

    'cn_time': 'time',
    'cn_flux': 'flux',
    'cn_flux_err': 'flux_err',

    'cf_time': 'TIME',
    'cf_flux': 'SAP_FLUX',
    'cf_flux_err': 'SAP_FLUX_ERR',
    'cf_quality': 'QUALITY',

    'apply_q_flags': True,
    'halve_chunks': False,
}


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
        expected_keys = set(default_settings.keys())
        config_keys = set(config_data.keys())

        # test whether config_keys is missing items
        missing_keys = expected_keys - config_keys
        if len(missing_keys) != 0:
            raise ValueError(f"Missing keys in configuration file: {missing_keys}")

        # test whether config_keys has excess items
        excess_keys = config_keys - config_keys
        if len(excess_keys) != 0:
            raise ValueError(f"Excess keys in configuration file: {excess_keys}")

        return None

    @classmethod
    def _initialize_default_settings(cls):
        """Initializes default settings if the configuration file is not found or invalid.

        Returns
        -------
        None
        """
        for key, value in default_settings.items():
            setattr(cls._instance, key, value)

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
            cls._initialize_default_settings()

        except yaml.YAMLError as e:
            print(f"Error parsing YAML from {cls._config_path}: {e}. Using default settings.")
            cls._initialize_default_settings()

        except ValueError as e:
            print(f"Error validating configuration from {cls._config_path}: {e}. Your config file may be out of date."
                  f"Using default settings.")
            cls._initialize_default_settings()

        return None

    @classmethod
    def update_config(cls, new_config_path):
        """Updates the settings with a user defined configuration file.

        Parameters
        ----------
        new_config_path: str
            Path to a valid configuration file.

        Returns
        -------
        None
        """
        cls._config_path = new_config_path
        cls._load_config()

        return None

def get_config():
    """Use this function to get the configuration

    Returns
    -------
    Config
        The singleton instance of Config.
    """
    return Config()
