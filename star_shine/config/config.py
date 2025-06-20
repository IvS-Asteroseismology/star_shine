"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This module contains the configuration class.

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

    resolution_factor: float
        Number that multiplies the resolution (1/T).

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
    _config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')

    # default settings below
    # General settings
    verbose: bool = False
    stop_at_stage: int = 0

    # Extraction settings
    select_next: str = 'hybrid'
    optimise_step: bool = True
    replace_step: bool = True
    stop_criterion: str = 'bic'
    bic_thr: float = 2.
    snr_thr: float = -1.
    nyquist_factor: float = 1.
    resolution_factor: float = 1.5
    window_width: float = 1.

    # Data and file settings
    overwrite: bool = False
    data_dir: str = ''
    save_dir: str = ''
    save_ascii: bool = False

    # Tabulated files settings
    cn_time: str = 'time'
    cn_flux: str = 'flux'
    cn_flux_err: str = 'flux_err'

    # Fits files settings
    cf_time: str = 'TIME'
    cf_flux: str = 'SAP_FLUX'
    cf_flux_err: str = 'SAP_FLUX_ERR'
    cf_quality: str = 'QUALITY'
    apply_q_flags: bool = True
    halve_chunks: bool = False

    # GUI settings
    dark_mode: bool = False
    h_size_frac: float = 0.8
    v_size_frac: float = 0.8

    def __new__(cls):
        """Ensures that only one instance of the Config class is created.

        Returns
        -------
        Config
            The single instance of the Config class.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config(cls._instance)

        return cls._instance

    @classmethod
    def _validate_config(cls, config_data):
        """Validates the loaded configuration data.

        Parameters
        ----------
        config_data: dict
            The loaded configuration data.

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

    def _load_config(self):
        """Loads and validates the configuration from a file.

        Notes
        -----
        - `FileNotFoundError`: Catches the error if the configuration file is not found.
        - `YAMLError`: Catches errors related to parsing YAML files.
        - `ValueError`: Catches errors related to the validation of the configuration.
        In these cases, the default settings are loaded.
        """
        # try to open the config file
        try:
            with open(self._config_path, 'r') as file:
                config_data = yaml.safe_load(file)
                self._validate_config(config_data)
                for key, value in config_data.items():
                    setattr(self._instance, key, value)

        except FileNotFoundError:
            print(f"Configuration file {self._config_path} not found. Using default settings.")

        except yaml.YAMLError as e:
            print(f"Error parsing YAML from {self._config_path}: {e}. Using default settings.")

        except ValueError as e:
            print(f"Error validating configuration from {self._config_path}: {e}. Your config file may be out of date. "
                  f"Using default settings.")

        return None

    def update_from_file(self, new_config_path):
        """Updates the settings with a user defined configuration file.

        Parameters
        ----------
        new_config_path: str
            Path to a valid configuration file.
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

    def save_to_file(self, new_config_path):
        """Save the configuration to a YAML file manually.

        Parameters
        ----------
        new_config_path: str
            Path to a valid configuration file.
        """
        # make sure the file has yaml extension
        if not (new_config_path.endswith('.yaml') or new_config_path.endswith('.yml')):
            new_config_path += '.yaml'

        # if save_dir is usr dir, revert to '' for saving (this is specifically for GUI behaviour)
        save_dir_value = self.save_dir
        if self.save_dir == os.path.expanduser('~'):
            save_dir_value = ''

        line_width = 120

        with open(new_config_path, 'w') as file:
            file.write("#" * line_width + "\n")
            file.write(fill_header_str("Star Shine settings file", line_width, fill_value='-', end='\n'))
            file.write("#" * line_width + "\n")

            file.write(fill_header_str("General settings", line_width, fill_value='-', end='\n'))

            desc = "Print information during runtime"
            file.write(config_item_description("verbose", self.verbose, desc))

            desc = "Run the analysis up to and including this stage; 0 means all stages are run"
            file.write(config_item_description("stop_at_stage", self.stop_at_stage, desc))

            file.write(fill_header_str("Extraction settings", line_width, fill_value='-', end='\n'))

            desc = ("Select the next frequency in iterative extraction based on 'amp', 'snr', "
                    "or 'hybrid' (first amp then snr)")
            file.write(config_item_description("select_next", self.select_next, desc))

            desc = "Optimise with a non-linear multi-sinusoid fit at every step (T) or only at the end (F)"
            file.write(config_item_description("optimise_step", self.optimise_step, desc))

            desc = ("Attempt to replace closely spaced sinusoids by one sinusoid "
                    "at every step (T) or only at the end (F)")
            file.write(config_item_description("replace_step", self.replace_step, desc))

            desc = "Stop criterion for the iterative extraction of sinusoids will be based on 'bic', or 'snr'"
            file.write(config_item_description("stop_criterion", self.stop_criterion, desc))

            desc = "Delta-BIC threshold for the acceptance of sinusoids"
            file.write(config_item_description("bic_thr", self.bic_thr, desc))

            desc = "Signal-to-noise threshold for the acceptance of sinusoids, uses a built-in method if set to -1"
            file.write(config_item_description("snr_thr", self.snr_thr, desc))

            desc = "The simple Nyquist frequency approximation (1/(2 delta_t_min)) is multiplied by this factor"
            file.write(config_item_description("nyquist_factor", self.nyquist_factor, desc))

            desc = "The frequency resolution (1/T) is multiplied by this factor"
            file.write(config_item_description("resolution_factor", self.resolution_factor, desc))

            desc = "Periodogram spectral noise is calculated over this window width"
            file.write(config_item_description("window_width", self.window_width, desc))

            file.write("#" * line_width + "\n")
            file.write(fill_header_str("Data and File settings", line_width, fill_value='-', end='\n'))

            desc = "Overwrite existing result files"
            file.write(config_item_description("overwrite", self.overwrite, desc))

            desc = "Root directory where the data files to be analysed are located; if empty will use current dir"
            file.write(config_item_description("data_dir", self.data_dir, desc))

            desc = "Root directory where analysis results will be saved; if empty will use current dir"
            file.write(config_item_description("save_dir", save_dir_value, desc))

            desc = "Save ascii variants of the HDF5 result files"
            file.write(config_item_description("save_ascii", self.save_ascii, desc))

            file.write(fill_header_str("Tabulated File settings", line_width, fill_value='-', end='\n'))

            desc = "Column name for the time stamps"
            file.write(config_item_description("cn_time", self.cn_time, desc))

            desc = "Column name for the flux measurements"
            file.write(config_item_description("cn_flux", self.cn_flux, desc))

            desc = "Column name for the flux measurement errors"
            file.write(config_item_description("cn_flux_err", self.cn_flux_err, desc))

            file.write(fill_header_str("FITS File settings", line_width, fill_value='-', end='\n'))

            desc = "Column name for the time stamps"
            file.write(config_item_description("cf_time", self.cf_time, desc))

            desc = "Column name for the flux [examples: SAP_FLUX, PDCSAP_FLUX, KSPSAP_FLUX]"
            file.write(config_item_description("cf_flux", self.cf_flux, desc))

            desc = "Column name for the flux errors [examples: SAP_FLUX_ERR, PDCSAP_FLUX_ERR, KSPSAP_FLUX_ERR]"
            file.write(config_item_description("cf_flux_err", self.cf_flux_err, desc))

            desc = "Column name for the flux quality flags"
            file.write(config_item_description("cf_quality", self.cf_quality, desc))

            desc = "Apply the quality flags supplied by the data source"
            file.write(config_item_description("apply_q_flags", self.apply_q_flags, desc))

            desc = "Cut the time chunks in half (TESS data often has a discontinuity mid-sector)"
            file.write(config_item_description("halve_chunks", self.halve_chunks, desc))

            file.write("#" * line_width + "\n")
            file.write(fill_header_str("GUI settings", line_width, fill_value='-', end='\n'))

            desc = "Dark mode. [WIP]"
            file.write(config_item_description("dark_mode", self.dark_mode, desc))

            desc = "Horizontal window size as a fraction of the screen width"
            file.write(config_item_description("h_size_frac", self.h_size_frac, desc))

            desc = "Vertical window size as a fraction of the screen height"
            file.write(config_item_description("v_size_frac", self.v_size_frac, desc))

            file.write("#" * line_width + "\n")

        return None


def fill_header_str(header, line_width, fill_value='-', end='\n'):
    """Fill a header string on eiter side with hyphens.

    Parameters
    ----------
    header: str
        Text to put in the centre of the header.
    line_width: int
        Width of the final string, excluding `end`.
    fill_value: str
        Value used to fill up the line.
    end: str
        Something to end the line.

    Returns
    -------
    str
        Filled header string.
    """
    # add spaces to the header
    header = " " + header + " "

    # calculate the space to fill on either side
    left = (line_width - len(header)) // 2
    right = line_width - left - len(header)

    header = "# " + fill_value * (left - 2) + header + fill_value * right + end

    return header


def config_item_description(item, value, description):
    """Return a description string for a setting in the config file.

    Parameters
    ----------
    item: str
        The name of the configuration item.
    value: any
        The value of the configuration item.
    description: str
        A description of what the configuration item does.

    Returns
    -------
    str
        A formatted string that includes the item, its value, and a description.
    """
    # start with standardised bit
    item_desc = f"# {item} description:\n"

    # add the description
    item_desc += f"# {description}\n"

    # if item is already a string, add ''
    if type(value) is str:
        value = "'" + value + "'"

    # finally add the item
    item_desc += f"{item}: {value}\n\n"  # empty line at the end

    return item_desc


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
