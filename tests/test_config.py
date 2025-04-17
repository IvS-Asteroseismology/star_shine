import unittest
from star_shine.config.config import Config, get_config

class TestConfig(unittest.TestCase):
    def setUp(self):
        """Setup method to initialize the config instance"""
        self.config_path = 'star_shine/data/config.yaml'

    def test_validate_config_valid_data(self):
        """Test with valid config data taken directly from the class"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}
        try:
            Config._validate_config(valid_config)
        except ValueError as e:
            self.fail(f"Validation failed for valid configuration: {e}")

    def test_validate_config_missing_keys(self):
        """Test with config data that is missing keys"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}

        # copy and remove two items
        incomplete_config = valid_config.copy()
        incomplete_config.popitem()
        incomplete_config.popitem()

        with self.assertRaises(ValueError) as context:
            Config._validate_config(incomplete_config)

        self.assertIn("Missing keys in configuration file", str(context.exception))

    def test_validate_config_excess_keys(self):
        """Test with config data that has excess keys"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}

        # copy and add two items
        excess_config = valid_config.copy()
        excess_config['extra_key1'] = 'extra_value1'
        excess_config['extra_key2'] = 'extra_value2'

        with self.assertRaises(ValueError) as context:
            Config._validate_config(excess_config)

        self.assertIn("Excess keys in configuration file", str(context.exception))

    def test_config_loads(self):
        """Check if the configuration loads without errors"""
        try:
            Config()
        except Exception as e:
            self.fail(f"Configuration failed to load: {e}")

    def test_config_getter(self):
        """Check if the configuration loads without errors"""
        try:
            config = get_config()
        except Exception as e:
            self.fail(f"Configuration failed to be gotten: {e}")

        self.assertIsInstance(config, Config)

    def test_config_updates_from_file(self):
        """Check if the configuration updates from file without errors"""
        try:
            config = Config()
            config.update_from_file(self.config_path)
        except Exception as e:
            self.fail(f"Configuration failed to update from file: {e}")

    def test_config_updates_from_dict(self):
        """Check if the configuration updates from dictionary without errors"""
        valid_config = {key: getattr(Config, key) for key in dir(Config)
                        if not callable(getattr(Config, key)) and not key.startswith('_')}

        # copy and remove two items
        incomplete_but_valid_config = valid_config.copy()
        incomplete_but_valid_config.popitem()
        incomplete_but_valid_config.popitem()

        try:
            config = Config()
            config.update_from_dict(incomplete_but_valid_config)
        except Exception as e:
            self.fail(f"Configuration failed to update from dictionary: {e}")

if __name__ == '__main__':
    unittest.main()