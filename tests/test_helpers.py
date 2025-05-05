import unittest
from unittest.mock import patch, mock_open, MagicMock
import os
import logging
import tomllib
import importlib.metadata
import importlib.resources

from star_shine.config.helpers import (get_version, get_config, get_config_path, get_mpl_stylesheet_path,
                                       get_custom_logger, get_images_path)


class TestHelpers(unittest.TestCase):
    def setUp(self):
        """Setup method to initialize any necessary mocks or state."""
        pass

    @patch('importlib.metadata.version')
    @patch('tomllib.load', return_value={'project': {'version': '1.2.3'}})
    def test_get_version_metadata_success(self, mock_file, mock_metadata):
        """Test get_version function when metadata is available."""
        mock_metadata.return_value = '0.1.0'
        version = get_version()

        # check obtained version
        self.assertEqual(version, '0.1.0')

        # check that metadata was called and the file not called
        mock_metadata.assert_called_once_with('star_shine')
        mock_file.assert_not_called()

    @patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError)
    @patch('tomllib.load', return_value={'project': {'version': '1.2.3'}})
    def test_get_version_fallback_success(self, mock_file, mock_metadata):
        """Test get_version function when falling back to pyproject.toml."""
        version = get_version()

        # check obtained version
        self.assertEqual(version, '1.2.3')

        # check that metadata was called and the file was called
        mock_metadata.assert_called_once_with('star_shine')
        mock_file.assert_called_once()

    @patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError)
    @patch('tomllib.load', side_effect=tomllib.TOMLDecodeError("Failed to decode TOML"))
    @patch('builtins.open', new_callable=mock_open)
    def test_get_version_file_decode_error(self, mock_file, mock_toml_load, mock_metadata):
        """Test get_version function when pyproject.toml contains invalid TOML."""
        with self.assertRaises(FileNotFoundError) as context:
            get_version()

        # check the returned error
        self.assertIn("Could not find or parse version in pyproject.toml", str(context.exception))

        # check that metadata was called and the file was called
        mock_metadata.assert_called_once_with('star_shine')
        mock_file.assert_called_once()

    @patch('importlib.metadata.version', side_effect=importlib.metadata.PackageNotFoundError)
    @patch('tomllib.load', return_value={'project': {'version': '1.2.3'}})
    @patch('builtins.open', new_callable=mock_open)
    def test_get_version_file_not_found(self, mock_file, mock_toml_load, mock_metadata):
        """Test get_version function when pyproject.toml is not found."""
        # Set the side_effect on mock_file to simulate FileNotFoundError
        mock_file.side_effect = FileNotFoundError("No such file or directory")

        with self.assertRaises(FileNotFoundError) as context:
            get_version()

        # check the returned error
        self.assertIn("Could not find or parse version in pyproject.toml", str(context.exception))

        # check that metadata was called and the file was called
        mock_metadata.assert_called_once_with('star_shine')
        mock_file.assert_called_once()

    @patch('star_shine.config.config.get_config', return_value=MagicMock())
    def test_get_config(self, mock_get_config):
        """Test get_config function."""
        config = get_config()

        # check that config is no longer None
        self.assertIsNotNone(config)
        mock_get_config.assert_called_once()

    @patch('star_shine.config.config.get_config_path', return_value='path/to/config.yaml')
    def test_get_config_path(self, mock_get_config_path):
        """Test get_config_path function."""
        path = get_config_path()

        # check returned path
        self.assertEqual(path, 'path/to/config.yaml')
        mock_get_config_path.assert_called_once()

    @patch('importlib.resources.files', return_value=MagicMock(joinpath=MagicMock(return_value='path/to/images')))
    def test_get_images_path(self, mock_images_path):
        """Test get_images_path function."""
        path = get_images_path()

        # check returned path
        self.assertEqual(path, 'path/to/images')
        mock_images_path.assert_called_once_with('star_shine.data')

    @patch('importlib.resources.files',
           return_value=MagicMock(joinpath=MagicMock(return_value='path/to/mpl_stylesheet.dat')))
    def test_get_mpl_stylesheet_path(self, mock_importlib_resources):
        """Test get_mpl_stylesheet_path function."""
        path = get_mpl_stylesheet_path()

        # check returned path
        self.assertEqual(path, 'path/to/mpl_stylesheet.dat')
        mock_importlib_resources.assert_called_once_with('star_shine.config')

    @patch('builtins.open', new_callable=mock_open)
    def test_get_custom_logger(self, mock_open):
        """Test get_custom_logger function."""
        save_dir = 'test_save'
        target_id = 'test_target'
        verbose = True

        logger = get_custom_logger(target_id, save_dir, verbose)

        # Verify that handlers are added to the logger
        self.assertEqual(len(logger.handlers), 2)
        self.assertIsInstance(logger.handlers[0], logging.StreamHandler)
        self.assertIsInstance(logger.handlers[1], logging.FileHandler)

        # Ensure open is called
        mock_open.assert_called_once()


if __name__ == '__main__':
    unittest.main()