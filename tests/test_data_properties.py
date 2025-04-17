import unittest
from unittest.mock import patch
import numpy as np

from star_shine.config import data_properties as dp


class TestDataProperties(unittest.TestCase):
    def setUp(self):
        """Set up test cases with different types of time series data:
        - Regular time series without gaps.
        - Time series with some random noise.
        - Time series with gaps longer than 27 days.
        """
        np.random.seed(42)  # fix randomness

        # Regular time series without gaps
        self.time_series_regular = np.arange(0, 50, 0.1)
        self.adjusted_time =np.copy(self.time_series_regular)
        self.adjusted_time[250:] += 0.01

        # Time series with some random noise
        self.time_series_noisy = np.arange(0, 50, 0.1) + 0.001 * np.random.rand(500)

        # Time series with gaps longer than 27 days
        self.time_series_gap = np.copy(self.time_series_noisy)
        self.time_series_gap[250:] += 30

    def test_signal_to_noise_threshold(self):
        """Test the signal-to-noise threshold calculation for a regular time series without gaps."""
        # Calculate SNR threshold without gaps
        fixed_snr_thr = -1
        mock_config_instance = dp.config
        mock_config_instance.snr_thr = fixed_snr_thr

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            snr_thr = dp.signal_to_noise_threshold(self.time_series_noisy)

        # Expected value using the given formula
        expected = 4.45

        self.assertAlmostEqual(snr_thr, expected)

    def test_signal_to_noise_threshold_gap(self):
        """Test the signal-to-noise threshold calculation for a regular time series with gaps."""
        # Calculate SNR threshold without gaps
        fixed_snr_thr = -1
        mock_config_instance = dp.config
        mock_config_instance.snr_thr = fixed_snr_thr

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            snr_thr = dp.signal_to_noise_threshold(self.time_series_gap)

        # Expected value using the given formula
        expected = 4.70

        self.assertAlmostEqual(snr_thr, expected)

    def test_signal_to_noise_threshold_user_defined(self):
        """Test the signal-to-noise threshold calculation when a user-defined value is provided."""
        fixed_snr_thr = 4.0
        mock_config_instance = dp.config
        mock_config_instance.snr_thr = fixed_snr_thr

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            snr_thr = dp.signal_to_noise_threshold(self.time_series_noisy)

        # Expected value is the user-defined value
        self.assertEqual(snr_thr, fixed_snr_thr)

    def test_frequency_resolution(self):
        """Test the frequency resolution calculation for a regularly spaced time series."""
        # Calculate frequency resolution for noisy time series
        factor = 1.5
        f_res = dp.frequency_resolution(self.time_series_regular, factor=factor)

        # Expected value using the given formula with factor=1
        expected = 0.03006012

        self.assertAlmostEqual(f_res, expected)

    def test_nyquist_sum_koen_2006_zero(self):
        """Test the Nyquist sum calculation for a regular time series using Koen (2006) formula."""
        # Calculate frequency resolution
        delta_t_min = np.min(self.time_series_regular[1:] - self.time_series_regular[:-1])

        # Calculate Nyquist sum
        nyquist_sum = dp.nyquist_sum_koen_2006(1, self.time_series_regular, delta_t_min)

        # Expected value using the given formula
        expected = 0

        self.assertAlmostEqual(nyquist_sum, expected)

    def test_nyquist_sum_koen_2006_nonzero(self):
        """Test the Nyquist sum calculation for a regular time series using Koen (2006) formula."""
        # Calculate frequency resolution
        delta_t_min = np.min(self.time_series_regular[1:] - self.time_series_regular[:-1])

        # Calculate Nyquist sum
        nyquist_sum = dp.nyquist_sum_koen_2006(1, self.time_series_regular, 3 * delta_t_min)

        # Expected value using the given formula
        expected = 0

        self.assertGreater(nyquist_sum, expected)

    def test_nyquist_frequency_simple_regular(self):
        """Test the Nyquist frequency calculation for a regular time series."""
        # Calculate Nyquist frequency for regular time series
        method_nyquist = 'simple'
        mock_config_instance = dp.config
        mock_config_instance.nyquist_method = method_nyquist

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            f_nyquist = dp.nyquist_frequency(self.time_series_regular)

        # Expected value using the given formula
        delta_t_min = np.min(self.time_series_regular[1:] - self.time_series_regular[:-1])
        expected = 0.5 / delta_t_min

        self.assertAlmostEqual(f_nyquist, expected)

    def test_nyquist_frequency_simple_noisy(self):
        """Test the Nyquist frequency calculation for a noisy time series."""
        # Calculate Nyquist frequency for regular time series
        method_nyquist = 'simple'
        mock_config_instance = dp.config
        mock_config_instance.nyquist_method = method_nyquist

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            f_nyquist = dp.nyquist_frequency(self.time_series_noisy)

        # Expected value using the given formula
        delta_t_min = np.min(self.time_series_noisy[1:] - self.time_series_noisy[:-1])
        expected = 0.5 / delta_t_min

        self.assertAlmostEqual(f_nyquist, expected)

    def test_nyquist_frequency_rigorous_regular(self):
        """Test the Nyquist frequency calculation for a regular time series."""
        # Calculate Nyquist frequency for regular time series
        method_nyquist = 'rigorous'
        mock_config_instance = dp.config
        mock_config_instance.nyquist_method = method_nyquist

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            f_nyquist = dp.nyquist_frequency(self.time_series_regular)

        # Expected value using the given formula
        delta_t_min = np.min(self.time_series_regular[1:] - self.time_series_regular[:-1])
        expected = 0.5 / delta_t_min

        self.assertAlmostEqual(f_nyquist, expected)

    def test_nyquist_frequency_rigorous_regular_adjusted(self):
        """Test the Nyquist frequency calculation for a regular time series."""
        # Calculate Nyquist frequency for regular time series
        method_nyquist = 'rigorous'
        mock_config_instance = dp.config
        mock_config_instance.nyquist_method = method_nyquist

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            f_nyquist = dp.nyquist_frequency(self.adjusted_time)

        # Expected value using the given formula
        delta_t_min = np.min(self.time_series_regular[1:] - self.time_series_regular[:-1])
        expected = 10 * 0.5 / delta_t_min

        self.assertAlmostEqual(f_nyquist, expected)

    def test_nyquist_frequency_custom(self):
        """Test the Nyquist frequency calculation for a noisy time series."""
        # Calculate Nyquist frequency for regular time series
        method_nyquist = 'custom'
        custom_nyquist = 50.
        mock_config_instance = dp.config
        mock_config_instance.nyquist_method = method_nyquist
        mock_config_instance.nyquist_value = custom_nyquist

        with patch('star_shine.config.data_properties.get_config', return_value=mock_config_instance):
            f_nyquist = dp.nyquist_frequency(self.time_series_noisy)

        self.assertAlmostEqual(f_nyquist, custom_nyquist)


if __name__ == '__main__':
    unittest.main()