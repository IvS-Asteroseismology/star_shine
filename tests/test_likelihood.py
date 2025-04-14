import unittest
import numpy as np

from star_shine.core import timeseries as tsf


class TestLikelihood(unittest.TestCase):

    def setUp(self):
        """Setup common test inputs before each test."""
        self.time = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.flux = np.array([11.0, 9.5, 10.2, 9.9, 10.05, 10.1, 9.95, 10.2, 9.9, 10.5])
        self.residual = np.array([1.0, -0.5, 0.2, -0.1, 0.05, 0.1, -0.05, 0.2, -0.1, 0.5])
        self.flux_err = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Constant error
        self.data = {'time': self.time, 'flux': self.flux, 'residual': self.residual, 'flux_err': self.flux_err}

    def test_likelihood_without_errors(self):
        """Test that likelihood runs and returns a finite number."""
        likelihood = tsf.calc_likelihood(**self.data)
        self.assertTrue(np.isfinite(likelihood), "Likelihood should be finite")

    def test_likelihood_with_errors(self):
        """Test that likelihood with measurement errors also returns a finite number."""
        likelihood = tsf.calc_likelihood(**self.data)
        self.assertTrue(np.isfinite(likelihood), "Likelihood with errors should be finite")

    def test_single_element_input(self):
        """Test that the function works correctly with a single element in the input arrays."""
        likelihood = tsf.calc_likelihood(time=np.array([0]), flux=np.array([10.0]), residual=np.array([1.0]),
                                         flux_err=np.array([0.1]))
        self.assertTrue(np.isfinite(likelihood), "Likelihood should be finite with single element inputs")

    def test_likelihood_deterministic_output(self):
        """Ensure function returns the same result for same input."""
        like1 = tsf.calc_likelihood(**self.data)
        like2 = tsf.calc_likelihood(**self.data)
        self.assertAlmostEqual(like1, like2, places=8, msg="Likelihood should be deterministic")

    def test_likelihood_changes_with_data(self):
        """Ensure likelihood changes when residuals change."""
        like_original = tsf.calc_likelihood(**self.data)
        modified_residuals = self.residual + 0.1  # Slightly alter residuals
        like_modified = tsf.calc_likelihood(time=self.time, residual=modified_residuals, flux_err=self.flux_err)
        self.assertNotEqual(like_original, like_modified, "Likelihood should change when residuals change")

    def test_none_input(self):
        """Test that the function handles None input gracefully."""
        with self.assertRaises(ValueError) as context:
            tsf.calc_likelihood(None, None, None, None)
        self.assertIn("Relevant input arrays must not be None", str(context.exception))

    def test_empty_input(self):
        """Test that the function handles empty input arrays gracefully."""
        with self.assertRaises(ValueError) as context:
            tsf.calc_likelihood(np.array([]), np.array([]), np.array([]), np.array([]))
        self.assertIn("Relevant input arrays must not be empty", str(context.exception))

    # def test_invalid_input_shape(self):
    #     """Test that the function raises an error when input arrays have mismatched shapes."""
    #     with self.assertRaises(ValueError) as context:
    #         tsf.calc_likelihood(time=self.time, residual=np.array([1.0]))
    #     self.assertIn("Relevant input arrays must have the same shape", str(context.exception))

if __name__ == '__main__':
    unittest.main()
