"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the analysis functions for the graphical user interface.

Code written by: Luc IJspeert
"""

from PySide6.QtCore import QThread, Signal


class PipelineThread(QThread):
    """A QThread subclass to perform analysis in the background."""
    # Define a signal that emits the log message
    result_signal = Signal()

    def __init__(self, pipeline_instance):
        super().__init__()
        self.pipeline_instance = pipeline_instance

    def extract_approx(self, f_approx):
        """Run extract_approx in a separate thread."""
        # Perform analysis using your Pipeline class
        try:
            self.pipeline_instance.extract_approx(f_approx)

            # Emit the signal
            self.result_signal.emit()

        except Exception as e:
            self.pipeline_instance.logger.error(f"Error from extract_approx: {e}")

    def remove_approx(self, f_approx):
        """Run remove_approx in a separate thread."""
        # Perform analysis using your Pipeline class
        try:
            self.pipeline_instance.remove_approx(f_approx)

            # Emit the signal
            self.result_signal.emit()

        except Exception as e:
            self.pipeline_instance.logger.error(f"Error from remove_approx: {e}")

    def iterative_prewhitening(self, n_extract=0):
        """Run iterative_prewhitening in a separate thread."""
        # Perform analysis using your Pipeline class
        try:
            self.pipeline_instance.iterative_prewhitening(n_extract=n_extract)

            # Emit the signal
            self.result_signal.emit()

        except Exception as e:
            self.pipeline_instance.logger.error(f"Error from iterative_prewhitening: {e}")

    def run(self):
        """Run the analysis pipeline in a separate thread."""
        # Perform analysis using your Pipeline class
        try:
            self.pipeline_instance.run()

            # Emit the signal
            self.result_signal.emit()

        except Exception as e:
            self.pipeline_instance.logger.error(f"Error during analysis: {e}")
