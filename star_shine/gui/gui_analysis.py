"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the analysis functions for the graphical user interface.

Code written by: Luc IJspeert
"""

from PySide6.QtCore import QThread, Signal


class PipelineThread(QThread):
    """A QThread subclass to perform analysis in the background."""

    def __init__(self, pipeline_instance, result_signal):
        super().__init__()
        self.pipeline_instance = pipeline_instance

        # signal to emit results to
        self.result_signal = result_signal

    def run(self):
        """Run the analysis pipeline in a separate thread."""
        # Perform analysis using your Pipeline class
        try:
            result = self.pipeline_instance.run()

            # Emit the signal with the result instance
            self.result_signal.emit(result)

        except Exception as e:
            self.pipeline_instance.logger.error(f"Error during analysis: {e}")
