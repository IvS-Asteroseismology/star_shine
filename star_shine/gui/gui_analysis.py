"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains the analysis functions for the graphical user interface.

Code written by: Luc IJspeert
"""
from PySide6.QtCore import QThread, Signal


class PipelineThread(QThread):
    """A QThread subclass to perform analysis in the background."""

    def __init__(self, pipeline_instance):
        super().__init__()
        self.pipeline_instance = pipeline_instance
        self.func_name = ''
        self.args = ()
        self.kwargs = {}

    def start_function(self, func_name, *args, **kwargs):
        """Start the thread with a specific function and arguments."""
        self.func_name = func_name
        self.args = args
        self.kwargs = kwargs
        super().start()

    def run(self):
        """Run the function in a separate thread."""
        if self.func_name != '':
            # try:
            # start a thread with the function
            function_to_run = getattr(self.pipeline_instance, self.func_name)
            function_to_run(*self.args, **self.kwargs)

            # except Exception as e:
            #     self.pipeline_instance.logger.error(f"Error during analysis: {e}")

    def stop(self):
        """Stop the thread."""
        pass
