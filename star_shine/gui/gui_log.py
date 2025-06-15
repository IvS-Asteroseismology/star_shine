"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains logging and text output functions for the graphical user interface.

Code written by: Luc IJspeert
"""
import logging

from PySide6.QtCore import QObject, Signal

from star_shine.config import helpers as hlp


class QLogEmitter(QObject):
    """A signal emitter helper class."""
    # Define a signal that emits the log message
    log_signal = Signal(str, bool)  # message, update flag

    def __init__(self):
        super().__init__()


class QSignalHandler(logging.Handler):
    """A custom logging handler that writes log messages to a QTextEdit widget."""
    # Define a signal that emits the log message
    log_signal = Signal(str)

    def __init__(self, emitter):
        """Initialise custom logging handler.

        Parameters
        ----------
        emitter: QLogEmitter
            Object that handles the signal emitting.
        """
        super().__init__()
        self.emitter = emitter  # instance of LogEmitter

    def emit(self, record):
        """Emit a log message to the QTextEdit widget.

        Parameters
        ----------
        record: logging.LogRecord
            The log record to be formatted and emitted.
        """
        msg = self.format(record)
        update = getattr(record, 'update', False)  # get the extra kwarg out of record
        self.emitter.log_signal.emit(msg, update)


def get_custom_gui_logger(target_id, save_dir):
    """Create a custom logger for logging to file and to the gui.

    Parameters
    ----------
    target_id: str
        Identifier to use for the log file.
    save_dir: str
        folder to save the log file. If empty, no saving happens.

    Returns
    -------
    logging.Logger
        Customised logger object
    """
    # get the normal non-verbose logger
    logger = hlp.get_custom_logger(target_id, save_dir=save_dir, verbose=False)

    # add a different stream handler
    qlog_emitter = QLogEmitter()
    qsignal_handler = QSignalHandler(qlog_emitter)
    qsignal_handler.setLevel(logging.EXTRA)  # print everything with level 15 or above
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    qsignal_handler.setFormatter(s_format)
    logger.addHandler(qsignal_handler)

    # expose the log_signal to be connected
    logger.log_signal = qsignal_handler.emitter.log_signal

    return logger
