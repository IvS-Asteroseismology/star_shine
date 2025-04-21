"""STAR SHINE
Satellite Time-series Analysis Routine using Sinusoids and Harmonics through Iterative Non-linear Extraction

This Python module contains logging and text output functions for the graphical user interface.

Code written by: Luc IJspeert
"""
import logging
from PySide6.QtGui import QTextCursor

from star_shine.config import helpers as hlp


class QTextEditLogger(logging.Handler):
    """A custom logging handler that writes log messages to a QTextEdit widget."""
    def __init__(self, text_edit):
        """Initialise custom logging handler.

        Parameters
        ----------
        text_edit : QTextEdit
            The QTextEdit widget where log messages will be displayed.
        """
        super().__init__()
        self.text_edit = text_edit

    def emit(self, record):
        """Emit a log message to the QTextEdit widget.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted and emitted.
        """
        msg = self.format(record)
        cursor = self.text_edit.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(msg + '\n')
        self.text_edit.setTextCursor(cursor)
        self.text_edit.ensureCursorVisible()


def get_custom_gui_logger(text_edit, save_dir, target_id):
    """Create a custom logger for logging to file and to the gui.

    Parameters
    ----------
    text_edit: QTextEdit
        The QTextEdit widget where log messages will be displayed.
    save_dir: str
        folder to save the log file
    target_id: str
        Identifier to use for the log file

    Returns
    -------
    logging.Logger
        Customised logger object
    """
    # get the normal non-verbose logger
    logger = hlp.get_custom_logger(save_dir, target_id, verbose=False)

    # add a different stream handler
    qtext_edit_handler = QTextEditLogger(text_edit)
    qtext_edit_handler.setLevel(logging.EXTRA)  # print everything with level 15 or above
    s_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    qtext_edit_handler.setFormatter(s_format)
    logger.addHandler(qtext_edit_handler)

    return logger