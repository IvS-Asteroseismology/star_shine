"""STAR SHINE __init__ file

Code written by: Luc IJspeert
"""

from .api.main import *
from .api.data import Data
from .api.result import Result
from .api.pipeline import Pipeline

try:
    # GUI
    from .gui.gui_app import launch_gui
except ImportError:
    print('GUI unavailable, likely missing dependency PySide6.')
    pass

__all__ = ['gui', 'api', 'core', 'config', 'data']
