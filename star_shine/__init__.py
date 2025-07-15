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
except ImportError as e:
    print(e)
    print('GUI unavailable, likely missing dependency PySide6.')
    pass

__pdoc__ = {"star_shine.gui": False}

__all__ = ['gui', 'api', 'core', 'config', 'data']
