"""STAR SHINE __init__ file

Code written by: Luc IJspeert
"""

import star_shine.api

try:
    # GUI
    from star_shine.gui.gui_app import launch_gui
except ImportError:
    print('GUI unavailable, likely missing dependency PySide6.')
    pass

__all__ = ['gui', 'api', 'core', 'config', 'data']
