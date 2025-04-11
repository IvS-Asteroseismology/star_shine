"""STAR SHINE __init__ file

Code written by: Luc IJspeert
"""

from star_shine.api.main import *
from .core import mcmc_functions as mcf, utility as ut, visualisation as vis, analysis_functions as af, \
    timeseries_fitting as tsfit, timeseries_functions as tsf
from .api import main
from star_shine.core.utility import update_config
