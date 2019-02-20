# -*- coding: utf-8 -*-
"""NSRDB all-sky module.
"""

import os
import pandas as pd

__version__ = "1.0.0"
__dir__ = os.path.dirname(os.path.realpath(__file__))
__testdatadir__ = os.path.join(os.path.dirname(__dir__), 'tests', 'data')

RADIUS = pd.read_csv(
    os.path.join(__dir__, 'sun_earth_radius_vector.csv')).set_index('doy')

# Constant clear/cloudy integer labels for use of AllSky
CLEAR_TYPES = (0, 1, 11, 12)
WATER_TYPES = (2, 3, 4, 5, 10)
ICE_TYPES = (6, 7, 8, 9)
CLOUD_TYPES = WATER_TYPES + ICE_TYPES

# Solar constant global variable. Flux density value measuring mean solar
# electromagnetic radiation per unit area, default is 1361.2 (W/m2).
SOLAR_CONSTANT = 1361.2
