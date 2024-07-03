"""NSRDB processing methods.
@author: gbuster
"""

import os

import pandas as pd

from ._version import __version__

NSRDBDIR = os.path.dirname(os.path.realpath(__file__))
CONFIGDIR = os.path.join(NSRDBDIR, 'config')
DATADIR = os.path.join(NSRDBDIR, 'data_model', 'data')
DEFAULT_VAR_META = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
TESTDATADIR = os.path.join(os.path.dirname(NSRDBDIR), 'tests', 'data')
VAR_DESCRIPTIONS = pd.read_csv(os.path.join(CONFIGDIR, 'var_descriptions.csv'))

# This needs to go last because it depends on the global dirs above
from nsrdb.nsrdb import NSRDB  # noqa: E402
