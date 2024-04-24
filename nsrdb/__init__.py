# -*- coding: utf-8 -*-
"""
NSRDB processing methods.
@author: gbuster
"""
import os

from ._version import __version__

NSRDBDIR = os.path.dirname(os.path.realpath(__file__))
CONFIGDIR = os.path.join(NSRDBDIR, 'config')
DATADIR = os.path.join(NSRDBDIR, 'data_model', 'data')
DEFAULT_VAR_META = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')
TESTDATADIR = os.path.join(os.path.dirname(NSRDBDIR), 'tests', 'data')

# This needs to go last because it depends on the global dirs above
from nsrdb.nsrdb import NSRDB
