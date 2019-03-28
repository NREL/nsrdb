# -*- coding: utf-8 -*-
"""NSRDB processing methods.
@author: gbuster
"""
import os

__version__ = "3.1.0"

NSRDBDIR = os.path.dirname(os.path.realpath(__file__))
DATADIR = os.path.join(os.path.dirname(NSRDBDIR), 'data')
CONFIGDIR = os.path.join(os.path.dirname(NSRDBDIR), 'config')
TESTDATADIR = os.path.join(os.path.dirname(NSRDBDIR), 'tests', 'data')
