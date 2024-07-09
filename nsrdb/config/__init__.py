"""Module with config templates and standard config file creation functions."""

import os

from nsrdb import CONFIGDIR

PRE2018_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_nsrdb_pre2018.json'
)
POST2017_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_nsrdb_post2017.json'
)
PIPELINE_CONFIG_TEMPLATE = os.path.join(
    CONFIGDIR, 'templates/config_pipeline.json'
)
