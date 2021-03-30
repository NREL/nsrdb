# -*- coding: utf-8 -*-
"""
Classes to handle NSRSDB h5 output files.
"""
from reV.handlers.outputs import Outputs as RevOutputs

from nsrdb.version import __version__


class Outputs(RevOutputs):
    """
    Base class to handle NSRDB output data in .h5 format
    """

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
