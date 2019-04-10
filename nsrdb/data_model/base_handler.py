# -*- coding: utf-8 -*-
"""Base handler class for NSRDB data sources."""

import pandas as pd
import logging
from warnings import warn

from nsrdb import DATADIR


logger = logging.getLogger(__name__)


class AncillaryVarHandler:
    """Base class for ancillary variable source data handling."""

    # default source data directory
    DEFAULT_DIR = DATADIR

    def __init__(self, var_meta, name, date):
        """
        Parameters
        ----------
        var_meta : str | pd.DataFrame
            CSV file or dataframe containing meta data for all NSRDB variables.
        name : str
            NSRDB var name.
        date : datetime.date
            Single day to extract data for.
        """
        self._var_meta = self._parse_var_meta(var_meta)
        self._name = name
        self._date = date

    @staticmethod
    def _parse_var_meta(inp):
        """Set the meta data for NSRDB variables.

        Parameters
        ----------
        inp : str
            CSV file containing meta data for all NSRDB variables.
        """
        var_meta = None
        if isinstance(inp, str):
            if inp.endswith('.csv'):
                var_meta = pd.read_csv(inp)
        elif isinstance(inp, pd.DataFrame):
            var_meta = inp

        if var_meta is None:
            raise TypeError('Could not parse meta data for NSRDB variables '
                            'from: {}'.format(inp))
        return var_meta

    @property
    def var_meta(self):
        """Return the meta data for NSRDB variables.

        Returns
        -------
        _var_meta : pd.DataFrame
            Meta data for NSRDB variables.
        """
        return self._var_meta

    @property
    def name(self):
        """Get the NSRDB variable name."""
        return self._name

    @property
    def mask(self):
        """Get a boolean mask to locate the current variable in the meta data.
        """
        if not hasattr(self, '_mask'):
            self._mask = self.var_meta['var'] == self._name
        return self._mask

    @property
    def elevation_correct(self):
        """Get the elevation correction preference.

        Returns
        -------
        elevation_correct : bool
            Whether or not to use elevation correction for the current var.
        """
        temp = self.var_meta.loc[self.mask, 'elevation_correct']
        return bool(temp.values[0])

    @property
    def spatial_method(self):
        """Get the spatial interpolation method.

        Returns
        -------
        spatial_method : str
            NN or IDW
        """
        return str(self.var_meta.loc[self.mask, 'spatial_interp'].values[0])

    @property
    def scale_factor(self):
        """Get the scale factor attribute.

        Returns
        -------
        scale_factor : float
            Factor to apply (multiply) before writing to disk.
        """
        return float(self.var_meta.loc[self.mask, 'scale_factor'].values[0])

    @property
    def dtype(self):
        """Get the data type attribute.

        Returns
        -------
        dtype : str
            Intended NSRDB disk data type.
        """
        return str(self.var_meta.loc[self.mask, 'final_dtype'].values[0])

    @property
    def units(self):
        """Get the units attribute.

        Returns
        -------
        units : str
            NSRDB variable units.
        """
        return str(self.var_meta.loc[self.mask, 'units'].values[0])

    @property
    def source_dir(self):
        """Get the source directory containing the variable data files.

        Returns
        -------
        source_dir : str
            Directory containing source data files (with possible sub folders).
        """
        d = self.var_meta.loc[self.mask, 'source_directory'].values[0]
        if not d:
            warn('Using default data directory for "{}"'.format(self.name))
            d = self.DEFAULT_DIR
        return str(d)

    @property
    def temporal_method(self):
        """Get the temporal interpolation method.

        Returns
        -------
        temporal_method : str
            linear or nearest
        """
        return str(self.var_meta.loc[self.mask, 'temporal_interp'].values[0])

    @property
    def dset(self):
        """Get the MERRA dset name from the NSRDB variable name.

        Returns
        -------
        dset : str
            MERRA dset name, e.g.:
                tavg1_2d_aer_Nx
                tavg1_2d_ind_Nx
                tavg1_2d_rad_Nx
                tavg1_2d_slv_Nx
        """
        return str(self.var_meta.loc[self.mask, 'merra_dset'].values[0])

    @property
    def units(self):
        """Get the variable units.

        Returns
        -------
        units : str
            Units for the current variable.
        """
        return str(self.var_meta.loc[self.mask, 'units'].values[0])

    @property
    def final_dtype(self):
        """Get the variable's intended storage datatype.

        Returns
        -------
        dtype : str
            Data type for the current variable.
        """
        return str(self.var_meta.loc[self.mask, 'final_dtype'].values[0])

    @property
    def scale_factor(self):
        """Get the variable's intended storage scale factor.

        Returns
        -------
        scale_factor : float
            Scale factor for the current variable. Data is multiplied by this
            scale factor before being stored.
        """
        return float(self.var_meta.loc[self.mask, 'scale_factor'].values[0])

    @staticmethod
    def _get_time_index(date, freq='1h'):
        """Get a pandas date time object for the given analysis date.

        Parameters
        ----------
        date : datetime.date
            Single day to get time index for.
        freq : str
            Pandas datetime frequency, e.g. '1h', '5min', etc...

        Returns
        -------
        ti : pd.DatetimeIndex
            Pandas datetime index for the current day.
        """

        ti = pd.date_range('1-1-{y}'.format(y=date.year),
                           '1-1-{y}'.format(y=date.year + 1),
                           freq=freq)[:-1]
        mask = (ti.month == date.month) & (ti.day == date.day)
        ti = ti[mask]
        return ti
