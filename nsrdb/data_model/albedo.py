# -*- coding: utf-8 -*-
"""A framework for handling Albedo data.

Current framework will extract albedo data from a directory of daily 2018
albedo files that are a combination of 1km MODIS (8day) with
1km IMS snow (daily).
"""

import numpy as np
import os
import h5py
import logging

from nsrdb.data_model.base_handler import AncillaryVarHandler


logger = logging.getLogger(__name__)


class AlbedoVar(AncillaryVarHandler):
    """Framework for Albedo data extraction."""

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

        super().__init__(var_meta, name, date)

    @property
    def date_stamp(self):
        """Get the Albedo datestamp corresponding to the specified date

        Returns
        -------
        date : str
            Date stamp that should be in the NSRDB Albedo file,
            format is DDD_YYYY where DDD is the zero-indexed day of year.
        """

        # day index is zero-indexed
        d_i = str(self._date.timetuple().tm_yday - 1).zfill(3)
        y = str(self._date.year)
        date = '{d_i}_{y}'.format(d_i=d_i, y=y)
        return date

    @property
    def file(self):
        """Get the Albedo file path for the target NSRDB date.

        Returns
        -------
        falbedo : str
            NSRDB Albedo file path.
        """

        flist = os.listdir(self.source_dir)
        for f in flist:
            if self.date_stamp in f:
                falbedo = os.path.join(self.source_dir, f)
                break
        return falbedo

    @property
    def source_data(self):
        """Get single day data from the Albedo source file.

        Returns
        -------
        data : np.ndarray
            2D spatially gridded numpy array (lon X lat) for a single day
            of albedo data.
        """

        # open h5py NSRDB albedo file
        with h5py.File(self.file, 'r') as f:
            attrs = dict(f['surface_albedo'].attrs)
            scale = attrs.get('scale_factor', 1)
            data = f['surface_albedo'][...].astype(np.float32)
            data /= scale

        return data

    @property
    def grid(self):
        """Return the Albedo source coordinates.

        Returns
        -------
        self._albedo_grid : dict
            Albedo grid data. The albedo grid (from MODIS) is an ordered
            lat-lon grid so this dict has two entries 'latitude' and
            'longitude' with 1D arrays for each.
        """

        if not hasattr(self, '_albedo_grid'):
            self._albedo_grid = {}

            with h5py.File(self.file, 'r') as f:
                self._albedo_grid['latitude'] = f['latitude'][...]
                self._albedo_grid['longitude'] = f['longitude'][...]

        return self._albedo_grid
