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
        self._lon_good = None
        self._lat_good = None
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

    def set_exclusions(self, lat_in=None, lat_ex=None, lon_in=None,
                       lon_ex=None):
        """Set latitude/longitude inclusions/exclusions to minimize data read.

        Only one inclusion or exclusion is used for each latitude or longitude.
        The protected grid data attribute will be modified based on the inputs.

        Parameters
        ----------
        lat_in : None | tuple
            Latitude range to include (everything outside range is excluded).
        lat_ex : None | tuple
            Latitude range to exclude (everything inside range is excluded).
        lon_in : None | tuple
            Longitude range to include (everything outside range is excluded).
        lon_ex : None | tuple
            Longitude range to exclude (everything inside range is excluded).
        """

        # find the good latitude indices if requested
        if lat_in is not None:
            # find coordinates greater than min, less than max
            self._lat_good = np.where(
                self.grid['latitude'] > np.min(lat_in) &
                self.grid['latitude'] < np.max(lat_in))[0]
        elif lat_ex is not None:
            # find coordinates less than min, greater than max
            self._lat_good = np.where(
                self.grid['latitude'] < np.min(lat_in) &
                self.grid['latitude'] > np.max(lat_in))[0]
        else:
            # no inclusion/exclusion requested, all data will be pulled
            self._lat_good = None

        # find the good longitude indices if requested
        if lon_in is not None:
            # find coordinates greater than min, less than max
            self._lon_good = np.where(
                self.grid['longitude'] > np.min(lon_in) &
                self.grid['longitude'] < np.max(lon_in))[0]
        elif lon_ex is not None:
            # find coordinates less than min, greater than max
            self._lon_good = np.where(
                self.grid['longitude'] < np.min(lon_in) &
                self.grid['longitude'] > np.max(lon_in))[0]
        else:
            # no inclusion/exclusion requested, all data will be pulled
            self._lon_good = None

        # reduce the protected albedo grid attribute based on
        # these inclusions/exclusions
        if self._lat_good is not None:
            self._albedo_grid['latitude'] = \
                self._albedo_grid['latitude'][self._lat_good]
        if self._lon_good is not None:
            self._albedo_grid['longitude'] = \
                self._albedo_grid['longitude'][self._lon_good]

    @property
    def source_data(self):
        """Get single day data from the Albedo source file.

        Returns
        -------
        data : np.ndarray
            Flattened albedo data. Note that the data originates as a 2D
            spatially gridded numpy array with shape (lat x lon).
        """

        # open h5py NSRDB albedo file
        with h5py.File(self.file, 'r') as f:
            attrs = dict(f['surface_albedo'].attrs)
            scale = attrs.get('scale_factor', 1)

            if self._lat_good is None and self._lon_good is None:
                data = f['surface_albedo'][...]
            elif self._lat_good is not None and self._lon_good is not None:
                data = f['surface_albedo'][self._lat_good, self._lon_good]
            elif self._lat_good is not None:
                data = f['surface_albedo'][self._lat_good, :]
            elif self._lon_good is not None:
                data = f['surface_albedo'][:, self._lon_good]

            data = data.ravel()
            data = data.astype(np.float32)
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
