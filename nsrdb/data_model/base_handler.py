# -*- coding: utf-8 -*-
"""Base handler class for NSRDB data sources."""
from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from warnings import warn
import datetime as dt

from nsrdb import DATADIR, DEFAULT_VAR_META
from nsrdb.file_handlers.file_system import NSRDBFileSystem as NFS

logger = logging.getLogger(__name__)


class AncillaryVarHandler:
    """Base class for ancillary variable source data handling."""

    # default source data directory
    DEFAULT_DIR = DATADIR

    # nearest neighbor tree method for this variable
    NN_METHOD = 'haversine'

    def __init__(self, name, var_meta=None, date=None, **kwargs):
        """
        Parameters
        ----------
        name : str
            NSRDB var name.
        var_meta : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        date : datetime.date
            Single day to extract data for.
        kwargs : dict
            Optional kwargs to overwrite relevant data in the var_meta
        """

        self._var_meta = self._parse_var_meta(var_meta)
        self._name = name
        self._date = date
        self._cache_file = False
        self._mask = None

        # overwrite data in var_meta with the passed in kwargs
        for k, v in kwargs.items():
            if k in self._var_meta:
                self._var_meta.loc[self.mask, k] = v

        # legacy kwarg alias for source_directory
        sd = kwargs.get('source_dir', None)
        if sd:
            self._var_meta.loc[self.mask, 'source_directory'] = sd

    @staticmethod
    def _parse_var_meta(inp):
        """Set the meta data for NSRDB variables.

        Parameters
        ----------
        inp : str | pd.DataFrame | None
            CSV file or dataframe containing meta data for all NSRDB variables.
            Defaults to the NSRDB var meta csv in git repo.
        """

        # default to repo default
        if inp is None or str(inp).lower().strip() == 'none':
            inp = DEFAULT_VAR_META

        var_meta = None
        if isinstance(inp, str):
            if inp.endswith('.csv'):
                var_meta = pd.read_csv(inp)
        elif isinstance(inp, pd.DataFrame):
            var_meta = inp

        if var_meta is None:
            raise TypeError('Could not parse meta data for NSRDB variables '
                            'from: {}'.format(inp))

        var_meta['var'] = var_meta['var'].str.strip(' ')

        return var_meta

    @property
    def attrs(self):
        """Return a dictionary of dataset attributes for HDF5 dataset attrs.

        Returns
        -------
        attrs : dict
            Namespace of attributes to define the dataset.
        """

        attrs = dict({'units': self.units,
                      'scale_factor': self.scale_factor,
                      'physical_min': self.physical_min,
                      'physical_max': self.physical_max,
                      'elevation_correction': self.elevation_correct,
                      'temporal_interp_method': self.temporal_method,
                      'spatial_interp_method': self.spatial_method,
                      'data_source': self.data_source,
                      'source_dir': self.source_dir,
                      'psm_units': self.units,
                      'psm_scale_factor': self.scale_factor,
                      'chunks': self.chunks,
                      })
        return attrs

    @property
    def cache_file(self):
        """Get the nearest neighbor result cache csv file for this var.

        Returns
        -------
        _cache_file : False | str
            False for no caching, or a string filename (no path).
        """
        return self._cache_file

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
        if self._mask is None:
            if self._name in self.var_meta['var'].values:
                self._mask = self.var_meta['var'] == self._name
            else:
                raise KeyError('Variable "{}" not found in NSRDB meta.'
                               .format(self._name))

        return self._mask

    @property
    def data_source(self):
        """Get the data source.

        Returns
        -------
        data_source : str
            Data source.
        """
        return str(self.var_meta.loc[self.mask, 'data_source'].values[0])

    @property
    def elevation_correct(self):
        """Get the elevation correction preference.

        Returns
        -------
        elevation_correct : bool
            Whether or not to use elevation correction for the current var.
        """
        temp = self.var_meta.loc[self.mask, 'elevation_correct']
        ec = bool(temp.values[0])
        if np.isnan(temp.values[0]):
            ec = False

        return ec

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
        """Get the variable's intended storage scale factor.

        Returns
        -------
        scale_factor : float
            Scale factor for the current variable. Data is multiplied by this
            scale factor before being stored.
        """
        return float(self.var_meta.loc[self.mask, 'scale_factor'].values[0])

    @property
    def date(self):
        """Get the date for this handler

        Returns
        -------
        datetime.date
        """
        return self._date

    @property
    def next_date(self):
        """Get the date after the date for this handler. This is used to get
        the data for the next date for temporal interpolation

        Returns
        -------
        datetime.date
        """
        return self._date + dt.timedelta(days=1)

    @property
    def doy(self):
        """Get the day of year for this handler

        Returns
        -------
        int
        """
        return self.date.timetuple().tm_yday

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

    def _get_pattern(self, key):
        """Get the source file pattern which is sent to glob()

        Parameters
        ----------
        key : str
            This should be either 'pattern' or 'next_pattern', corresponding to
            the current date pattern or the next date pattern

        Returns
        -------
        str | None
        """
        pat = None
        if key in self._var_meta:
            pat = self.var_meta.loc[self.mask, key].values[0]

        if isinstance(pat, (int, float)):
            pat = None

        msg = 'Bad pattern: {} {}'.format(pat, type(pat))
        assert isinstance(pat, (type(None), str)), msg

        return pat

    @property
    def pattern(self):
        """Get the source file pattern which is sent to glob().

        Returns
        -------
        str | None
        """
        return self._get_pattern('pattern')

    @property
    def next_pattern(self):
        """Get the next date source file pattern which is sent to glob().

        Returns
        -------
        str | None
        """
        return self._get_pattern('next_pattern')

    @property
    def source_dir(self):
        """Get the source directory containing the variable data files.

        Returns
        -------
        source_dir : str
            Directory containing source data files (with possible sub folders).
        """
        sd = self.var_meta.loc[self.mask, 'source_directory'].values[0]
        if not sd:
            warn('Using default data directory for "{}"'
                 .format(self.name))
            sd = self.DEFAULT_DIR

        return str(sd)

    def _get_file(self, key):
        """Get the file path for the target NSRDB variable name based on
        the glob self.pattern or self.next_pattern.

        Parameters
        ----------
        key : str
            The should be either 'pattern' or 'next_pattern', corresponding to
            the current date pattern or the next date pattern

        Returns
        -------
        str
        """
        pat = getattr(self, key)
        fps = NFS(pat).glob()
        if not any(fps) or len(fps) > 1:
            emsg = ('Could not find or found too many source files '
                    'for dataset "{}" with glob pattern: "{}". '
                    'Found {} files: {}'
                    .format(self.name, pat, len(fps), fps))
            logger.error(emsg)
            raise FileNotFoundError(emsg)

        return fps[0]

    @property
    def file(self):
        """Get the file path for the target NSRDB variable name based on
        the glob self.pattern.

        Returns
        -------
        str
        """
        return self._get_file(key='pattern')

    @property
    def next_file(self):
        """Get the file path for the date for the target NSRDB variable
        name based on the glob self.next_pattern. The file is used to get the
        data for the next date for temporal interpolation

        Returns
        -------
        str
        """
        return self._get_file(key='next_pattern')

    @property
    def next_file_exists(self):
        """Check if file for next date exists"""
        fps = NFS(self.next_pattern).glob()
        return any(fps) or len(fps) > 1

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
    def file_set(self):
        """Get the source file set name for the NSRDB variable. This is
        typically used for MERRA source filesets such as tavg1_2d_aer_Nx or
        tavg1_2d_slv_Nx (for MERRA)

        Returns
        -------
        str
        """
        return str(self.var_meta.loc[self.mask, 'file_set'].values[0])

    @property
    def dset_name(self):
        """Get the source dataset name for the NSRDB variable. This is
        typically the netcdf or h5 source dataset name for the variable such as
        T2M or TOTANGSTR (for MERRA temp and alpha)

        Returns
        -------
        str
        """
        return str(self.var_meta.loc[self.mask, 'dset_name'].values[0])

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
    def chunks(self):
        """Get the variable's intended storage chunk shape.

        Returns
        -------
        chunks : tuple
            Data storage chunk shape (row_chunk, col_chunk).
        """
        # pylint: disable=no-member
        r = self.var_meta.loc[self.mask, 'row_chunks'].values[0]
        c = self.var_meta.loc[self.mask, 'col_chunks'].values[0]
        try:
            r = int(r)
        except ValueError:
            r = None

        try:
            c = int(c)
        except ValueError:
            c = None

        return (r, c)

    @property
    def physical_min(self):
        """Get the variable's physical minimum value.

        Returns
        -------
        physical_min : float
            Physical minimum value for the variable. Variable range can be
            truncated at this value. Must be consistent with the final dtype
            and scale factor.
        """
        return float(self.var_meta.loc[self.mask, 'min'].values[0])

    @property
    def physical_max(self):
        """Get the variable's physical maximum value.

        Returns
        -------
        physical_max : float
            Physical maximum value for the variable. Variable range can be
            truncated at this value. Must be consistent with the final dtype
            and scale factor.
        """
        return float(self.var_meta.loc[self.mask, 'max'].values[0])

    def pre_flight(self):
        """Perform pre-flight checks - source dir check.

        Returns
        -------
        missing : str
            Look for the source dir and return the string if not found.
            If nothing is missing, return an empty string.
        """

        missing = ''
        # empty cell (no source dir) evaluates to 'nan'.
        if self.source_dir != 'nan' and ~np.isnan(self.source_dir):
            if not NFS(self.source_dir).exists():
                # source dir is not nan and does not exist
                missing = self.source_dir

        return missing

    def scale_data(self, array):
        """Perform a safe data scaling operation on a source data array.

        Steps:
            1. Enforce physical range limits
            2. Apply scale factor (mulitply)
            3. Round if integer
            4. Enforce dtype bit range limits
            5. Perform dtype conversion
            6. Return manipulated array

        Parameters
        ----------
        array : np.ndarray
            Source data array with full precision (likely float32).

        Returns
        -------
        array : np.ndarray
            Source data array with final datatype.
        """

        # check to make sure variable is in NSRDB meta config
        if self._name in self.var_meta['var'].values:

            # if the data is not in the final dtype yet
            if not np.issubdtype(self.final_dtype, array.dtype):

                # Warning if nan values are present. Will assign d_min below.
                if np.sum(np.isnan(array)) != 0:
                    d_min = ''
                    if np.issubdtype(self.final_dtype, np.integer):
                        d_min = np.iinfo(self.final_dtype).min

                    w = ('NaN values found in "{}" before dtype conversion '
                         'to "{}". Will be assigned value of: "{}"'
                         .format(self.name, self.final_dtype, d_min))
                    logger.warning(w)
                    warn(w)

                # truncate unscaled array at physical min/max values
                array[array < self.physical_min] = self.physical_min
                array[array > self.physical_max] = self.physical_max

                if self.scale_factor != 1:
                    # apply scale factor
                    array *= self.scale_factor

                # if int, round at decimal precision determined by scale factor
                if np.issubdtype(self.final_dtype, np.integer):
                    array = np.round(array)

                    # Get the min/max of the bit range
                    d_min = np.iinfo(self.final_dtype).min
                    d_max = np.iinfo(self.final_dtype).max

                    # set any nan values to the min of the bit range
                    array[np.isnan(array)] = d_min

                    # Truncate scaled array at bit range min/max
                    array[array < d_min] = d_min
                    array[array > d_max] = d_max

                # perform type conversion to final dtype
                array = array.astype(self.final_dtype)

        return array

    def unscale_data(self, array):
        """Perform a safe data unscaling operation on a source data array.

        Parameters
        ----------
        array : np.ndarray
            Scaled source data array with integer precision.

        Returns
        -------
        array : np.ndarray
            Unscaled source data array with float32 precision.
        """

        # check to make sure variable is in NSRDB meta config
        if self._name in self.var_meta['var'].values:

            # if the data is not in the desired dtype yet
            if not np.issubdtype(np.float32, array.dtype):

                # increase precision to float32
                array = array.astype(np.float32)

                # apply scale factor
                array /= self.scale_factor

        return array

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
        # pylint: disable=no-member
        mask = (ti.month == date.month) & (ti.day == date.day)
        ti = ti[mask]

        return ti


class BaseDerivedVar(ABC):
    """Base class for variables derived from datasets in source data"""

    # Class variable to store list of strings that are datasets interpolated
    # from source data like MERRA that are used to derive this variable
    DEPENDENCIES = tuple()

    @abstractmethod
    def derive(self):
        """Placeholder for derive method"""
