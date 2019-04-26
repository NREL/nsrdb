# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

import datetime
import pandas as pd
import os
import logging

from nsrdb import CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs
from nsrdb.file_handlers.collection import collect_daily_files


logger = logging.getLogger(__name__)


class NSRDB:
    """Entry point for NSRDB data pipeline execution."""

    DEFAULT_VAR_META = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')

    OUTS = {'nsrdb_ancillary_{y}.h5': ('alpha',
                                       'aod',
                                       'asymmetry',
                                       'dew_point',
                                       'relative_humidity',
                                       'ozone',
                                       'ssa',
                                       'surface_albedo',
                                       'surface_pressure',
                                       'total_precipitable_water',
                                       'air_temperature',
                                       'wind_direction',
                                       'wind_speed'),
            'nsrdb_clouds_{y}.h5': ('cloud_type',
                                    'cld_opd_dcomp',
                                    'cld_reff_dcomp',
                                    'cld_press_acha',
                                    'fill_flag',
                                    'solar_zenith_angle'),
            'nsrdb_irradiance_{y}.h5': ('dhi', 'dni', 'ghi',
                                        'clearsky_dhi',
                                        'clearsky_dni',
                                        'clearsky_ghi')}

    def __init__(self, out_dir, year, nsrdb_grid, nsrdb_freq='5min',
                 cloud_extent='east', var_meta=None):
        """
        Parameters
        ----------
        out_dir : str
            Target directory to dump all-sky-ready data files.
        year : int | str
            Processing year.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        var_meta : str | None
            File path to NSRDB variables meta data. None will use the default
            file from the github repo.
        """

        self._out_dir = out_dir
        self._year = int(year)
        self._nsrdb_grid = nsrdb_grid
        self._nsrdb_freq = nsrdb_freq
        self._cloud_extent = cloud_extent
        self._var_meta = var_meta
        self._ti = None

        if self._var_meta is None:
            self._var_meta = self.DEFAULT_VAR_META

    @property
    def time_index_year(self):
        """Get the NSRDB full-year time index.

        Returns
        -------
        nsrdb_ti : pd.DatetimeIndex
            Pandas datetime index for the current year at the NSRDB resolution.
        """
        if self._ti is None:
            self._ti = pd.date_range('1-1-{y}'.format(y=self._year),
                                     '1-1-{y}'.format(y=self._year + 1),
                                     freq=self._nsrdb_freq)[:-1]
        return self._ti

    @property
    def meta(self):
        """Get the NSRDB meta dataframe from the grid file.

        Returns
        -------
        meta : pd.DataFrame
            DataFrame of meta data from grid file csv.
        """

        if isinstance(self._nsrdb_grid, str):
            self._nsrdb_grid = pd.read_csv(self._nsrdb_grid)
        return self._nsrdb_grid

    def _exe_daily_data_model(self, month, day, var_list=None, parallel=True):
        """Execute the data model for a single day.

        Parameters
        ----------
        month : int | str
            Month to run data model for.
        day : int | str
            Day to run data model for.
        var_list : list | tuple | None
            Variables to process with the data model. None will default to all
            variables.
        parallel : bool
            Flag to perform data model processing in parallel.

        Returns
        -------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        date = datetime.date(year=self._year, month=int(month), day=int(day))

        if var_list is None:
            var_list = DataModel.ALL_VARS

        logger.info('Starting daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        # run data model
        data_model = DataModel.run_multiple(
            var_list, self._var_meta, date, self._nsrdb_grid,
            nsrdb_freq=self._nsrdb_freq, parallel=parallel,
            cloud_extent=self._cloud_extent, return_obj=True)

        logger.info('Finished daily data model execution for {}-{}-{}'
                    .format(month, day, self._year))

        return data_model

    def _exe_fout(self, data_model):
        """Send the single-day data model results to output files.

        Parameters
        ----------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        logger.info('Starting file export of daily data model results to: {}'
                    .format(self._out_dir))

        # output handling for each entry in data model
        for var, arr in data_model._processed.items():
            if var not in ['time_index', 'meta']:
                # filename format is YYYYMMDD_varname.h5
                fname = ('{}{}{}_{}.h5'.format(data_model.date.year,
                         str(data_model.date.month).zfill(2),
                         str(data_model.date.day).zfill(2), var))
                out_file = os.path.join(self._out_dir, fname)

                logger.debug('\tWriting file: {}'.format(fname))

                # make file for each var
                with Outputs(out_file, mode='w') as fout:
                    fout.time_index = data_model.nsrdb_ti
                    fout.meta = data_model.nsrdb_grid

                    var_obj = VarFactory.get_base_handler(self._var_meta, var,
                                                          data_model.date)
                    attrs = var_obj.attrs

                    fout._add_dset(dset_name=var, data=arr,
                                   dtype=var_obj.final_dtype,
                                   chunks=None, attrs=attrs)

        logger.info('Finished file export of daily data model results to: {}'
                    .format(self._out_dir))

    def _init_final_out(self, f_out, dsets):
        """Initialize the final output file.

        Parameters
        ----------
        f_out : str
            File path to final .h5 file.
        dsets : list
            List of dataset / variable names that are to be contained in f_out.
        """

        if not os.path.isfile(f_out):
            logger.info('Initializing {} for the following datasets: {}'
                        .format(f_out, dsets))

            attrs, chunks, dtypes = self.get_dset_attrs(dsets)

            Outputs.init_h5(f_out, dsets, attrs, chunks, dtypes,
                            self.time_index_year, self.meta)

    @staticmethod
    def get_dset_attrs(dsets):
        """Get output file dataset attributes for a set of datasets.

        Parameters
        ----------
        dsets : list
            List of dataset / variable names.

        Returns
        -------
        attrs : dict
            Dictionary of dataset attributes keyed by dset name.
        chunks : dict
            Dictionary of chunk tuples keyed by dset name.
        dtypes : dict
            dictionary of numpy datatypes keyed by dset name.
        """

        attrs = {}
        chunks = {}
        dtypes = {}

        for dset in dsets:
            var_obj = VarFactory.get_base_handler(NSRDB.DEFAULT_VAR_META, dset)
            attrs[dset] = var_obj.attrs
            chunks[dset] = var_obj.chunks
            dtypes[dset] = var_obj.final_dtype

        return attrs, chunks, dtypes

    @classmethod
    def run_day(cls, out_dir, date, nsrdb_grid, nsrdb_freq='5min',
                cloud_extent='east'):
        """Run daily data model, and save output files.

        Parameters
        ----------
        out_dir : str
            Target directory to dump data model output files.
        date : datetime.date | str | int
            Single day to extract ancillary data for.
            Can be str or int in YYYYMMDD format.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        """

        if isinstance(date, (int, float)):
            date = str(int(date))
        if isinstance(date, str):
            if len(date) == 8:
                date = datetime.date(year=int(date[0:4]),
                                     month=int(date[4:6]),
                                     day=int(date[6:]))
            else:
                raise ValueError('Could not parse date: {}'.format(date))

        nsrdb = cls(out_dir, date.year, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                    cloud_extent=cloud_extent)

        data_model = nsrdb._exe_daily_data_model(date.month, date.day)

        nsrdb._exe_fout(data_model)

    @classmethod
    def collect_data_model(cls, daily_dir, out_dir, year, nsrdb_grid,
                           nsrdb_freq='5min'):
        """Init output file and collect daily data model output files.

        Parameters
        ----------
        daily_dir : str
            Directory with daily files to be collected.
        out_dir : str
            Directory to put final output files.
        year : int | str
            Year of analysis
        nsrdb_grid : str
            NSRDB grid file.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        """

        nsrdb = cls(out_dir, year, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                    cloud_extent=None)

        for fname, dsets in cls.OUTS.items():
            if 'irradiance' not in fname:
                f_out = os.path.join(out_dir, fname.format(y=year))
                nsrdb._init_final_out(f_out, dsets)
                collect_daily_files(daily_dir, f_out, dsets)
