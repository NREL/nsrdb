# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

import datetime
import os
from nsrdb import CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs


class NSRDB:
    """Entry point for NSRDB data pipeline execution."""

    DEFAULT_VAR_META = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')

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

        if self._var_meta is None:
            self._var_meta = self.DEFAULT_VAR_META

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

        # run data model
        data_model = DataModel.run_multiple(
            var_list, self._var_meta, date, self._nsrdb_grid,
            nsrdb_freq=self._nsrdb_freq, parallel=parallel,
            cloud_extent=self._cloud_extent, return_obj=True)

        return data_model

    def _exe_fout(self, data_model):
        """Send the daily data model to output files.

        Parameters
        ----------
        data_model : nsrdb.data_model.DataModel
            Daily data model object.
        """

        # output handling for each entry in data model
        for var, arr in data_model._processed.items():
            if var not in ['time_index', 'meta']:
                # filename format is YYYYMMDD_varname.h5
                fname = ('{}{}{}_{}.h5'.format(data_model.date.year,
                         str(data_model.date.month).zfill(2),
                         str(data_model.date.day).zfill(2)), var)
                out_file = os.path.join(self._out_dir, fname)

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

    @classmethod
    def run_day(cls, out_dir, date, nsrdb_grid, nsrdb_freq='5min',
                cloud_extent='east'):
        """Run daily data model, and save output files.

        Parameters
        ----------
        out_dir : str
            Target directory to dump all-sky-ready data files.
        date : datetime.date
            Single day to extract ancillary data for.
        nsrdb_grid : str | pd.DataFrame
            CSV file containing the NSRDB reference grid to interpolate to,
            or a pre-extracted (and reduced) dataframe.
        nsrdb_freq : str
            Final desired NSRDB temporal frequency.
        cloud_extent : str
            Regional (satellite) extent to process for cloud data processing,
            used to form file paths to cloud data files.
        """

        nsrdb = cls(out_dir, date.year, nsrdb_grid, nsrdb_freq=nsrdb_freq,
                    cloud_extent=cloud_extent)

        data_model = nsrdb._exe_daily_data_model(date.month, date.day)

        nsrdb._exe_fout(data_model)
