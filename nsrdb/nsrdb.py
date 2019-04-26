# -*- coding: utf-8 -*-
"""Entry point for NSRDB data pipeline execution.


Created on Thu Apr 25 15:47:53 2019

@author: gbuster
"""

import os
from nsrdb import CONFIGDIR
from nsrdb.data_model import DataModel, VarFactory
from nsrdb.file_handlers.outputs import Outputs


def run_data_model(out_dir, date, nsrdb_grid, nsrdb_freq='5min',
                   cloud_extent='east'):
    """Run data model, saving each all-sky-ready variable to a seperate file.

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

    # run for all vars
    var_list = DataModel.ALL_VARS

    # default variable meta path
    var_meta = os.path.join(CONFIGDIR, 'nsrdb_vars.csv')

    # run data model
    data_model = DataModel.run_multiple(var_list, var_meta, date, nsrdb_grid,
                                        nsrdb_freq=nsrdb_freq, parallel=True,
                                        cloud_extent=cloud_extent,
                                        return_obj=True)

    # output handling for each entry in data model
    for var, arr in data_model._processed.items():
        if var not in ['time_index', 'meta']:
            # filename format is YYYYMMDD_varname.h5
            fname = ('{}{}{}_{}.h5'.format(date.year, str(date.month).zfill(2),
                                           str(date.day).zfill(2)), var)
            out_file = os.path.join(out_dir, fname)

            # make file for each var
            with Outputs(out_file, mode='w') as fout:
                fout.time_index = data_model.nsrdb_ti
                fout.meta = data_model.nsrdb_grid

                var_obj = VarFactory.get_base_handler(var_meta, var, date)
                attrs = var_obj.attrs

                fout._add_dset(dset_name=var, data=arr,
                               dtype=var_obj.final_dtype,
                               chunks=None, attrs=attrs)
