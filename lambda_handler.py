"""
Lambda function handler
"""
from cloud_fs import FileSystem
from datetime import datetime
import h5py
import json
from nsrdb import NSRDB
from nsrdb.data_model.clouds import CloudVar
import numpy as np
import os
import pandas as pd
from rex import init_logger, safe_json_load
import sys
import tempfile
import time


class LambdaHandler(dict):
    """
    Lambda Handler class
    """
    def __init__(self, event):
        """
        Parameters
        ----------
        event : dict
            Event or test dictionary
        """
        self.update({k.lower(): v for k, v in os.environ.items()})

        if isinstance(event, str):
            event = safe_json_load(event)

        self.update(event)

        rfd = self.get('run_full_day', False)
        self._var_meta, self._timestep = self.load_var_meta(
            self['var_meta'], self.day, run_full_day=rfd)
        self._timestep = None
        self._fpath_out = None
        self._data_model = None

        if self.get('verbose', False):
            log_level = 'DEBUG'
        else:
            log_level = 'INFO'

        self.logger = init_logger('nsrdb', log_level=log_level)
        self.logger.propagate = False

    @property
    def day(self):
        """
        Date (day) to run NSRDB for, in the format YYYYMMDD

        Returns
        -------
        str
        """
        day = self.get('date', None)
        if not day:
            day = datetime.utcnow().strftime("%Y%m%d")

        return day

    @property
    def year(self):
        """
        Year of the day being run through NSRDB

        Returns
        -------
        str
        """
        return self.day[:4]

    @property
    def grid(self):
        """
        Path to .csv with NSRDB grid to compute NSRDB values one

        Returns
        -------
        str
        """
        return self['grid']

    @property
    def fpath_out(self):
        """
        Final file path of output .h5 file, typically on S3

        Returns
        -------
        str
        """
        if self._fpath_out is None:
            out_dir = self['out_dir']
            if not out_dir.endswith(self.year):
                out_dir = os.path.join(out_dir, self.year)

            fname = '{}-{}.h5'.format(self['file_prefix'], self.day)

            self._fpath_out = os.path.join(out_dir, fname)

        return self._fpath_out

    @property
    def temp_dir(self):
        """
        Path to root for temporary directory that .h5 file will be written
        to before being copied to self.fpath_out

        Returns
        -------
        str
        """
        temp_dir = self.get('temp_dir', "/tmp")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        return temp_dir

    @property
    def out_dir(self):
        """
        Final output directory, typically on S3

        Returns
        -------
        str
        """
        return os.path.dirname(self.fpath_out)

    @property
    def fname(self):
        """
        Output .h5 file name

        Returns
        -------
        str
        """
        return os.path.basename(self.fpath_out)

    @property
    def var_meta(self):
        """
        DataFrame with variable meta data needed to run NSRDB

        Returns
        -------
        pandas.DataFrame
        """
        return self._var_meta

    @property
    def timestep(self):
        """
        Numerical timestep of data to update in self.fpath_out. The actual
        timestamp being updated is data_model.nsrdb_ti[timestamp] or
        time_index[timestep] where time_index is pulled from self.fpath_out
        If None the entire day will be run and dumped to .h5.

        Returns
        -------
        int
        """
        return self._timestep

    @property
    def factory_kwargs(self):
        """
        Dictionary of factory kwargs to pass to NSRDB. These will override
        values in self.var_meta

        Returns
        -------
        dict
        """
        factory_kwargs = {'air_temperature': {'handler': 'GfsVar'},
                          'alpha': {'handler': 'NrelVar'},
                          'aod': {'handler': 'NrelVar'},
                          'dew_point': {'handler': 'GfsDewPoint'},
                          'ozone': {'handler': 'GfsVar'},
                          'relative_humidity': {'handler': 'GfsVar'},
                          'ssa': {'handler': 'NrelVar'},
                          'surface_pressure': {'handler': 'GfsVar'},
                          'total_precipitable_water': {'handler': 'GfsVar'},
                          'wind_direction': {'handler': 'GfsVar'},
                          'wind_speed': {'handler': 'GfsVar'},
                          }
        factory_kwargs = self.get("factory_kwargs", factory_kwargs)
        if isinstance(factory_kwargs, str):
            factory_kwargs = json.loads(factory_kwargs)

        return factory_kwargs

    @property
    def freq(self):
        """
        Temporal frequency to run the NSRDB at, default is '5min'

        Returns
        -------
        str
        """
        return self.get('freq', '5min')

    @property
    def data_model(self):
        """
        Completed NSRDB data model

        Returns
        -------
        NSRDB.DataModel
        """
        if self._data_model is None:
            self._data_model = NSRDB.run_full(
                self.day, self.grid, self.freq,
                var_meta=self.var_meta,
                factory_kwargs=self.factory_kwargs,
                fill_all=self.get('fill_all', False),
                low_mem=self.get('low_mem', False),
                max_workers=self.get('max_workers', 1),
                log_level=None)

        return self._data_model

    @staticmethod
    def load_var_meta(var_meta_path, date, run_full_day=False):
        """
        Load variable meta and update cloud variable pattern

        Parameters
        ----------
        var_meta_path : str
            Path to variable meta .csv file
        date : str
            Date that NSRDB will be run for, used to create pattern
        run_full_day : bool, optional
            Flag indicating if the entire day is going to be run or if the most
            recent file should be run, by default False

        Returns
        -------
        var_meta : pandas.DataFrame
            Variable meta table with cloud variables pattern updated
        timestep : int
            Timestep of newest file to run, None if run_full_day is True
        """
        var_meta = pd.read_csv(var_meta_path)
        date = NSRDB.to_datetime(date)
        year = date.strftime('%Y')
        cloud_vars = var_meta['data_source'] == 'UW-GOES'
        var_meta.loc[cloud_vars, 'pattern'] = \
            var_meta.loc[cloud_vars, 'source_directory'].apply(
            lambda d: os.path.join(d, year, '{doy}', '*.nc')).values
        if not run_full_day:
            name = var_meta.loc[cloud_vars, 'var'].values[0]
            cloud_files = CloudVar(name, var_meta, date).file_df
            timestep = np.where(~cloud_files['flist'].isna())[0].max()
            var_meta.loc[cloud_vars, 'pattern'] = \
                cloud_files.iloc[timestep].values[0]
        else:
            timestep = None

        return var_meta, timestep

    @staticmethod
    def update_timestep(out_fpath, data_model, timestep):
        """
        Update the given timestep in out_fpath with data in data_model

        Parameters
        ----------
        out_fpath : str
            Path to output .h5 file to update
        data_model : nsrdb.DataModel
            NSRDB DataModel with computed data for given timestep
        timestep : int
            Position of timestep to update
        """
        with h5py.File(out_fpath, mode='a') as f:
            dump_vars = [v for v in f
                         if v not in ['time_index', 'meta', 'coordinates']]
            for v in dump_vars:
                ds = f[v]
                ds[timestep] = data_model[v][timestep]

    def get_out_vars(self, data_model):
        """
        Determine which variable to dump to .h5

        Parameters
        ----------
        data_model : NSRDB.DataModel
            Completed NSRDB data model

        Returns
        -------
        out_vars :list
            List of NSRDB variables to dump to .h5
        """
        if not self.get('debug_dsets', False):
            out_vars = ['ghi', 'dni', 'dhi',
                        'clearsky_ghi', 'clearsky_dni',
                        'clearsky_dhi', 'fill_flag']
        else:
            out_vars = [d for d
                        in list(data_model.processed_data.keys())
                        if d not in ('time_index', 'meta', 'flag')]

        return out_vars

    def dump_to_h5(self, data_model):
        """
        Dump NSRDB data to .h5 file. Depending on input parameters the
        following can occur:
        1) Run the NSRDB for the full day and dump all data to .h5, this will
           include gap filled data using MlClouds.
        2) Update a single timestep of NSRDB data corresponding to the most
           recently created CLAVR-X file. This will ignore gap-filled data.

        Parameters
        ----------
        data_model : NSRDB.DataModel
            Completed NSRDB data model
        """
        with tempfile.TemporaryDirectory(prefix=f'NSRDB_{self.day}_',
                                         dir=self.temp_dir) as temp_dir:
            local_out = os.path.join(temp_dir, self.fname)
            out_vars = self.get_out_vars(data_model)
            self.logger.info('Dumping data for {} to {}'
                             .format(out_vars, local_out))

            if not self.get('run_full_day', False):
                if FileSystem(self.fpath_out).exists():
                    self.logger.debug('Copying {} to {} to fill in newest '
                                      'timestep'
                                      .format(self.fpath_out, local_out))
                    FileSystem.copy(self.fpath_out, local_out)
                else:
                    self.logger.debug('Initializing {}'.format(local_out))
                    nsrdb = NSRDB(temp_dir, self.year, self.grid,
                                  freq=self.freq,
                                  var_meta=self.var_meta,
                                  make_out_dirs=False)
                    nsrdb._init_output_h5(local_out, out_vars,
                                          data_model.nsrdb_ti,
                                          data_model.nsrdb_grid)

                self.logger.debug('Updating varible data in {} at timestep {}'
                                  .format(local_out, self.timestep))
                self.update_timestep(local_out, data_model, self.timestep)
            else:
                self.logger.debug('Dumping the entire days worth of data for '
                                  '{} in {}'.format(out_vars, local_out))
                for v in out_vars:
                    try:
                        data_model.dump(v, local_out, None, mode='a')
                    except Exception as e:
                        msg = ('Could not write "{}" to disk, got error: {}'
                               .format(v, e))
                        self.logger.warning(msg)

            FileSystem.copy(local_out, self.fpath_out)

    @classmethod
    def run(cls, event, context=None):
        """
        Run NSRDB from given event and update .h5 file on S3

        Parameters
        ----------
        event : dict
            The event dict that contains the parameters sent when the function
            is invoked.
        context : dict, optional
            The context in which the function is called.
        """
        nsrdb = cls(event)

        nsrdb.logger.debug(f'event: {event}')
        nsrdb.logger.debug(f'context: {context}')
        var_meta = nsrdb['var_meta']
        nsrdb.logger.debug(f'NSRDB inputs:'
                           f'\nday = {nsrdb.day}'
                           f'\ngrid = {nsrdb.grid}'
                           f'\nfreq = {nsrdb.freq}'
                           f'\nvar_meta = {var_meta}'
                           f'\nfactory_kwargs = {nsrdb.factory_kwargs}')

        try:
            nsrdb.dump_to_h5(nsrdb.data_model)
        except Exception:
            nsrdb.logger.exception('Failed to run NSRDB!')
            raise

        success = {'statusCode': 200,
                   'body': json.dumps('NSRDB ran successfully and '
                                      f'create/updated {nsrdb.fpath_out}')}

        return success


def handler(event, context):
    """
        Wrapper for NSRDB to allow AWS Lambda invocation

        Parameters
        ----------
        event : dict
            The event dict that contains the parameters sent when the function
            is invoked.
        context : dict
            The context in which the function is called.
        """
    return LambdaHandler.run(event, context=context)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        event = safe_json_load(sys.argv[1])
        ts = time.time()
        LambdaHandler.run(event)
        print('NSRDB lambda runtime: {:.4f} minutes'
              .format((time.time() - ts) / 60))
