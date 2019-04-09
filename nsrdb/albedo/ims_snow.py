# -*- coding: utf-8 -*-
"""NSRDB IMS snow data processing tools for snow-based albedo.

Adapted from Nick Gilroy's initial script:
    - https://github.nrel.gov/dav-gis/pv_task/tree/dev/pv_task/ims_workflow

@author: gbuster
"""

from dask.distributed import Client, LocalCluster
import time
import os
import re
import h5py
import calendar
import numpy as np
import pandas as pd
import psutil
from scipy.spatial import cKDTree
import logging
from warnings import warn

from nsrdb.file_handlers.outputs import Outputs
from nsrdb.utilities.loggers import init_logger, NSRDB_LOGGERS
from nsrdb.utilities.execution import PBS
from nsrdb.utilities.file_utils import url_download, unzip_gz

logger = logging.getLogger(__name__)


def mem_str():
    """Get a string to log memory status."""
    mem = psutil.virtual_memory()
    msg = ('{0:.3f} GB used of {1:.3f} GB total ({2:.1f}% used) '
           '({3:.3f} GB free) ({4:.3f} GB available).'
           .format(mem.used / 1e9,
                   mem.total / 1e9,
                   100 * mem.used / mem.total,
                   mem.free / 1e9,
                   mem.available / 1e9))
    return msg


class RetrieveIMS:
    """Class to manage IMS data retrieval"""
    SOURCE = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02156/'
    META_SOURCE = 'ftp://sidads.colorado.edu/DATASETS/NOAA/G02156/metadata/'
    META_FILES = {'1km': ('IMS1kmLats.24576x24576x1.tar.gz',
                          'IMS1kmLons.24576x24576x1.tar.gz'),
                  '4km': ('imslat_4km.bin.gz', 'imslon_4km.bin.gz'),
                  '24km': ('imslat_24km.bin.gz', 'imslon_24km.bin.gz'),
                  }

    def __init__(self, target_path, year, res='1km'):
        """
        Parameters
        ----------
        target_path : str
            Directory to save the downloaded files.
        year : int
            Year to download files for.
        res : str
            Grid resolution to get snow data from (24km, 4km, 1km).
        """

        self._target_path = target_path
        self._year = year
        self._res = res

    def retrieve_data(self):
        """Retrieve ims snow data files from colorado.edu.

        Note that the days in the IMS file name correspond to the day on which
        the data was processed (day after the data's timestep). So the first
        day of the year is removed and the day of the next year is added. This
        is later renamed.
        """

        # get a list of preexisting data to check
        check_list = os.listdir(self._target_path)

        failed_urls = []
        n_days = 365
        if calendar.isleap(self._year):
            n_days = 366

        # available datasets: 1km, 24km, 4km
        base_fname = 'ims{year}{day}_{res}_v1.3.asc.gz'
        days = [str(d).zfill(3) for d in range(1, n_days + 1, 1)]
        years = range(self._year, self._year + 1)

        flinks = ['/'.join([self.SOURCE, self._res, str(self._year),
                           base_fname.format(res=self._res, year=self._year,
                                             day=day)])
                  for year in years
                  for day in days]

        # remove first day and add 1st day of next year because day in filename
        # is processed day, 1 day after data's day.
        del flinks[0]
        flinks.append('/'.join([self.SOURCE, self._res, str(self._year + 1),
                                base_fname.format(res=self._res,
                                                  year=self._year + 1,
                                                  day='001')]))

        # iterate through all desired files
        for url in flinks:
            # don't download files that have already been downloaded
            fname = url.split('/')[-1]
            if fname in check_list:
                logger.info('Skipping (already exists): {}'.format(fname))
            else:
                logger.info('Downloading {}'.format(fname))
                dfname = os.path.join(self._target_path, fname)
                failed = url_download(url, dfname)
                if failed:
                    failed_urls.append(failed)

        if not failed_urls:
            logger.info('All IMS files downloaded for {} with no failures.'
                        .format(self._year))
        else:
            logger.info('The following IMS downloads failed:\n{}'
                        .format(failed_urls))
        return failed_urls

    def retrieve_meta(self):
        """Retrieve the IMS meta data for a given grid resolution."""
        flist = self.META_FILES[self._res]

        # get a list of preexisting data to check
        check_list = os.listdir(self._target_path)

        for f in flist:
            if f in check_list:
                logger.info('Skipping (already exists): {}'.format(f))
            else:
                logger.info('Downloading {}'.format(f))
                url_download(os.path.join(self.SOURCE, f),
                             os.path.join(self._target_path, f))

    def batch_rename(self):
        """Rename the IMS gzipped files in the target path.

        Files are renamed in a sorted ascending order in timestep, so nothing
        is overwritten. Only *.asc.gz files are renamed so that files are not
        continually renamed after they have been unzipped.
        """

        # open all files
        files = sorted(os.listdir(self._target_path))

        # check number files affected by batch rename
        counter = 0
        # loop through all files in directory
        for f in files:
            if f.endswith('.asc.gz'):

                # extract the timestamp in format YYYYDDD where DDD in [1, 366]
                match = re.match(r'.*([1-2][0-9]{3}[0-3][0-9]{2})', f)
                if match:
                    timestamp = match.group(1)
                    year = int(timestamp[0:4])
                    day = int(timestamp[4:])

                    if day < 1 or day > 366:
                        raise ValueError('Day "{}" is out of expected range '
                                         '[1, 366].'.format(day))

                    # calculate the previous day
                    if day != 1:
                        day -= 1
                    else:
                        # previous year is required
                        year -= 1
                        day = 365
                        if calendar.isleap(year):
                            # previous year is a leap year
                            day = 366

                    # make the new timestamp and file string
                    new_timestamp = '{}{}'.format(year, str(day).zfill(3))
                    f_new = f.replace(timestamp, new_timestamp)

                    logger.info('Renaming {} to {}'.format(f, f_new))

                    os.rename(os.path.join(self._target_path, f),
                              os.path.join(self._target_path, f_new))
                    counter += 1

        logger.info('{} files renamed in: {}'
                    .format(counter, self._target_path))

    @classmethod
    def run(cls, target_path, year, log_level='INFO'):
        """Run the IMS data retrieval and data unpacking pipeline.

        Parameters
        ----------
        target_path : str
            Directory to save the downloaded files.
        year : int
            Year to download files for.
        log_level : str
            Level to log messages at.

        Returns
        -------
        failed : list
            List of files that failed to download.
        """
        init_logger(__name__, log_file=None, log_level=log_level)
        init_logger('nsrdb.utilities.file_utils', log_file=None,
                    log_level=log_level)
        ims = cls(target_path, year)
        failed = ims.retrieve_data()
        ims.retrieve_meta()
        ims.batch_rename()
        unzip_gz(ims._target_path)
        return failed


class ProcessIMS:
    """This class is designed to process the IMS snow data and match it to
    the NSRDB dataset extent. The IMS data can be accessed and downloaded
    from (http://nsidc.org/data/docs/noaa/g02156_ims_snow_ice_analysis/)
    The NSRDB data is stored on the Peregrine HPC. The output of this
    script is a HDF snow dataset with the same extent as the NSRDB. In
    addition a time array and fill flag dataset has beeen created to cross
    reference days with missing data.

    INPUT FILE FORMAT
    ------------------
    snow_cover = 0 - 4
        0 = outside the coverage area
        1 = sea
        2 = land
        3 = sea ice
        4 = snow covered land

    OUTPUT FILE FORMAT
    ------------------
    snow_cover = 0 - 2.
        0 = no snow
        1 = snow
        2 = no data
    """

    def __init__(self, year, nsrdb_meta, ims_lon, ims_lat, ims_data_dir,
                 output_hdf, res='1km', hpc=False):
        """
        Parameters
        ----------
        year : int
            Year to process IMS snow data for.
        nsrdb_meta : str
            CSV filename with path that contains the current NSRDB meta data
            that the IMS data will be mapped to.
        ims_lon : str
            Filename with path to unzipped IMS longitude data.
        ims_lat : str
            Filename with path to unzipped IMS latitude data.
        ims_data_dir : str
            Path to the source IMS files that are to be processed.
        output_hdf : str
            Target .h5 filename with path that the processed snow data will be
            saved to.
        res : str
            Source IMS data grid resolution (4km, 1km).
        hpc : bool
            Flag for HPC parallel computation.
        """

        self.year = year
        self.ims_lon = ims_lon
        self.ims_lat = ims_lat
        self.ims_data_dir = ims_data_dir
        self.output_hdf = output_hdf
        self.res = res
        self.hpc = hpc

        # find nearest neighbor for NSRDB centroids from IMS centroids.
        if nsrdb_meta.endswith('.csv'):
            logger.info('Getting NSRDB meta data: {}'.format(nsrdb_meta))
            self.meta = pd.read_csv(nsrdb_meta, encoding="ISO-8859-1",
                                    low_memory=False)
        else:
            raise TypeError('NSRDB meta data file should be input as a csv.')

    @staticmethod
    def extract_values(fname, res='1km'):
        """Extract IMS data from a file.

        Parameters
        ----------
        fname : str
            Target IMS data file with path.
        res : str
            Grid resolution of the target IMS data file (4km, 1km)

        Returns
        -------
        arr : np.ndarray
            Extracted and flattened IMS snow data from target file with
            dtype int8. Data contains values [0, 1, 2, 3, 4]
        """
        dat = open(fname, 'r')
        lines = dat.readlines()
        out = []
        for line in lines:
            # put a check in place to search for non numeric characters
            # and break the script.
            if re.search('[a-z]', line.strip()) is None:
                out.extend([int(l) for l in list(line.strip())])
        if res == '4km':
            arr = np.flipud(np.array(out).reshape((6144, 6144))).flatten()\
                .astype(np.int8)

        elif res == '1km':
            arr = np.flipud(np.array(out).reshape((24576, 24576))).flatten()\
                .astype(np.int8)

        else:
            raise ValueError('Should not be using the 24km IMS grid data.')
        return arr

    @staticmethod
    def get_1k_meta(ims_lat_file, ims_lon_file):
        """Get IMS meta data (lat/lon df) from 1k meta files."""
        # 1km workflow ONLY
        # open latitude file for 1km resolution.
        with open(ims_lat_file, 'rb') as f:
            lat = np.fromfile(f, dtype='<d', count=24576 * 24576)\
                .astype(np.float32)

        # open longitude file for 1km resolution.
        with open(ims_lon_file, 'rb') as f:
            lon = np.fromfile(f, dtype='<d', count=24576 * 24576)\
                .astype(np.float32)

        # correct positive only longitude
        lon = np.where(lon > 180, lon - 360, lon)

        meta = pd.DataFrame({'latitude': lat, 'longitude': lon})
        return meta

    def get_indices(self):
        """Get the IMS to NSRDB mapping index.

        Returns
        -------
        valid_data : np.ndarray
            Boolean mask on the IMS source data. Apply this mask to the IMS
            source data to get valid data.
        indices :  np.ndarray
            KDTree query results. Use this to index extracted IMS data that has
            already been put through the valid_data mask.
        """

        nsrdb_pnts = np.vstack([self.meta['lons'],
                                self.meta['lats']]).T

        # create extents for a mask to match the extent of the NSRDB dataset.
        neg_lon = self.meta['lons'][self.meta['lons'] < 0].max()
        pos_lon = self.meta['lons'][self.meta['lons'] > 0].min()

        if self.res == '4km':
            # 4km workflow ONLY
            sample_ims_f = os.path.join(self.ims_data_dir,
                                        os.listdir(self.ims_data_dir)[0])
            data = self.extract_values(sample_ims_f, self.res)
            # open longitude file for 4km resolution.
            x = np.fromfile(self.ims_lon, dtype='<f4')
            # correct positive only longitude
            x = np.where(x > 180, x - 360, x)
            # open latitude file for 4km resolution.
            y = np.fromfile(self.ims_lat, dtype='<f4')
            # Create 4km mask
            x_mask = (x < (neg_lon + 0.35)) | (x > pos_lon - 0.35)

            valid_data = ((data == 4) | (data == 2)) & x_mask
            x_pnts = x[valid_data]
            y_pnts = y[valid_data]

        elif self.res == '1km':
            # 1km workflow ONLY
            # open latitude file for 1km resolution.
            with open(self.ims_lat, 'rb') as f:
                y = np.fromfile(f, dtype='<d', count=24576 * 24576)
                # y = np.reshape(data, [24576, 24576], order='F')

            # open longitude file for 1km resolution.
            with open(self.ims_lon, 'rb') as f:
                x = np.fromfile(f, dtype='<d', count=24576 * 24576)
                # x = np.reshape(data, [24576, 24576], order='F')

            sample_ims_f = os.path.join(self.ims_data_dir,
                                        os.listdir(self.ims_data_dir)[0])

            data = self.extract_values(sample_ims_f, self.res)

            # correct positive only longitude
            x = np.where(x > 180, x - 360, x)

            # Create 4km mask
            x_mask = (x < (neg_lon + 0.35)) | (x > pos_lon - 0.35)
            valid_data = ((data == 4) | (data == 2)) & x_mask

            x_pnts = x[valid_data]
            y_pnts = y[valid_data]

        # run cKDTree for 1km or 4km resolution.
        logger.info('Building cKDTree...')
        tree = cKDTree(np.vstack([x_pnts, y_pnts]).T)
        logger.info('Querying cKDTree...')
        indices = tree.query(nsrdb_pnts)[1]

        logger.debug('valid_data array has shape {} and dtype {}'
                     .format(valid_data.shape, valid_data.dtype))
        logger.debug('indices array has shape {} and dtype {}'
                     .format(indices.shape, indices.dtype))

        return valid_data, indices

    def extract_snow_cover(self):
        """
        Match IMS data latitude and longitude to NSRDB data extent.
        Run a cKDTree to find the nearest neighbor. Open all the IMS
        files to be run through the extract_values function. Finally,
        save the matched data as HDF file along with date and fill flag.

        Returns
        -------
        output : np.ndarray
            Compiled IMS snow cover data with shape (number days, number sites)
            snow_cover = 0, 1, or 2.
                0 = no snow
                1 = snow
                2 = no data
        """

        # glob to open all the files in the directory
        # regardless of name or version.
        available = os.listdir(self.ims_data_dir)
        # print all_data, 'all_data'

        # Set number of days and handle potential leap year
        days = (366 if calendar.isleap(self.year) else 365)

        valid_data, indices = self.get_indices()

        futures = []

        for i in range(0, days):
            day = i + 1
            fname = None
            # add one to row so now it is equal to 0.
            # create a string to identify the files by julian calendar.
            year_day = str(self.year) + str(day).zfill(3)

            for favail in available:
                if year_day in favail:
                    fname = favail
                    break
            # create the file name.
            # identify the file path and pull in resolution & filename.
            fpath = os.path.join(self.ims_data_dir, fname)

            if self.hpc:
                if 'client' not in locals():
                    logger.info('Starting Dask Client...')
                    cluster = LocalCluster(
                        n_workers=10, threads_per_worker=1,
                        memory_limit=0)
                    client = Client(cluster)
                    client.run(NSRDB_LOGGERS.init_logger, __name__)
                logger.debug('Kicking off future #{}'.format(i))
                futures.append(client.submit(self.run_future, fpath, self.res,
                                             indices, valid_data))
            else:
                futures.append(self.run_future(fpath, self.res, indices,
                                               valid_data))

        if self.hpc:
            logger.info('Waiting on parallel futures...')
            futures = client.gather(futures)
            client.close()
            logger.info('Futures gathered and client is closed.')
            logger.info(mem_str())

        # multiplying the array by two creates our fill flag, (2 = no data).
        # then mapping in the boolean results from the snow cover data sets no
        # snow to 0 and snow to 1
        logger.info('Compiling outputs...')
        output = np.ones(shape=(days, self.meta.shape[0]), dtype=np.int8) * 2
        for i, future in enumerate(futures):
            output[i, :] = future

        logger.info('Outputs compiled.')
        logger.info(mem_str())
        return output

    @staticmethod
    def run_future(fpath, res, indices, valid_data):
        """Execute a single IMS data extract, can be run as parallel futures.

        Parameters
        ----------
        fpath : str
            Target filepath for single IMS data to extract.
        res : str
            IMS data resolution ('4km', '1km')
        indices :  np.ndarray
            KDTree query results. Use this to index extracted IMS data that has
            already been put through the valid_data mask.
        valid_data : np.ndarray
            Boolean mask on the IMS source data. Apply this mask to the IMS
            source data to get valid data.

        Returns
        -------
        output : np.ndarray
            Boolean array representing the IMS data corresponding to the NSRDB
            extent with snow (data == 4).
        """
        try:
            t0 = time.time()
            # call the function.
            data = ProcessIMS.extract_values(fpath, res)[valid_data]
            data_nsrdb = data[indices]
            logger.info('Extracted "{}" in {} minutes'
                        .format(fpath, (time.time() - t0) / 60.0))
            output = (data_nsrdb == 4)
        except Exception as e:
            logger.warning('Could not process: "{}" for timestep {}'
                           .format(fpath))
            logger.exception(e)
            output = None
            pass

        # log the node's memory usage
        msg = mem_str()
        logger.debug(msg)

        return output

    def flush(self, processed_ims):
        """Flush the processed IMS data to the output hdf target file.

        Parameters
        ----------
        processed_ims : np.ndarray
            Processed IMS data for the entire year. Data is 2D with days in
            the y axis and NSRDB sites in the x axis.
        """

        # write final output after building an array with snow cover
        # from all the 24k and 4k daily data.
        logger.info("Flushing processed IMS data to: {}"
                    .format(self.output_hdf))

        meta_rec_array = Outputs.to_records_array(self.meta)

        with h5py.File(self.output_hdf, 'w') as hfile:

            # write meta lat/lon pairs as floats
            logger.info('Writing meta data to output...')
            # hfile['meta'] = self.meta.loc[:, ['lats', 'lons']]\
            #     .astype(np.float64)

            hfile.create_dataset('meta', data=meta_rec_array)

            logger.info('Writing time index to output...')
            # create a time index indicating the year_month_day across the year
            time_index = pd.date_range('{}-01-01'.format(self.year),
                                       '{}-12-31'.format(self.year), freq='D')
            time_index = np.array(time_index.astype(str), dtype='S20')
            hfile['time_index'] = time_index

            logger.info('Writing snow cover data to output...')
            hfile.create_dataset('snow_cover', data=processed_ims,
                                 dtype=np.int8)

            # create a fill flag array to put in here
            #   (would be 1D (number of days in 18 years))
            # values indicating if we have IMS data or it was a missing day.
            # 2 == no data
            # 0 == data

            logger.info('Writing fill flag data to output...')
            hfile.create_dataset('fill_flag', data=np.where(
                processed_ims.max(axis=1) == 2, 2, 0))

    def main(self):
        """Run the main IMS data processing methods."""
        t1 = time.time()
        out = self.extract_snow_cover()
        self.flush(out)
        # time it took to run function
        logger.info("Completed in {} minutes"
                    .format((time.time() - t1) / 60.0))

    @classmethod
    def process(cls, year, hpc=False, log_level='DEBUG'):
        """Run processing with test filepaths.

        Parameters
        ----------
        year : int
            Year to process IMS snow data for.
        hpc : bool
            Flag for HPC parallel computation.
        log_level : str
            Logging level for this module (DEBUG or INFO).
        """

        init_logger(__name__, log_file='ims.log', log_level=log_level)
        output_hdf = ('/scratch/ngilroy/nsrdb/albedo/outputs/'
                      'ims_{}_daily_snow_cover.h5'
                      .format(year))
        nsrdb_meta = '/projects/PXS/reference_grids/east_psm_extent_2k.csv'
        lon4 = ('/scratch/ngilroy/nsrdb/albedo/ims_lat_lon/'
                'IMS1kmLons.24576x24576x1.double')
        lat4 = ('/scratch/ngilroy/nsrdb/albedo/ims_lat_lon/'
                'IMS1kmLats.24576x24576x1.double')
        ims_data_dir = '/scratch/ngilroy/nsrdb/albedo/ims_1k'

        ims = cls(year, nsrdb_meta, lon4, lat4, ims_data_dir,
                  output_hdf, hpc=hpc)
        ims.main()
        print('IMS snow data processing script is complete, check outputs!')

    @classmethod
    def peregrine(cls, year, alloc='pxs', queue='short', log_level='DEBUG',
                  stdout_path='/scratch/ngilroy/nsrdb/albedo/outputs/'):
        """Run IMS snow data processing on a Peregrine node for one year.

        Parameters
        ----------
        year : int
            Year to process IMS snow data for.
        alloc : str
            Peregrine project allocation (pxs).
        queue : str
            Target Peregrine job queue.
        log_level : str
            Logging level for this module (DEBUG or INFO).
        stdout_path : str
            Path to dump stdout/stderr files.
        """

        name = 'IMS_{}'.format(year)
        cmd = ('python -c '
               '\'from nsrdb.albedo.ims_snow import ProcessIMS; '
               'ProcessIMS.process({year}, hpc=True, log_level="{log_level}")'
               '\''
               .format(year=year, log_level=log_level))

        pbs = PBS(cmd, alloc=alloc, queue=queue, name=name,
                  stdout_path=stdout_path)

        print('\ncmd:\n{}\n'.format(cmd))

        if pbs.id:
            msg = ('Kicked off job "{}" (PBS jobid #{}) on '
                   'Peregrine.'.format(name, pbs.id))
        else:
            msg = ('Was unable to kick off job "{}". '
                   'Please see the stdout error messages'
                   .format(name))
        print(msg)


def year_1k_to_h5(year, ims_dir, fout,
                  f_lat=('/scratch/ngilroy/nsrdb/albedo/ims_lat_lon/'
                         'IMS1kmLats.24576x24576x1.double'),
                  f_lon=('/scratch/ngilroy/nsrdb/albedo/ims_lat_lon/'
                         'IMS1kmLons.24576x24576x1.double')):
    """Extract year of IMS 1km asc files to single h5.

    INPUT IMS SNOW FORMAT
    ------------------
    snow_cover = 0 - 4
        0 = outside the coverage area
        1 = sea
        2 = land
        3 = sea ice
        4 = snow covered land

    OUTPUT SNOW FORMAT
    ------------------
    snow_cover = 0 - 2.
        0 = no snow
        1 = snow
        -1 = no data

    Parameters
    ----------
    year : int | str
        Year to process IMS data for. This number should be found in the IMS
        file names.
    ims_dir : str
        Directory containing 365 or 366 IMS .asc files for the given year.
    fout : str
        Full file path to target output .h5 file.
    f_lat : str
        IMS latitude meta data file (.double).
    f_lon : str
        IMS longitude meta data file (.double).
    """

    logger.info('Extracting {} 1km IMS data from {} to {}'
                .format(year, ims_dir, fout))

    # mapping of IMS snow values to final snow values
    snow_mapping = {0: 0,
                    1: 0,
                    2: 0,
                    3: 1,
                    4: 1}
    # any other IMS data will be -1
    missing = -1

    # get the sorted file list in the target ims directory
    flist_raw = os.listdir(ims_dir)
    flist = [os.path.join(ims_dir, f) for f in flist_raw if
             str(year) in f and f.endswith('.asc')]
    if flist:
        if len(flist) != 365 and len(flist) != 366:
            raise IOError('Bad number of IMS files: {}. Expected 365 or 366.'
                          .format(len(flist)))
        flist = sorted(flist, key=lambda x: os.path.basename(x)
                       .split('_')[0].strip('ims'))
    else:
        raise IOError('No valid .asc IMS files for {} found in {}'
                      .format(year, ims_dir))

    # get the IMS meta data
    logger.info('Extracting IMS 1km meta data.')
    meta = ProcessIMS.get_1k_meta(f_lat, f_lon)
    if np.sum(meta.isnull().values) > 0:
        warn('IMS meta data has null values!')
    meta_rec_arr = Outputs.to_records_array(meta)

    # make a daily time index
    ti = pd.date_range('1-1-{y}'.format(y=year),
                       '1-1-{y}'.format(y=int(year) + 1),
                       freq='1D')[:-1]
    ti = np.array(ti.astype(str), dtype='S20')

    # write file
    logger.info('Initializing output file: {}'.format(fout))
    with h5py.File(fout, 'w-') as f:
        # initialize datasets
        f.create_dataset('meta', shape=meta_rec_arr.shape,
                         dtype=meta_rec_arr.dtype, data=meta_rec_arr)
        f.create_dataset('time_index', shape=ti.shape, dtype=ti.dtype,
                         data=ti)
        # int8 means 2e6 indices will be 2MB (optimal chunk size)
        f.create_dataset('snow_cover', shape=(len(ti), len(meta)),
                         dtype=np.int8, chunks=(1, int(2e6)))
        f.create_dataset('fill_flag', shape=(len(ti),),
                         dtype=np.int8)

        # iterate through IMS file list
        for i, f_ims in enumerate(flist):
            logger.info('Processing IMS file #{}: {}'.format(i, f_ims))
            arr = ProcessIMS.extract_values(f_ims)
            if len(arr) != len(meta):
                warn('Data from {} has length {} but meta has length {}'
                     .format(f_ims, len(arr), len(meta)))

            # map IMS snow values (keys) to boolean (values)
            for k, v in snow_mapping.items():
                arr = np.where(arr == k, v, arr)

            # flag bad data where snow != boolean no (0) or yes (1)
            bad_data = ((arr != 0) & (arr != 1))
            arr = np.where(bad_data, missing, arr)

            # write to h5 dataset
            f['snow_cover'][i, :] = arr

            # current timestep has missing values, flag for filling
            fill_flag = 0
            if np.sum(arr == missing) > 0:
                fill_flag = 1
                logger.info('Missing data at index {} from file {}'
                            .format(i, f_ims))
            f['fill_flag'][i] = fill_flag


def gap_fill_ims(ims_dir, f_in, f_out, log_level='DEBUG'):
    """Fill any gaps in the IMS daily snow dataset.

    This script is designed to gap fill the daily ims snow dataset. It does so
    by splitting the way the data is gap filled into two parts. The first
    method is to take all days with consecutive missing days and fill it with
    merra snow depth data. The merra data measure snow in meters, this script
    uses an arbitrary threshold of 3cm to be considered a day with snow. The
    second method of gap filling is to take days where only one day by itself
    is missing, if there is both snow on either side, or no snow on either
    side it will be considered a day with snow or no snow respectively. Both
    methods are combined to create a final dataset where there are no gaps
    within the data. Finally the script also outputs a check called
    'dataset_fill'. This check will identify how the data was gap filled:
        - 0 signifies that no gap fill was applied to that day.
        - 1 signifies that a temporal gap fill was applied to that day.
        - 2 signifies that the merra gap fill was applied to that day.

    Parameters
    ----------
    ims_dir : str
        Path of extracted IMS dataset and location to dump filled data file.
    f_in : str
        Input h5 file, the extracted IMS dataset to be filled.
    f_out : str
        Output h5 file, the file to dump filled IMS snow data.
    log_level : str
        Level to log messages at. Log file will be created in the ims_dir.
    """

    # initialize a logger output file for this method in the ims directory.
    init_logger(__name__, log_file=os.path.join(ims_dir, 'gap_fill_ims.log'),
                log_level=log_level)

    # start timer.
    t1 = time.time()

    # open IMS & pull necessary data.
    logger.info('Opening IMS source file: {}.'.format(f_in))
    with h5py.File(os.path.join(ims_dir, f_in), 'r') as hf:
        days = hf['time_index'][...]
        fill = hf['fill_flag'][...]
        ims = hf['snow_cover'][...]
        meta = pd.DataFrame(hf['meta'][...], columns=['lats', 'lons'])

    logger.info('IMS data successfully loaded.')
    logger.info('IMS meta head:\n{}'.format(meta.head()))
    logger.debug('IMS snow_cover shape:\n{}'.format(ims.shape))
    logger.debug('IMS snow_cover head:\n{}'.format(ims[0:10, 0:10]))

    meta_rec_array = Outputs.to_records_array(meta)

    # create an empty parameter to be later written to new hdf file as
    # gap_fill_flag.
    dataset_fill = np.zeros(ims.shape)

    # loop through fill - branch based on temporal fill or merra fill.
    for day, _ in enumerate(fill):
        # if the day was missing
        if fill[day] == 2:
            snow = ((ims[day - 1:day + 2, :] == 1).sum(axis=0)) == 2
            no_snow = ((ims[day - 1:day + 2, :] == 0).sum(axis=0)) == 2

            # change it: if sandwiched by snow we put 1, no snow we put a 0,
            # else we put a 2.
            ims[day, :] = np.where(snow is True, 1,
                                   np.where(no_snow is True, 0, 2))

            # fill with a 1 to signify data are filled from temporal.
            dataset_fill[day, :][ims[day, :] < 2] = 1
            dataset_fill[day, :][ims[day, :] == 2] = 2

            # this branch excludes using merra as a gap fill method after 2014,
            # since the nsrdb stopped incorporating nsrdb in 2014.
            # if int(year) > 2014:
            # arbitrarily use the day before as a gap fill.
            ims[day, :] = ims[day - 1, :]

    logger.info('IMS data successfully gap filled.')
    logger.info('Writing filled IMS data to: {}'.format(f_out))

    # write outputs to a new hdf file.
    with h5py.File(os.path.join(ims_dir, f_out), 'w') as hf2:
        # hf2['meta'] = meta.loc[:, ['latitude', 'longitude']]\
        #     .astype(np.float64)
        hf2.create_dataset('meta', data=meta_rec_array)
        hf2.create_dataset('snow_cover', data=ims, dtype=np.int8)
        hf2.create_dataset('time_index', data=days)
        hf2.create_dataset('fill_flag', data=fill)
        hf2.create_dataset('gap_fill_flag', data=dataset_fill)

    logger.info("Completed in {0:.2f} minutes"
                .format((time.time() - t1) / 60.0))


def append_new_year(source_my, new_year, new_my, log_level='INFO',
                    log_file=None):
    """Append IMS snow data for a single year to the full multi-year data file.

    This script appends the latest gap-filled ims data to the complete version
    of the gap-filled ims dataset which currently is (1998-2015).
    You will need to use 'bigmem' on peregrin otherwise you will raise a
    memory error due to the size of the numpy arrays during concatenation.

    Parameters
    ----------
    source_my : str
        Source multi-year IMS snow h5 file (with path) that the new year will
        be added to.
    new_year : str
        Source single-year new IMS snow h5 file (with path) that will be added
        to the source mult-year data.
    new_my : str
        Target multi-year h5 file (with path) that will be created combining
        source_my and new_year data.
    log_level : str
        Level to log messages at. Logs will be sent to stdout.
    log_file : str
        Target file with path to log messages to.
    """

    # initialize a logger output file for this method in the ims directory.
    init_logger(__name__, log_file=log_file, log_level=log_level)

    # start timer.
    t1 = time.time()

    # read hdfs.
    with h5py.File(source_my, 'r') as hf_old:
        with h5py.File(new_year, 'r') as hf_new:
            # meta does not need to change or be appended to.
            meta = hf_old['meta'][...]

            logger.info('hdfs loaded. Begin to append data.')
            # append new gap-filled ims to old.
            di_old = hf_old['time_index']
            di_new = hf_new['time_index']
            di_concat = np.concatenate([di_old, di_new])
            logger.info('Finished appending time_index.')

            ff_old = hf_old['fill_flag']
            ff_new = hf_new['fill_flag']
            ff_concat = np.concatenate([ff_old, ff_new])
            logger.info('Finished appending fill_flag.')

            gff_old = hf_old['gap_fill_flag'][...]
            gff_new = hf_new['gap_fill_flag'][...]
            gff_concat = np.concatenate([gff_old, gff_new], axis=0)
            logger.info('Finished appending gap_fill_flag.')

            sc_old = hf_old['snow_cover'][...]
            sc_new = hf_new['snow_cover'][...]
            sc_concat = np.concatenate([sc_old, sc_new], axis=0)
            logger.info('Finished appending snow_cover.')

    logger.info('Appended new ims data to old dataset. writing hdf.')
    with h5py.File(new_my, 'w') as hfile:
        hfile.create_dataset('time_index', data=di_concat)
        hfile.create_dataset('fill_flag', data=ff_concat)
        hfile.create_dataset('gap_fill_flag', data=gff_concat)
        hfile.create_dataset('meta', data=meta)
        hfile.create_dataset('snow_cover', data=sc_concat)
    logger.info("Completed in {} minutes".format((time.time() - t1) / 60.0))


def update_albedo_where_snow(
        albedo_h5=('/scratch/ngilroy/nsrdb/albedo/outputs/'
                   'nsrdb_wsa_albedo_daily.h5'),
        snow_h5=('/scratch/ngilroy/nsrdb/albedo/outputs/'
                 'ims_2018_daily_gap_filled_snow_cover.h5'),
        output_h5='/scratch/ngilroy/nsrdb/albedo/outputs/nsrdb_albedo_2018.h5',
        leap_year=False, snow_albedo=0.8669, log_level='DEBUG',
        log_file='/scratch/ngilroy/nsrdb/albedo/update_albedo.log'):
    """Update an Albedo dataset with a fixed snow albedo based on IMS snow data

    Parameters
    ----------
    albedo_h5 : str
        Albedo data source file with path.
    snow_h5 : str
        Source IMS snow data file with path.
    output_h5 : str
        Target final albedo file with path.
    leap_year : bool
        Flag to generate 365 or 366 days of albedo data.
    snow_albedo : float
        Constant value to assign to albedo when snow is present.
    log_level : str
        Level to log messages at.
    log_file : str
        Target file with path to log messages to.
    """

    # initialize a logger output file for this method in the ims directory.
    init_logger(__name__, log_file=log_file, log_level=log_level)

    t1 = time.time()

    logger.info('Opening the source albedo file: {}'.format(albedo_h5))
    with h5py.File(albedo_h5, 'r') as hf:

        # extract the last 365 days of albedo or 366 if leap year
        albedo_year_index = -355
        if leap_year:
            albedo_year_index = -366
        albedo = hf['albedo'][albedo_year_index:, :]
        meta = hf['meta'][...]
        albedo_latlon = pd.DataFrame(meta).loc[:, ['lats', 'lons']]

        # Diagnostics
        logger.debug('Source albedo dataset shape: {}'
                     .format(hf['albedo'].shape))
        logger.debug('Extracted albedo dataset with shape: {}'
                     .format(albedo.shape))
        logger.debug('Source meta has {} entries, head:\n{}'
                     .format(len(meta), meta[0:10]))

    logger.info('Albedo data imported. Reading snow file.')
    with h5py.File(snow_h5, 'r') as hf2:
        ims = hf2['snow_cover'][...]
        time_index = hf2['time_index'][...]
        snow_latlon = pd.DataFrame(hf2['meta'][...],
                                   columns=['lats', 'lons'])
        logger.debug('IMS snow dataset has shape: {}'.format(ims.shape))

    logger.debug('Running NN on albedo and snow meta data')
    tree = cKDTree(snow_latlon.values)
    _, indices = tree.query(albedo_latlon.values, k=1)

    # reduce the IMS snow data to just the indices that match the albedo data.
    snow_latlon = snow_latlon.loc[indices, :]
    # time is in axis-0, sites are in axis-1, index the site axis not time axis
    ims = ims[:, indices]

    diff_lat = np.sum(snow_latlon.iloc[:, 0].values - albedo_latlon.iloc[:, 0])
    diff_lon = np.sum(snow_latlon.iloc[:, 1].values - albedo_latlon.iloc[:, 1])

    logger.debug('Difference in lat/lon arrays (should be zero): {}/{}'
                 .format(diff_lat, diff_lon))

    logger.info('After matching the snow data to the Albedo data using NN, '
                'the IMS snow data has shape {} and the albedo data has shape '
                '{}'.format(ims.shape, albedo.shape))

    logger.info('Snow file data imported. Updating albedo where snow.')
    for day in range(len(albedo)):
        try:
            albedo[day, :] = np.where(ims[day, :] == 1, snow_albedo,
                                      albedo[day, :])
        except Exception as e:
            logger.debug('day: {} failed to write snow albedo value'
                         .format(day))
            logger.exception(e)

    logger.info('Albedo snow updated. Writing output hdf: {}'
                .format(output_h5))
    with h5py.File(output_h5, 'w') as hfile:
        hfile.create_dataset('albedo', data=albedo)
        hfile.create_dataset('time_index', data=time_index)
        hfile.create_dataset('meta', data=meta)

    logger.info("Completed in {} minutes".format((time.time() - t1) / 60.0))
