# -*- coding: utf-8 -*-
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
import h5py
import numpy as np
import os
import pandas as pd
from scipy.spatial import cKDTree
from warnings import warn


class ExtractNSRDB:
    """Utility class to manage NSRDB data extraction for subsets."""

    IGNORE_LIST = ('meta', 'time_index', 'config', 'stats')

    def __init__(self, target, source):
        """
        Parameters
        ----------
        target : str
            Target file (with path) to dump extracted data to (.csv for simple
            meta data extractions or .h5 for nsrdb data extractions).
        source : str
            Source NSRDB file (with path). Data is extracted from this file and
            written to the target file.
        """

        if os.path.exists(target):
            warn('Target file already exists: "{}"'.format(target))
            if os.path.getsize(target) > 1e9:
                raise IOError('Refuse to write to target file; the file '
                              'already exists and is large: {}'.format(target))

        self.target = target
        self.source = source

    def extract_map(self, dset, time_index=0, sort=False):
        """Extract a lat-lon-data csv for one timestep and all sites for
        mapping applications.

        Parameters
        ----------
        dset : str
            Target dataset in source h5 file to extract data from.
        time_index : int
            Time series index to extract. Data from all sites for this single
            time index will be extracted.
        sort : bool
            Flag on whether to sort the data by lat/lon.
        """

        if not self.target.endswith('.csv'):
            raise TypeError('Target must be .csv for map data extraction.')

        df = self.meta.loc[:, ['latitude', 'longitude']]

        with h5py.File(self.source, 'r') as f:
            print('Extracting "{}" from {}.'.format(dset, self.source))
            df[dset] = f[dset][time_index, :]

        if sort:
            df = df.sort_values(by=['latitude', 'longitude'])

        print('Writing data to "{}"'.format(self.target))
        df.to_csv(self.target)

    def extract_dsets(self, dsets):
        """Extract entire datasets with meta from h5 to new h5.

        Parameters
        ----------
        dsets : list | tuple
            Target datasets in source h5 file to extract data from.
        """

        if not self.target.endswith('.h5'):
            raise TypeError('Target must be .h5 for site data extraction.')

        with h5py.File(self.target, 'w-') as t:
            with h5py.File(self.source, 'r') as s:
                for dset in dsets:
                    if dset not in s:
                        raise KeyError('Could not find dataset "{}" in {}'
                                       .format(dset, self.source))
                for dset in dsets:
                    print('Copying "{}" from {} to {}'
                          .format(dset, self.source, self.target))
                    chunks = None
                    if hasattr(s[dset], 'chunks'):
                        chunks = s[dset].chunks
                    t.create_dataset(dset, data=s[dset][...],
                                     shape=s[dset].shape,
                                     dtype=s[dset].dtype,
                                     chunks=chunks)

                    if hasattr(s[dset], 'attrs'):
                        attrs = dict(s[dset].attrs)
                        for k, v in attrs.items():
                            t[dset].attrs[k] = v

                t.create_dataset('meta', data=s['meta'][...])
                t.create_dataset('time_index', data=s['time_index'][...])

    def extract_sites(self, sites=range(100)):
        """Extract data from h5 for given site indices and write to new h5.

        Parameters
        ----------
        sites : range | list | slice
            Site indicies to extract.
        """
        n_sites = len(list(sites))

        if not self.target.endswith('.h5'):
            raise TypeError('Target must be .h5 for site data extraction.')

        with h5py.File(self.target, 'w-') as t:
            with h5py.File(self.source, 'r') as s:
                for dset in s.keys():
                    if dset not in self.IGNORE_LIST:
                        print(dset)
                        dset_shape = s[dset].shape

                        if len(dset_shape) > 1:
                            t.create_dataset(dset, data=s[dset][:, sites],
                                             shape=(dset_shape[0], n_sites),
                                             dtype=s[dset].dtype)
                        else:
                            t.create_dataset(dset, data=s[dset][sites],
                                             shape=(n_sites, ),
                                             dtype=s[dset].dtype)

                        for attr in s[dset].attrs.keys():
                            t[dset].attrs[attr] = s[dset].attrs[attr]

                t.create_dataset('meta', data=s['meta'][sites])
                t.create_dataset('time_index', data=s['time_index'][...])

                for site_meta in s['meta'][sites]:
                    print(site_meta)

    def extract_closest_meta(self, coords):
        """Get NSRDB meta data for pixels closest to input coordinate set.

        Parameters
        ----------
        coords : np.ndarray
            N x 2 array of lat/lon pairs.

        Returns
        -------
        subset_meta : pd.DataFrame
            A subset of the source meta data with the closest sites to the
            input coordinates. Has length N (length of coordinate array).
        """
        # pylint: disable=not-callable
        tree = cKDTree(self.meta[['latitude', 'longitude']].values)
        ind = tree.query(coords)[1]
        return self.meta.iloc[ind, :]

    @property
    def meta(self):
        """Get the NSRDB meta data as a DataFrame."""
        with h5py.File(self.source, 'r') as s:
            meta = pd.DataFrame(s['meta'][...])
        return meta

    def meta_to_disk(self):
        """Retrieve the NSRDB meta data and save to csv."""
        self.meta.to_csv(self.target)

    def filter_meta(self, values, label):
        """Return a meta df filtered where the label is equal to the value.

        Parameters
        ----------
        values : str | int | float | list
            Search variable(s). Could be a country, state, population, etc...
        label : str
            Meta data column label corresponding to the value.

        Returns
        -------
        meta : pd.DataFrame
            Filtered meta data.
        """

        if isinstance(values, str):
            values = values.encode()
        if isinstance(values, list):
            if isinstance(values[0], str):
                values = [v.encode() for v in values]
        if not isinstance(values, list):
            values = [values]
        return self.meta.loc[self.meta[label].isin(values), :]


class ExtractPuertoRico(ExtractNSRDB):
    """Extraction utilities for Puerto Rico data."""

    @classmethod
    def puerto_rico_vi_meta(cls):
        """Extract the Puerto Rico with USVI and BVI meta data to csv."""
        target = 'pr_nsrdb_meta.csv'
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1', 'nsrdb_2015.h5')

        ex = cls(target, source)

        val = ['Puerto Rico', 'U.S. Virgin Is.', 'British Virgin Is.']
        label = 'country'

        df = ex.filter_meta(val, label)
        df.to_csv(target)

    @classmethod
    def puerto_rico_vi_data(cls, year=2015):
        """Extract the Puerto Rico with USVI and BVI NSRDB data for a given
        year to target h5."""
        target = 'pr_nsrdb_{}.h5'.format(year)
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))

        ex = cls(target, source)
        val = ['Puerto Rico', 'U.S. Virgin Is.', 'British Virgin Is.']
        label = 'country'
        df = ex.filter_meta(val, label)
        sites = list(df.index.values)
        ex.extract_sites(sites=sites)

    @classmethod
    def puerto_rico_meta(cls):
        """Extract the Puerto Rico meta data to csv."""
        target = 'pr_nsrdb_meta.csv'
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1', 'nsrdb_2015.h5')

        ex = cls(target, source)

        val = 'Puerto Rico'
        label = 'country'

        df = ex.filter_meta(val, label)
        df.to_csv(target)

    @classmethod
    def puerto_rico_data(cls, year=2015):
        """Extract the Puerto Rico NSRDB data for a given year to target h5."""
        target = 'pr_nsrdb_{}.h5'.format(year)
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))

        ex = cls(target, source)
        val = 'Puerto Rico'
        label = 'country'
        df = ex.filter_meta(val, label)
        sites = list(df.index.values)
        ex.extract_sites(sites=sites)

    @classmethod
    def puerto_rico_data_50(cls, year=2015):
        """Extract 50 sites in PR for a given year to target h5."""
        target = 'pr_50_nsrdb_{}.h5'.format(year)
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))

        ex = cls(target, source)
        val = 'Puerto Rico'
        label = 'country'
        df = ex.filter_meta(val, label)
        sites = list(df.index.values)[0:50]
        ex.extract_sites(sites=sites)

    @classmethod
    def extract_gen_meta(cls, target, source):
        """Extract PR solar site data."""

        ex = cls(target, source)
        coords = np.array(((17.947249, -66.157321),  # ilumina
                           (18.412413, -65.903370),  # san fermin
                           (17.979177, -66.220556),  # horizon
                           (18.474486, -67.047259),  # oriana
                           ))
        subset_meta = ex.extract_closest_meta(coords)
        subset_meta = subset_meta.sort_index()
        if not os.path.exists(target) and target.endswith('.csv'):
            subset_meta.to_csv(target)
        else:
            raise IOError('Cannot write to: {}'.format(target))


class ExtractValidationData(ExtractNSRDB):
    """Extraction utilities for NSRDB validation ground-measurement sites."""

    # static SURFRAD validation site coordinates [lat,lon].
    COORDS = np.array(((40.052, -88.373),  # Bondville, Illinois
                       (40.125, -105.237),  # Table Mountain, Boulder, CO
                       (36.624, -116.019),  # Desert Rock, Nevada
                       (48.308, -105.102),  # Fort Peck, Montana
                       (34.255, -89.873),  # Goodwin Creek, Mississippi
                       (40.720, -77.931),  # Penn. State Univ., Pennsylvania
                       (43.734, -96.623),  # Sioux Falls, South Dakota
                       (36.604, -97.485),  # ARM Southern Great Plains, OK
                       (39.742, -105.180),  # SRRL-NREL
                       ))

    @classmethod
    def save_meta(cls, target,
                  source='/projects/PXS/nsrdb/v3.0.1/nsrdb_2017.h5'):
        """Save the meta data for the validation ground-measurement sites."""
        ex = cls(target, source)
        subset_meta = ex.extract_closest_meta(cls.COORDS)
        subset_meta.to_csv(target)

    def extract_sites(self):
        """Extract validation data to target h5."""
        subset_meta = self.extract_closest_meta(self.COORDS)
        subset_meta = subset_meta.sort_index()
        sites = list(subset_meta.index.values)
        super().extract_sites(sites=sites)


class ExtractTestData(ExtractNSRDB):
    """Extraction utilities for miscellaneous NSRDB test data sets."""

    @classmethod
    def oregon_50(cls, dir_out, year=2015):
        """Extract NSRDB data from 50 sites from oregon to target h5."""
        # Random sites in Oregon
        target = os.path.join(dir_out, 'test_data_{}.h5'.format(year))
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))

        ex = cls(target, source)
        ex.extract_sites(sites=range(200050, 200100))

    @classmethod
    def srrl_2017(cls, dir_out, year=2017):
        """Extract NSRDB data from NREL SRRL site to target h5."""
        # Site 145809 is close to NREL
        target = os.path.join(dir_out,
                              'test_data_NREL_{}.h5'.format(year))
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))
        ex = cls(target, source)
        ex.extract_sites(sites=range(145809, 145810))

    @classmethod
    def copy_nsrdb_test_file(cls, dsets, year=1998):
        """Make a copy NSRDB test file with certain dsets"""
        target = '/scratch/gbuster/nsrdb_test_data/nsrdb_{}.h5'.format(year)
        source = '/projects/PXS/nsrdb/v3.0.1/nsrdb_{}.h5'.format(year)
        ex = cls(target, source)
        ex.extract_dsets(dsets)
