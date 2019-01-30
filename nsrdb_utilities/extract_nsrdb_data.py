# -*- coding: utf-8 -*-
"""Test data extraction.

Created on Tue Dec  10 08:22:26 2018

@author: gbuster
"""
import h5py
import os
import pandas as pd

from downscale import __testdatadir__


class ExtractNSRDB:
    """Utility class to manage NSRDB data extraction for subsets."""

    IGNORE_LIST = ('meta', 'time_index', 'config', 'stats')

    def __init__(self, target, source):
        """Initialize an extraction utility.

        Parameters
        ----------
        target : str
            Target file (.csv for simple meta data extractions or .h5 for
            nsrdb data extractions)
        source : str
            Source NSRDB file with path.
        """

        self.target = target
        self.source = source

    def extract(self, sites=range(100)):
        """Extract data from h5 for given site indices and write to new h5.

        Parameters
        ----------
        sites : range | list | slice
            Site indicies to extract.
        """
        n_sites = len(list(sites))

        with h5py.File(self.target, 'w') as t:
            with h5py.File(self.source, 'r') as s:
                for dset in s.keys():
                    if dset not in self.IGNORE_LIST:
                        print(dset)
                        print(s[dset].shape)
                        n_time = s[dset].shape[0]
                        t.create_dataset(dset, data=s[dset][:, sites],
                                         shape=(n_time, n_sites),
                                         dtype=s[dset].dtype)
                        print(t[dset].shape)
                        print(s[dset][0:5, 0:10])
                        print(t[dset][0:5, 0:10])
                        for attr in s[dset].attrs.keys():
                            t[dset].attrs[attr] = s[dset].attrs[attr]
                            print(attr,
                                  t[dset].attrs[attr],
                                  t[dset].attrs[attr])

                t.create_dataset('meta', data=s['meta'][sites])
                t.create_dataset('time_index', data=s['time_index'][...])

                for site_meta in s['meta'][sites]:
                    print(site_meta)

    @property
    def meta(self):
        """Get the NSRDB meta data as a DataFrame."""
        with h5py.File(self.source, 'r') as s:
            meta = pd.DataFrame(s['meta'][...])
        return meta

    def meta_to_disk(self):
        """Retrieve the NSRDB meta data and save to csv."""
        self.meta.to_csv(self.target)

    def filter_meta(self, value, label):
        """Return a meta df filtered where the label is equal to the value.

        Parameters
        ----------
        value : str | int | float
            Search variable. Could be a country, state, population, etc...
        label : str
            Meta data column label corresponding to the value.

        Returns
        -------
        meta : pd.DataFrame
            Filtered meta data.
        """
        if isinstance(value, str):
            value = value.encode()
        return self.meta.loc[(self.meta[label] == value), :]

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
        ex.extract(sites=sites)

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
        ex.extract(sites=sites)

    @classmethod
    def oregon_50(cls, year=2015):
        """Extract NSRDB data from 50 sites from oregon to target h5."""
        # Random sites in Oregon
        target = os.path.join(__testdatadir__, 'test_data_{}.h5'.format(year))
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))

        ex = cls(target, source)
        ex.extract(sites=range(200050, 200100))

    @classmethod
    def srrl_2017(cls, year=2017):
        """Extract NSRDB data from NREL SRRL site to target h5."""
        # Site 145809 is close to NREL
        target = os.path.join(__testdatadir__,
                              'test_data_NREL_{}.h5'.format(year))
        source = os.path.join('/projects/PXS/nsrdb/v3.0.1',
                              'nsrdb_{}.h5'.format(year))
        ex = cls(target, source)
        ex.extract(sites=range(145809, 145810))
