"""
Classes to handle reading NSRDB resource data.
"""
import h5py
import numpy as np
import pandas as pd


def parse_keys(keys):
    """
    Parse keys for complex __getitem__ and __setitem__

    Parameters
    ----------
    keys : string | tuple
        key or key and slice to extract

    Returns
    -------
    key : string
        key to extract
    key_slice : slice | tuple
        Slice or tuple of slices of key to extract
    """
    if isinstance(keys, tuple):
        key = keys[0]
        key_slice = keys[1:]
    else:
        key = keys
        key_slice = (slice(None, None, None),)

    return key, key_slice


class Resource:
    """
    Base class to handle NSRDB .h5 files
    """

    SCALE_ATTR = 'scale_factor'
    UNIT_ATTR = 'units'
    ADD_ATTR = 'add_offset'

    def __init__(self, h5_file, unscale=True, hsds=False):
        """
        Parameters
        ----------
        h5_file : str
            Path to .h5 resource file
        unscale : bool
            Boolean flag to automatically unscale variables on extraction
        hsds : bool
            Boolean flag to use h5pyd to handle .h5 'files' hosted on AWS
            behind HSDS
        """
        self._h5_file = h5_file
        if hsds:
            import h5pyd
            self._h5 = h5pyd.File(self._h5_file, 'r')
        else:
            self._h5 = h5py.File(self._h5_file, 'r')

        self._unscale = unscale
        self._meta = None
        self._time_index = None

    def __repr__(self):
        msg = "{} for {}".format(self.__class__.__name__, self._h5_file)
        return msg

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

        if type is not None:
            raise

    def __len__(self):
        return self._h5['meta'].shape[0]

    def __getitem__(self, keys):
        ds, ds_slice = parse_keys(keys)

        if ds == 'time_index':
            out = self._get_time_index(*ds_slice)
        elif ds == 'meta':
            out = self._get_meta(*ds_slice)
        else:
            out = self._get_ds(ds, *ds_slice)

        return out

    @property
    def dsets(self):
        """
        Datasets available in h5_file

        Returns
        -------
        list
            List of datasets in h5_file
        """
        return list(self._h5)

    @property
    def shape(self):
        """
        Resource shape (timesteps, sites)
        shape = (len(time_index), len(meta))

        Returns
        -------
        shape : tuple
            Shape of resource variable arrays (timesteps, sites)
        """
        _shape = (self._h5['time_index'].shape[0], self._h5['meta'].shape[0])
        return _shape

    @property
    def meta(self):
        """
        Meta data DataFrame

        Returns
        -------
        meta : pandas.DataFrame
            Resource Meta Data
        """
        if self._meta is None:
            self._meta = pd.DataFrame(self._h5['meta'][...])

        return self._meta

    @property
    def time_index(self):
        """
        DatetimeIndex

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Resource datetime index
        """
        if self._time_index is None:
            ti = self._h5['time_index'][...].astype(str)
            self._time_index = pd.to_datetime(ti)

        return self._time_index

    def get_attrs(self, dset=None):
        """
        Get h5 attributes either from file or dataset

        Parameters
        ----------
        dset : str
            Dataset to get attributes for, if None get file (global) attributes

        Returns
        -------
        attrs : dict
            Dataset or file attributes
        """
        if dset is None:
            attrs = dict(self._h5.attrs)
        else:
            attrs = dict(self._h5[dset].attrs)

        return attrs

    def get_dset_properties(self, dset):
        """
        Get dataset properties (shape, dtype, chunks)

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        shape : tuple
            Dataset array shape
        dtype : str
            Dataset array dtype
        chunks : tuple
            Dataset chunk size
        """
        ds = self._h5[dset]
        return ds.shape, ds.dtype, ds.chunks

    def get_scale(self, dset):
        """
        Get dataset scale factor

        Parameters
        ----------
        dset : str
            Dataset to get scale factor for

        Returns
        -------
        float
            Dataset scale factor, used to unscale int values to floats
        """
        return self._h5[dset].attrs.get(self.SCALE_ATTR, 1)

    def get_units(self, dset):
        """
        Get dataset units

        Parameters
        ----------
        dset : str
            Dataset to get units for

        Returns
        -------
        str
            Dataset units, None if not defined
        """
        return self._h5[dset].attrs.get(self.UNIT_ATTR, None)

    def _get_time_index(self, *ds_slice):
        """
        Extract and convert time_index to pandas Datetime Index

        Examples
        --------
        self['time_index', 1]
            - Get a single timestep This returns a Timestamp
        self['time_index', :10]
            - Get the first 10 timesteps
        self['time_index', [1, 3, 5, 7, 9]]
            - Get a list of timesteps

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            tuple describing slice of time_index to extract

        Returns
        -------
        time_index : pandas.DatetimeIndex
            Vector of datetime stamps
        """
        time_index = self._h5['time_index'][ds_slice[0]]
        time_index: np.array
        return pd.to_datetime(time_index.astype(str))

    def _get_meta(self, *ds_slice):
        """
        Extract and convert meta to a pandas DataFrame

        Examples
        --------
        self['meta', 1]
            - Get site 1
        self['meta', :10]
            - Get the first 10 sites
        self['meta', [1, 3, 5, 7, 9]]
            - Get the first 5 odd numbered sites
        self['meta', :, 'timezone']
            - Get timezone for all sites
        self['meta', :, ['latitude', 'longitdue']]
            - Get ('latitude', 'longitude') for all sites

        Parameters
        ----------
        ds_slice : tuple of int | list | slice
            Pandas slicing describing which sites and columns to extract

        Returns
        -------
        meta : pandas.Dataframe
            Dataframe of location meta data
        """
        sites = ds_slice[0]
        if isinstance(sites, int):
            sites = slice(sites, sites + 1)

        meta = self._h5['meta'][sites]

        if isinstance(sites, slice):
            if sites.stop:
                sites = list(range(*sites.indices(sites.stop)))
            else:
                sites = list(range(len(meta)))

        meta = pd.DataFrame(meta, index=sites)
        if len(ds_slice) == 2:
            meta = meta[ds_slice[1]]

        return meta

    def _get_ds(self, ds_name, *ds_slice):
        """
        Extract data from given dataset

        Examples
        --------
        self['dni', :, 1]
            - Get 'dni'timeseries for site 1
        self['dni', ::2, :]
            - Get hourly 'dni' timeseries for all sites (NSRDB)


        Parameters
        ----------
        ds_name : str
            Variable dataset to be extracted
        ds_slice : tuple of int | list | slice
            tuple describing slice of dataset array to extract

        Returns
        -------
        ds : ndarray
            ndarray of variable timeseries data
            If unscale, returned in native units else in scaled units
        """
        if ds_name not in self.dsets:
            raise KeyError('{} not in {}'.format(ds_name, self.dsets))

        ds = self._h5[ds_name]
        out = ds[ds_slice]
        if self._unscale:
            scale_factor = ds.attrs.get(self.SCALE_ATTR, 1)
            add_factor = ds.attrs.get(self.ADD_ATTR, 0)

            if ds_name == 'cloud_type':
                out = out.astype('int8')
            else:
                out = out.astype('float32')

            if add_factor != 0:
                # cloud properties have both scale and offset
                out *= scale_factor
                out += add_factor
            else:
                # most variables have just scale factor
                if scale_factor != 1:
                    out /= scale_factor

        return out

    def close(self):
        """
        Close h5 instance
        """
        self._h5.close()
