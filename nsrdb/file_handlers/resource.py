"""
Classes to handle reading NSRDB resource data.
"""
from rex.resource import ResourceDataset
from rex.resource import Resource as rexResource
from rex.multi_file_resource import MultiFileResource as rexMultiFileResource
from rex.utilities.exceptions import ResourceKeyError
from rex.utilities.parse_keys import parse_slice


class Resource(rexResource):
    """
    Base class to handle NSRDB .h5 files with handling of legacy NSRDB
    scale factors
    """

    SCALE_ATTR = 'scale_factor'
    UNIT_ATTR = 'units'
    ADD_ATTR = 'add_offset'

    OLD_SCALE_ATTR = 'psm_scale_factor'
    OLD_UNIT_ATTR = 'psm_units'
    OLD_ADD_ATTR = 'psm_add_offset'

    def _get_ds(self, ds_name, ds_slice):
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
        if ds_name not in self.datasets:
            raise ResourceKeyError('{} not in {}'
                                   .format(ds_name, self.datasets))

        ds = self.h5[ds_name]
        ds_slice = parse_slice(ds_slice)
        attrs = dict(ds.attrs)

        scale_attr = self.OLD_SCALE_ATTR
        if self.SCALE_ATTR in attrs:
            scale_attr = self.SCALE_ATTR

        add_attr = self.OLD_ADD_ATTR
        if self.ADD_ATTR in attrs:
            add_attr = self.ADD_ATTR

        out = ResourceDataset.extract(ds, ds_slice, scale_attr=scale_attr,
                                      add_attr=add_attr,
                                      unscale=self._unscale)

        if ds_name == 'cloud_type':
            out = out.astype('int8')
        else:
            out = out.astype('float32')

        return out


class MultiFileResource(rexMultiFileResource, Resource):
    """Multi file resource handler with handling of legacy nsrdb scale factors
    """
