# -*- coding: utf-8 -*-
"""
Utility to abstractly handle filesystem operations locally and in the cloud
"""
from warnings import warn

import h5py
import netCDF4 as nc
from cloud_fs import FileSystem

from nsrdb.file_handlers.resource import Resource


class NSRDBFileSystem(FileSystem):
    """
    Custom FileSystem handler for NSRDB with unique logic to open .h5 and .nc
    files in AWS S3
    """

    def __init__(self, path, anon=False, profile=None, use_rex=False,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            S3 object path or file path
        anon : bool, optional
            Whether to use anonymous credentials, by default False
        profile : str, optional
            AWS credentials profile, by default None
        use_rex : bool
            Flag to uss rex/nsrdb Resource as the .h5 handler instead of the
            h5py.
        """
        super().__init__(path, anon=anon, profile=profile, **kwargs)
        self._fs_handler = None
        self._file_handler = None
        self._use_rex = use_rex

    def __enter__(self):
        return self.open()

    def __exit__(self, type, value, traceback):
        self._close()

        if type is not None:
            raise

    @staticmethod
    def _close_handler(handler):
        """
        Close given file handler

        Parameters
        ----------
        handler : object
            File handler, could be s3fs.File, netCDF.Dataset, or h5py.File
        """
        if handler is not None and not isinstance(handler, str):
            try:
                handler.close()
            except Exception as ex:
                warn('Could not close {}: {}'.format(handler, ex))

    def _close(self):
        """
        Close file and filesystem handler
        """
        self._close_handler(self._file_handler)
        self._close_handler(self._fs_handler)

    def open(self):
        """
        open filesystem handler, if file is a .h5 or .nc file also open a
        h5py.File or netCDF4.Dataset handler respectively.

        Returns
        -------
        _file_handler : obj
            Proper file handler, either:
            - local file path
            - open s3fs.File object
            - open h5py.File object
            - open netCDF4.Dataset object
        """

        self._fs_handler = super().open()
        if self.path.endswith(('.nc', '.nc4')):
            # pylint: disable=no-member
            if isinstance(self._fs_handler, str):
                self._file_handler = nc.Dataset(self._fs_handler, mode='r')
            else:
                self._file_handler = nc.Dataset('inmemory.nc', mode='r',
                                                memory=self._fs_handler.read())
        elif self.path.endswith('.h5'):
            if self._use_rex:
                self._file_handler = Resource(self._fs_handler)
            else:
                self._file_handler = h5py.File(self._fs_handler, mode='r')
        else:
            self._file_handler = self._fs_handler

        return self._file_handler
