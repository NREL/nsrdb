"""Classes to handle NSRSDB h5 output files."""

import importlib
import json
import logging
import os
import sys

import farms
import netCDF4
import numpy as np
import rest2
from rex.outputs import Outputs as RexOutputs
from rex.rechunk_h5.chunk_size import ArrayChunkSize
from rex.utilities.loggers import create_dirs

from nsrdb import __version__

VERSION_RECORD = {
    'nsrdb': __version__,
    'farms': farms.__version__,
    'rest2': rest2.__version__,
    'netCDF4': netCDF4.__version__,
    'python': sys.version,
}

spec = importlib.util.find_spec('mlclouds')
if spec is not None:
    VERSION_RECORD['mlclouds'] = getattr(
        importlib.util.module_from_spec(spec), '__version__', '0.0.1'
    )

logger = logging.getLogger(__name__)


class Outputs(RexOutputs):
    """Base class to handle NSRDB output data in .h5 format"""

    def set_version_attr(self):
        """Set the version attribute to the h5 file."""
        self.h5.attrs['version'] = __version__
        self.h5.attrs['version_record'] = json.dumps(VERSION_RECORD)
        self.h5.attrs['package'] = 'nsrdb'

    @classmethod
    def init_h5(
        cls,
        fout,
        dsets,
        attrs,
        chunks,
        dtypes,
        time_index,
        meta,
        add_coords=False,
        mode='w-',
    ):
        """Initialize a full h5 output file with the final intended shape.

        Parameters
        ----------
        fout : str
            Full output filepath.
        dsets : list
            List of dataset name strings.
        attrs : dict
            Dictionary of dataset attributes.
        chunks : dict
            Dictionary of chunk tuples corresponding to each dataset.
        dtypes : dict
            dictionary of numpy datatypes corresponding to each dataset.
        time_index : pd.datetimeindex
            Full pandas datetime index.
        meta : pd.DataFrame
            Full meta data.
        mode : str
            Outputs write mode. w- will raise error if fout exists. w will
            overwrite file.
        add_coords: bool
            Option to include coordinates in output
        """

        meta_chunks = ArrayChunkSize.compute(meta.to_numpy())
        chunks['meta'] = meta_chunks

        if not os.path.exists(os.path.dirname(fout)):
            create_dirs(os.path.dirname(fout))

        logger.info('Initializing output file: {}'.format(fout))

        with cls(fout, mode=mode) as f:
            f['time_index'] = time_index
            f['meta'] = meta

            if add_coords:
                coords = meta[['latitude', 'longitude']].to_numpy()
                coords = coords.astype(np.float32)
                coords_chunks = ArrayChunkSize.compute(coords)
                chunks['coordinates'] = coords_chunks
                f._create_dset(
                    'coordinates',
                    coords.shape,
                    np.float32,
                    chunks=coords_chunks,
                    data=coords,
                )

            shape = (len(time_index), len(meta))

            for dset in dsets:
                # initialize each dset to disk
                f._create_dset(
                    dset,
                    shape,
                    dtypes[dset],
                    chunks=chunks[dset],
                    attrs=attrs[dset],
                    data=None,
                )

        logger.info('{} is complete'.format(fout))
