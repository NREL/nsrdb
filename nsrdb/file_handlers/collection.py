# -*- coding: utf-8 -*-
"""NSRDB chunked file collection tools.
"""
import numpy as np
import os
import logging
from nsrdb.file_handlers.outputs import Outputs


logger = logging.getLogger(__name__)


def get_flist(d, var):
    """Get a date-sorted .h5 file list for a given var.

    Expects files with format "YYYYMMDD_var.h5"

    Parameters
    Returns
    """

    flist = os.listdir(d)
    flist = [f for f in flist if '.h5' in f and var in f]
    flist = sorted(flist, key=lambda x: int(x.split('_')[0]))

    return flist


def collect(f_dir, f_out, dsets):
    """Collect files from a dir to one output file.
    """
    logger.info('Collecting data from {} to {}'.format(f_dir, f_out))

    with Outputs(f_out, mode='r') as f:
        time_index = f.time_index
        meta = f.meta

    for dset in dsets:
        flist = get_flist(f_dir, dset)
        for fname in flist:
            fpath = os.path.join(f_dir, fname)

            with Outputs(fpath, unscale=False, mode='r') as f:
                logger.debug('Collecting data from {}'.format(fpath))
                f_ti = f.time_index
                f_meta = f.meta
                f_data = f[dset][...]

            r_loc = np.where(time_index == f_ti)[0]
            c_loc = np.where(meta == f_meta)[0]

            with Outputs(f_out, mode='a') as f:
                f[dset, r_loc, c_loc] = f_data
