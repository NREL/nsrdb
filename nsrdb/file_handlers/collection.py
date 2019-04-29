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

    Filename requirements:
     - Expects file names with leading "YYYYMMDD_".
     - Must have var in the file name.
     - Should end with ".h5"

    Parameters
    ----------
    d : str
        Directory to get file list from.
    var : str
        Variable name that is searched for in files in d.

    Returns
    -------
    flist : list
        List of .h5 files in directory d that contain the var string. Sorted
        by integer before the first underscore in the filename.
    """

    flist = os.listdir(d)
    flist = [f for f in flist if '.h5' in f and var in f]
    flist = sorted(flist, key=lambda x: int(x.split('_')[0]))

    return flist


def collect_daily_files(f_dir, f_out, dsets):
    """Collect files from a dir to one output file.

    Parameters
    ----------
    f_dir : str
        Directory of chunked files. Each file should be one variable for one
        day.
    f_out : str
        File path of final output file.
    dsets : list
        List of datasets / variable names to collect.
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

            # pylint: disable-msg=C0121
            r_loc = np.where(time_index.isin(f_ti) == True)[0]  # noqa: E712
            c_loc = np.where(
                meta.index.isin(f_meta.index) == True)[0]  # noqa: E712
            r_loc = slice(np.min(r_loc), np.max(r_loc) + 1)
            c_loc = slice(np.min(c_loc), np.max(c_loc) + 1)

            with Outputs(f_out, mode='a') as f:
                f[dset, r_loc, c_loc] = f_data
