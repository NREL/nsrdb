# -*- coding: utf-8 -*-
"""NSRDB data mover and manipulation utilities.

@author: gbuster
"""
import logging
import os
import time
from warnings import warn

import h5py
import numpy as np
import pandas as pd
from rex.utilities.hpc import PBS, SLURM
from rex.utilities.loggers import init_logger

from nsrdb.utilities.file_utils import repack_h5

logger = logging.getLogger(__name__)


def get_meta_df(fname):
    """Get the meta dataframe from fname."""
    with h5py.File(fname, 'r') as f:
        meta = pd.DataFrame(f['meta'][...])
    return meta


def get_dset_dtype(fname, dset):
    """Get the dset data type from fname."""
    with h5py.File(fname, 'r') as f:
        dtype = f[dset].dtype
    return dtype


def get_dset_shape(fname, dset):
    """Get the dset shape from fname."""
    with h5py.File(fname, 'r') as f:
        shape = f[dset].shape
    return shape


def get_dset_attrs(fname, dset):
    """Get the dset attribute dictionary from fname."""
    with h5py.File(fname, 'r') as f:
        attrs = dict(f[dset].attrs)
    return attrs


def get_dset_list(fname):
    """Get the list of datasets in target fname .h5 file."""
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
    return keys


def pull_data(fname, dset, slc=(slice(0, 8760), slice(0, 100))):
    """Get the unscaled data from the target dataset in fname.

    Parameters
    ----------
    fname : str
        file name with path to pull data from.
    dset : str
        Target dataset to pull data from.
    slc : slice | list | tuple
        Slices of the datasets to compare. For resource data this should be a
        2 slice entry list/tuple.
    """
    with h5py.File(fname, 'r') as f:
        data = f[dset][slc]
        if 'psm_scale_factor' in f[dset].attrs:
            data = data / f[dset].attrs['psm_scale_factor']
    return data


def check_dsets(dir1, dir2, slc=(slice(0, 8760), slice(0, 100)),
                ignore=('config', 'stats', 'meta', 'time_index')):
    """Verify that dsets have the same contents for similar files in two dirs.

    Parameters
    ----------
    dir1/dir2 : str
        Two directories that have h5 files with the same names. h5 files in
        dir1 will be searched for in dir2. Matching filenames will be compared.
    slc : slice | list | tuple
        Slices of the datasets to compare. For resource data this should be a
        2 slice entry list/tuple.
    ignore : list | tuple
        Datasets to ignore.
    """

    # initialize a logger to the stdout
    init_logger(__name__, log_file=None, log_level='INFO')

    flist = os.listdir(dir1)

    for fname in flist:
        if fname.endswith('.h5'):
            fname1 = os.path.join(dir1, fname)
            fname2 = os.path.join(dir2, fname)
            for f in [fname1, fname2]:
                if not os.path.exists(f):
                    warn('!!! Warning, does not exist: {}'
                         .format(f))
        else:
            logger.info('\n=====================\n{}'.format(fname))
            dsets = get_dset_list(fname1)

            for dset in dsets:
                if dset not in ignore:
                    logger.info(dset)

                    d1 = pull_data(fname1, dset, slc)
                    d2 = pull_data(fname2, dset, slc)
                    delta = d1 - d2
                    del_per_num = (np.sum(delta)
                                   / (d1.shape[0] * d1.shape[1]))

                    logger.info(del_per_num)

                    if np.abs(del_per_num) > 1:
                        raise ValueError('Error found in {} in {}. '
                                         .format(dset, fname1))
                    else:
                        logger.info('Dataset "{}" is same between {} and '
                                    '{}'.format(dset, fname1, fname2))


def interrogate_dset(fname, dset):
    """Interrogate and log dataset information.

    Parameters
    ----------
    fname : str
        h5 filename (with path) to interrogate.
    dset : str
        Target dataset to interrogate.
    """
    with h5py.File(fname, 'r') as f:
        logger.info('\nSource "{}" dataset in {}\n'.format(dset, fname))
        logger.info(f[dset][0:10, 0:10])
        x = f[dset][0, 0]
        logger.info(x)
        logger.info(f[dset].dtype)
        shape = f[dset].shape
        logger.info(shape)
        logger.info(f[dset].chunks)
        logger.info(dict(f[dset].attrs))


def change_dtypes(source_f, target_f, source_dir, target_dir, dsets,
                  new_scale_factor=1000, new_dtype=np.int16):
    """Take datasets from source and write to target with new scale and dtype.

    Note that the source data will be unscaled then re-scaled.

    Parameters
    ----------
    source_f : str
        Source h5 file in source_dir with dsets and correct data
        (dtype doesnt matter).
    target_f : str
        Target file in target_dir that will contain the final re-typed datasets
    source_dir : str
        Location of the source file.
    target_dir : str
        Location of the target file.
    dsets : list | tuple
        Datasets to re-type.
    new_scale_factor : int | float
        New scale factor to apply to the dsets in the target file.
    new_dtype : np.dtype
        New data type to apply to the dsets in the target file.
    """

    # initialize a logger to the stdout
    init_logger(__name__, log_file=None, log_level='INFO')

    t1 = time.time()
    logger.info('Running re_write_dtypes for {}'.format(source_f))
    source_f = os.path.join(source_dir, source_f)

    with h5py.File(source_f, 'r') as source:
        with h5py.File(os.path.join(target_dir, target_f), 'a') as target:

            for dset in dsets:

                interrogate_dset(source_f, dset)
                shape = source[dset].shape

                if dset in target:
                    logger.info('Deleting {} from {}'.format(dset, target_f))
                    del target[dset]
                target.create_dataset(dset,
                                      shape=shape,
                                      dtype=new_dtype,
                                      chunks=source[dset].chunks)
                target[dset].attrs['psm_scale_factor'] = new_scale_factor
                target[dset].attrs['psm_units'] = \
                    source[dset].attrs['psm_units']

                # get the old scale factor to unscale then rescale the data.
                old_scale = 1
                if 'psm_scale_factor' in source[dset].attrs:
                    old_scale = source[dset].attrs['psm_scale_factor']

                end = 0
                chunk = 10000

                for i in range(0, 300):
                    start = end
                    end = np.min([start + chunk, shape[1]])
                    # make sure to unscale and re-scale the target data.
                    target[dset][:, start:end] = (source[dset][:, start:end]
                                                  / old_scale
                                                  * new_scale_factor)
                    min_elapsed = (time.time() - t1) / 60
                    logger.info('Rewrote {0} for {1} through {2} (chunk #{3}).'
                                ' Time elapsed: {4:.2f} minutes.'
                                .format(dset, start, end, i, min_elapsed))

                    if end == shape[1]:
                        break

                interrogate_dset(os.path.join(target_dir, target_f), dset)

    new_f = target_f.replace('.h5', '_repacked.h5')
    repack_h5(target_f, new_f, target_dir)

    for dset in dsets:
        interrogate_dset(os.path.join(target_dir, target_f), dset)


def update_dset(source_f, target_f, dsets, start=0):
    """Update the datasets in target_f with the data from source_f.

    Note that this also updates the dataset attributes but not the shape,
    chunks, or dtype. Furthermore, this method is scaling-agnostic, such that
    the data from source is written to target without unscaling/rescaling
    (source data must be properly scaled with corresponding attributes).

    Reference time durations (great variability on Peregrine):
        One dataset w shape (17568, 2018392), int16, batch-h, took 37 hours
        One dataset w shape (17520, 2018392), int8, batch-h, took 12 hours

    Parameters
    ----------
    source_f : str
        Source h5 file (with path) with dsets and correct data. This must
        have the same meta lat/lon array as target_f.
    target_f : str
        Target file (with path) that will contain the final updated datasets.
        This must have the same meta lat/lon array as source_f.
    dsets : list | tuple
        Datasets to update. Must be present in both files, have the same dtype
        and shape.
    start : int
        Starting column index for dset update (column indicies equal to or
        greater than this value will be updated).
    """

    # initialize a logger to the stdout
    init_logger(__name__, log_file=None, log_level='INFO')

    logger.info('Updating dsets "{}" from source "{}" to target "{}" at '
                'starting index {}.'
                .format(dsets, source_f, target_f, start))
    t0 = time.time()

    source_meta = get_meta_df(source_f)
    target_meta = get_meta_df(target_f)

    check = (all(source_meta['latitude'] != target_meta['latitude'])
             or all(source_meta['longitude'] != target_meta['longitude']))
    if check:
        raise ValueError('Meta data coordinate arrays do not match between '
                         '{} and {}. Data updating should not be performed.'
                         .format(source_f, target_f))
    else:
        logger.info('Lat/lon meta data test passed.')

    # check datasets present in files
    for f in [source_f, target_f]:
        with h5py.File(f, 'r') as fhandler:
            for dset in dsets:
                if dset not in list(fhandler):
                    raise KeyError('Dataset "{}" not found in {}. Contents: {}'
                                   .format(dset, f, list(fhandler)))
                else:
                    logger.info('Dataset "{}" present in both files.'
                                .format(dset))

    # check dataset dtypes in files
    for dset in dsets:
        source_dtype = get_dset_dtype(source_f, dset)
        target_dtype = get_dset_dtype(target_f, dset)
        if source_dtype != target_dtype:
            raise TypeError('Datatype of dataset "{}" does not match between '
                            '{} and {}. Respective dtypes are: {} and {}'
                            .format(dset, source_f, target_f,
                                    source_dtype, target_dtype))
        else:
            logger.info('Dataset "{}" has same dtype in both files: {}.'
                        .format(dset, source_dtype))

        source_shape = get_dset_shape(source_f, dset)
        target_shape = get_dset_shape(target_f, dset)
        if source_shape != target_shape:
            raise ValueError('Shapes of dataset "{}" does not match between '
                             '{} and {}. Respective shapes are: {} and {}'
                             .format(dset, source_f, target_f,
                                     source_shape, target_shape))
        else:
            logger.info('Dataset "{}" has same shape in both files: {}.'
                        .format(dset, source_shape))

        # dataset dtypes match, proceed.
        t1 = time.time()
        with h5py.File(target_f, 'a') as target:

            # overwrite with new attributes.
            for k in dict(target[dset].attrs).keys():
                logger.info('Deleting attribute "{}" from dset "{}" in {}'
                            .format(k, dset, target_f))
                del target[dset].attrs[k]
            attrs = get_dset_attrs(source_f, dset)
            for k, v in attrs.items():
                logger.info('Setting attribute "{}" in dset "{}" to: {}'
                            .format(k, dset, v))
                target[dset].attrs[k] = v

            with h5py.File(source_f, 'r') as source:

                # number of columns to update at once
                chunk = 10000

                for i in range(0, 10000):
                    end = np.min([start + chunk, source_shape[1]])
                    target[dset][:, start:end] = source[dset][:, start:end]
                    min_elapsed = (time.time() - t1) / 60
                    logger.info('Rewrote "{0}" for {1} through {2} '
                                '(chunk #{3}). Time elapsed: {4:.2f} minutes.'
                                .format(dset, start, end, i, min_elapsed))
                    start = end

                    if end == source_shape[1]:
                        logger.info('Reached end of dataset "{}" (dataset '
                                    'column index {} and dataset shape is {})'
                                    .format(dset, end, source_shape))
                        break
    min_elapsed = (time.time() - t0) / 60
    logger.info('Finished copying datasets from {0} to {1} in {2:.2f} min. '
                'The following datasets were copied: {3}'
                .format(source_f, target_f, min_elapsed, dsets))


def rename_dset(h5_fpath, dset, new_dset):
    """Rename a dataset in an h5 file.

    Parameters
    ----------
    h5_fpath : str
        Filepath to h5 file with dataset to rename.
    dset : str
        Original (current) dataset name.
    new_dset : str
        Desired dataset name.
    """

    with h5py.File(h5_fpath, 'a') as f:
        if dset in list(f):
            f[new_dset] = f[dset]
            del f[dset]


def peregrine(fun_str, arg_str, alloc='pxs', queue='batch-h',
              node_name='mover', stdout_path='/scratch/gbuster/stdout/'):
    """Kick off a peregrine job to execute a mover function.

    Parameters
    ----------
    fun_str : str
        Name of the function in movers.py to execute in the pbs job.
    arg_str : str
        Arguments passed to the target function in the command line call.
        Care must be taken to use proper quotations for string args.
        Example:
            arg_str = ('source_f="source.h5", target_f="target.h5", '
                       'dsets=["dset1"]')
    alloc : str
        PBS project allocation.
    queue: str
        PBS job queue.
    node_name : str
        Name for the PBS job.
    stdout_path : str
        Path to dump the stdout/stderr files.
    """

    cmd = ('python -c '
           '\'from nsrdb.utilities.movers import {fun}; '
           '{fun}({args})\'')

    cmd = cmd.format(fun=fun_str, args=arg_str)

    pbs = PBS(cmd, alloc=alloc, queue=queue, name=node_name,
              stdout_path=stdout_path, feature=None)

    print('\ncmd:\n{}\n'.format(cmd))

    if pbs.id:
        msg = ('Kicked off job "{}" (PBS jobid #{}) on '
               'Peregrine.'.format(node_name, pbs.id))
    else:
        msg = ('Was unable to kick off job "{}". '
               'Please see the stdout error messages'
               .format(node_name))
    print(msg)


def hpc(fun_str, arg_str, alloc='pxs', memory=96,
        walltime=10, node_name='mover',
        stdout_path='//scratch/gbuster/data_movers/'):
    """Kick off an hpc job to execute a mover function.

    Parameters
    ----------
    fun_str : str
        Name of the function in movers.py to execute in the SLURM job.
    arg_str : str
        Arguments passed to the target function in the command line call.
        Care must be taken to use proper quotations for string args.
        Example:
            arg_str = ('source_f="source.h5", target_f="target.h5", '
                       'dsets=["dset1"]')
    alloc : str
        SLURM project allocation.
    memory : int
        Node memory request in GB.
    walltime : int
        Node walltime request in hours.
    node_name : str
        Name for the SLURM job.
    stdout_path : str
        Path to dump the stdout/stderr files.
    """

    cmd = ('python -c '
           '\'from nsrdb.utilities.movers import {fun}; '
           '{fun}({args})\'')

    cmd = cmd.format(fun=fun_str, args=arg_str)

    slurm_manager = SLURM()
    out = slurm_manager.sbatch(cmd,
                               alloc=alloc,
                               memory=memory,
                               walltime=walltime,
                               name=node_name,
                               stdout_path=stdout_path)[0]

    print('\ncmd:\n{}\n'.format(cmd))

    if out:
        msg = ('Kicked off job "{}" (SLURM jobid #{}) on '
               'HPC.'.format(node_name, out))
    else:
        msg = ('Was unable to kick off job "{}". '
               'Please see the stdout error messages'
               .format(node_name))
    print(msg)
