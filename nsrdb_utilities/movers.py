# -*- coding: utf-8 -*-
"""NSRDB data mover and manipulation utilities.

@author: gbuster
"""

import logging
import numpy as np
import pandas as pd
import os
import h5py
import time
import shlex
from subprocess import Popen
from warnings import warn


logger = logging.getLogger(__name__)


def get_meta_df(fname):
    """Get the meta dataframe from fname."""
    with h5py.File(fname) as f:
        meta = pd.DataFrame(f['meta'][...])
    return meta


def get_dset_dtype(fname, dset):
    """Get the dset data type from fname."""
    with h5py.File(fname) as f:
        dtype = f[dset].dtype
    return dtype


def get_dset_shape(fname, dset):
    """Get the dset shape from fname."""
    with h5py.File(fname) as f:
        shape = f[dset].shape
    return shape


def get_dset_attrs(fname, dset):
    """Get the dset attribute dictionary from fname."""
    with h5py.File(fname) as f:
        attrs = dict(f[dset].attrs)
    return attrs


def get_dset_list(fname):
    """Get the list of datasets in target fname .h5 file."""
    with h5py.File(fname, 'r') as f:
        keys = list(f.keys())
    return keys


def pull_data(fname, dset, slc):
    """Get the unscaled data from the target dataset in fname."""
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
                logger.info('\n=====================\n', fname)
                dsets = get_dset_list(fname1)

                for dset in dsets:
                    if dset not in ignore:
                        logger.info(dset)

                        d1 = pull_data(fname1, dset, slc)
                        d2 = pull_data(fname2, dset, slc)
                        delta = d1 - d2
                        del_per_num = np.sum(delta) / (d1.shape[0] *
                                                       d1.shape[1])

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


def repack_h5(f_orig, f_new, dir_orig, dir_new=None, inplace=True):
    """Repack an h5 file potentially decreasing its memory footprint.

    Parameters
    ----------
    f_orig : str
        Source/target h5 file (without path) to repack.
    f_new : str
        Intended destination h5 file. If inplace is specified, this can be
        "temp.h5".
    dir_orig : str
        Source directory containing f_orig.
    dir_new : str | NoneType
        Target directory that the newly repacked file will be located in.
        If this is None or inplace is requested, this will be the same as
        dir_orig.
    inplace : bool
        If repacking inplace is requested, the final h5 file will have the same
        name and location as the original file. The original file will be
        removed.
    """

    if dir_new is None or inplace is True:
        dir_new = dir_orig

    if dir_orig == dir_new and f_new == f_orig:
        # protect against repacking to the same location
        # (might cause error, unsure)
        f_new = f_new.replace('.h5', '_repacked.h5')

    f_orig = os.path.join(dir_orig, f_orig)
    f_new = os.path.join(dir_new, f_new)

    # Repack to new file and rename
    t1 = time.time()
    cmd = 'h5repack -i {i} -o {o}'.format(i=f_orig, o=f_new)
    cmd = shlex.split(cmd)
    logger.info('Submitting the following cmd as a subprocess:\n\t{}'
                .format(cmd))

    # use subprocess to submit command and wait until it is done
    process = Popen(cmd)
    process.wait()

    if inplace:
        # remove the original file and rename the newly packed file
        os.remove(f_orig)
        os.rename(f_new, f_orig)

    min_elapsed = (time.time() - t1) / 60
    logger.info('Finished repacking {} to {}. Time elapsed: {0:.2f} minutes.'
                .format(f_orig, f_new, min_elapsed))


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
                    target[dset][:, start:end] = (source[dset][:, start:end] /
                                                  old_scale *
                                                  new_scale_factor)
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


def update_dset(source_f, target_f, dsets):
    """Update the datasets in target_f with the data from source_f.

    Note that this also updates the dataset attributes but not the shape,
    chunks, or dtype. Furthermore, this method is scaling-agnostic, such that
    the data from source is written to target without unscaling/rescaling
    (source data must be properly scaled with corresponding attributes).

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
    """

    source_meta = get_meta_df(source_f)
    target_meta = get_meta_df(target_f)

    if (all(source_meta['latitude'] != target_meta['latitude']) or
            all(source_meta['longitude'] != target_meta['longitude'])):
        raise ValueError('Meta data coordinate arrays do not match between '
                         '{} and {}. Data updating should not be performed.'
                         .format(source_f, target_f))

    # check datasets present in files
    for f in [source_f, target_f]:
        for dset in dsets:
            if dset not in f:
                raise KeyError('Dataset "{}" not found in {}'.format(dset, f))

    # check dataset dtypes in files
    for dset in dsets:
        source_dtype = get_dset_dtype(source_f, dset)
        target_dtype = get_dset_dtype(target_f, dset)
        if source_dtype != target_dtype:
            raise TypeError('Datatype of dataset "{}" does not match between '
                            '{} and {}. Respective dtypes are: {} and {}'
                            .format(dset, source_f, target_f,
                                    source_dtype, target_dtype))

        source_shape = get_dset_shape(source_f, dset)
        target_shape = get_dset_shape(target_f, dset)
        if source_shape != target_shape:
            raise ValueError('Shapes of dataset "{}" does not match between '
                             '{} and {}. Respective shapes are: {} and {}'
                             .format(dset, source_f, target_f,
                                     source_shape, target_shape))

        # dataset dtypes match, proceed.
        t1 = time.time()
        with h5py.File(target_f, 'a') as target:
            # overwrite with new attributes.
            target[dset].attrs = get_dset_attrs(source_f, dset)

            with h5py.File(source_f, 'r') as source:

                end = 0
                chunk = 10000

                for i in range(0, 300):
                    start = end
                    end = np.min([start + chunk, source_shape[1]])
                    target[dset][:, start:end] = source[dset][:, start:end]
                    min_elapsed = (time.time() - t1) / 60
                    logger.info('Rewrote {0} for {1} through {2} (chunk #{3}).'
                                ' Time elapsed: {4:.2f} minutes.'
                                .format(dset, start, end, i, min_elapsed))

                    if end == source_shape[1]:
                        logger.info('Reached end of dataset "{}" (dataset '
                                    'column index {} and dataset shape is {})'
                                    .format(dset, end, source_shape))
                        break
