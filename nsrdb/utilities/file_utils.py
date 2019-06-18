# -*- coding: utf-8 -*-
"""NSRDB data mover and manipulation utilities.

@author: gbuster
"""

import logging
import os
import shlex
import time
import gzip
import shutil
from urllib.request import urlopen
from urllib.error import URLError
from subprocess import Popen, PIPE
from dask.distributed import Client, LocalCluster

from nsrdb.utilities.loggers import init_logger, NSRDB_LOGGERS


logger = logging.getLogger(__name__)


DIR = os.path.dirname(os.path.realpath(__file__))
TOOL = os.path.join(DIR, '_h4h5tools-2.2.2-linux-x86_64-static',
                    'bin', 'h4toh5')


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

    # initialize a logger to the stdout
    init_logger(__name__, log_file=None, log_level='INFO')

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
    logger.info('Finished repacking {0} to {1}. Time elapsed: {2:.2f} minutes.'
                .format(f_orig, f_new, min_elapsed))


def unzip_gz(target_path):
    """Unzip all *.gz files in the target path.

    Note that the original *.gz files are removed (unzipped in place).
    """
    flist = os.listdir(target_path)
    for i, f in enumerate(flist):
        if f.endswith('.gz'):
            logger.info('Unzipping file #{} (out of {}): "{}"'
                        .format(i, len(flist), f))
            gz_file = os.path.join(target_path, f)
            target_file = os.path.join(target_path,
                                       f.replace('.gz', ''))

            with gzip.open(gz_file, 'rb') as f_in:
                with open(target_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            os.remove(gz_file)

    return True


def url_download(url, target):
    """Download file from url to target location.

    Parameters
    ----------
    url : str
        Source file url.
    target : str
        Local target file location to dump data from url.
    """
    failed = False
    logger.debug('URL downloading: {}'.format(url))

    try:
        req = urlopen(url)
        with open(target, 'wb') as dfile:
            # gz archive must be written as a binary file
            dfile.write(req.read())

    except URLError as e:
        logger.info('Skipping: {} was not downloaded'
                    .format(url))
        logger.exception(e)
        failed = url
        pass

    return failed


def convert_h4(path4, f_h4, path5, f_h5):
    """Use a subprocess to convert a single h4 to h5 file.

    Parameters
    ----------
    path4 : str
        Path of source h4 file.
    f_h4 : str
        Filename of source h4 file to convert.
    path5 : str
        Destination of final h5 file.
    f_h5 : str
        Filename of final converted h5 file.
    """

    if not f_h4.endswith('.h4') and not f_h4.endswith('.hdf'):
        raise TypeError('Specified h4 file not recognized as an .hdf or .h4: '
                        '"{}"'.format(f_h4))
    if not f_h5.endswith('.h5'):
        f_h5 += '.h5'

    h4 = os.path.join(path4, f_h4)
    h5 = os.path.join(path5, f_h5)

    if not os.path.exists(h4):
        raise IOError('Could not locate file for conversion to h5: {}'
                      .format(h4))
    if os.path.exists(h5):
        logger.info('Target h5 file already exists, may have already been '
                    'converted, skipping: {}'.format(h5))
        stdout = 'File already exists: {}'.format(h5)
        stderr = ''
    else:
        cmd = '{tool} {h4} {h5}'.format(tool=TOOL, h4=h4, h5=h5)
        logger.debug('Executing the command: {}'.format(cmd))
        cmd = shlex.split(cmd)

        # submit subprocess and wait for stdout/stderr
        t0 = time.time()
        process = Popen(cmd, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        stderr = stderr.decode('ascii').rstrip()
        stdout = stdout.decode('ascii').rstrip()
        elapsed = (time.time() - t0) / 60
        if stderr:
            logger.warning('Conversion of "{}" returned a stderr: "{}"'
                           .format(stderr))
        else:
            logger.info('Finished conversion of "{0}" in {1:.2f} min, '
                        'stdout: "{2}"'.format(f_h5, elapsed, stdout))

    return (stdout, stderr)


def get_conversion_list(path4, path5):
    """Get a list of hdf/h4 files to convert with source/target entries.

    Parameters
    ----------
    path4 : str
        Path of directory containing hdf4 files to be converted. Files with
        .hdf or .h4 extensions in this directory will be converted.
    path5: str
        Path of target directory to dump converted hdf5 files (directory
        structure matching path4 will be created).

    Returns
    -------
    conversion_list : list
        List of paths and files to convert for input to convert4to5.
        Format is: conversion_list = [[path4, f_h4, path5, f_h5], ...]
    """

    conversion_list = []
    if not path4.endswith('/'):
        path4 += '/'
    if not path5.endswith('/'):
        path5 += '/'
    for root, _, files in os.walk(path4):
        # walk through the directory tree.
        for name in files:
            if name.endswith('.h4') or name.endswith('.hdf'):

                # get just the sub directory without the source path4
                sub = root.replace(path4, '')
                source_path = root
                source_file = name
                target_path = os.path.join(path5, sub)
                target_file = name.replace('.h4', '.h5').replace('.hdf', '.h5')

                conversion_list.append([source_path, source_file,
                                        target_path, target_file])

    return conversion_list


def convert_list_serial(conversion_list):
    """Convert h4 to h5 files in serial based on the conversion list.

    Parameters
    -------
    conversion_list : list
        List of paths and files to convert for input to convert4to5.
        Format is: conversion_list = [[path4, f_h4, path5, f_h5], ...]
    """

    logger.info('Converting {} hdf files in serial.'
                .format(len(conversion_list)))
    for [path4, f_h4, path5, f_h5] in conversion_list[0:1]:
        convert_h4(path4, f_h4, path5, f_h5)


def convert_list_parallel(conversion_list, n_workers=2):
    """Convert h4 to h5 files in parallel based on the conversion list.

    Parameters
    -------
    conversion_list : list
        List of paths and files to convert for input to convert4to5.
        Format is: conversion_list = [[path4, f_h4, path5, f_h5], ...]
    n_workers : int
        Number of Dask local workers to use.
    """

    futures = []
    # start client with n_workers to use
    logger.info('Starting a Dask client with {} workers to convert {} '
                'hdf files.'.format(n_workers, len(conversion_list)))
    with Client(LocalCluster(n_workers=n_workers)) as client:
        # initialize loggers on workers.
        client.run(NSRDB_LOGGERS.init_logger, __name__)
        # iterate through list to convert
        for [path4, f_h4, path5, f_h5] in conversion_list:
            # kick off conversion on a worker without caring about result.
            futures.append(client.submit(
                convert_h4, path4, f_h4, path5, f_h5))

        futures = client.gather(futures)


def convert_directory(path4, path5, n_workers=1):
    """Convert a directory of hdf4 files to hdf5 in a new directory.

    Parameters
    ----------
    path4 : str
        Path of directory containing hdf4 files to be converted. Files with
        .hdf or .h4 extensions in this directory will be converted.
    path5 : str
        Path of target directory to dump converted hdf5 files (directory
        structure matching path4 will be created).
    n_workers : int
        Number of workers to use. 1 converts all files in serial, >1 has each
        worker convert a file. None uses all available workers on the node.
    """

    logger.info('Converting h4 files in directory "{}" to "{}"'
                .format(path4, path5))

    # get the list of paths/files to convert
    conversion_list = get_conversion_list(path4, path5)

    # make directory tree in new location if it doesnt exist
    for [_, _, path5, _] in conversion_list:
        if not os.path.exists(path5):
            os.makedirs(path5)

    # kick off conversion in serial or parallel
    if n_workers == 1:
        convert_list_serial(conversion_list)
    else:
        convert_list_parallel(conversion_list, n_workers=n_workers)

    logger.info('Finished converting h4 files in "{}" to "{}"'
                .format(path4, path5))


if __name__ == '__main__':
    path4 = '/scratch/mfoster/2018/adj_920/213/level2/'
    path5 = '/scratch/gbuster/wisc_h5_data'
    init_logger(__name__, log_level='DEBUG',
                log_file=os.path.join(path5, 'convert.log'))
    convert_directory(path4, path5, n_workers=5)
