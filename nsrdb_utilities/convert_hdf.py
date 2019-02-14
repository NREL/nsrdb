# -*- coding: utf-8 -*-
"""NSRDB data mover and manipulation utilities.

@author: gbuster
"""

import logging
import os
import shlex
import subprocess

from nsrdb_utilities.loggers import init_logger


logger = logging.getLogger(__name__)


DIR = os.path.dirname(os.path.realpath(__file__))
TOOL = os.path.join(DIR, 'h4h5tools-2.2.2-linux-x86_64-static',
                    'bin', 'h4toh5')


def convert4to5(path4, f_h4, path5, f_h5):
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

    if not f_h4.endswith('.h4') or not f_h4.endswith('.hdf'):
        raise TypeError('Specified h4 file not recognized as an .hdf or .h4: '
                        '"{}"'.format(f_h4))
    if not f_h5.endswith('.h5'):
        f_h5 += '.h5'

    h4 = os.path.join(path4, f_h4)
    h5 = os.path.join(path5, f_h5)

    logger.info('Converting "{}" to "{}"'.format(h4, h5))
    cmd = '{tool} {h4} {h5}'.format(tool=TOOL, h4=h4, h5=h5)
    cmd = shlex.split(cmd)
    subprocess.call(cmd)


def get_conversion_list(path4, path5):
    """Get a list of files to convert with source/target entries.

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


def convert_directory(path4, path5):
    """Convert a directory of hdf4 files to hdf5 in a new directory.

    Parameters
    ----------
    path4 : str
        Path of directory containing hdf4 files to be converted. Files with
        .hdf or .h4 extensions in this directory will be converted.
    path5: str
        Path of target directory to dump converted hdf5 files (directory
        structure matching path4 will be created).
    """

    conversion_list = get_conversion_list(path4, path5)
    print(conversion_list)
    print('')

    logger.info('Converting directory "{}" to "{}"'.format(path4, path5))

    for [path4, f_h4, path5, f_h5] in conversion_list:
        if not os.path.exists(path5):
            # make directory tree in new location if it doesnt exist
            os.makedirs(path5)
        print(path4, f_h4, path5, f_h5)


if __name__ == '__main__':
    init_logger(__name__, log_level='DEBUG', log_file=None)
    convert_directory('/scratch/mfoster/2018/adj_920/213/level2',
                      '/scratch/gbuster/wisc_h5_data')
