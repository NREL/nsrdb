# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:50:16 2019

@author: gbuster
"""
import os

import h5py
import numpy as np
import pandas as pd
from farms import SZA_LIM

DAT_COLS = (
    "year",
    "jday",
    "month",
    "day",
    "hour",
    "min",
    "dt",
    "zen",
    "dw_solar",
    "qc_dwsolar",
    "uw_solar",
    "qc_uw_solar",
    "direct_n",
    "qc_direct_n",
    "diffuse",
    "qc_diffuse",
)

DAT_MAPPING = {
    "dw_solar": "ghi",
    "direct_n": "dni",
    "diffuse": "dhi",
    "zen": "sza",
}

LW1_MAPPING = {"swdn": "ghi", "dirsw": "dni", "difsw": "dhi", "sza": "sza"}

MISSING = -999


def filter_measurement_df(df, var_list=("dhi", "dni", "ghi", "sza")):
    """Filter the measurement dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Ground measurement dataframe containing variables in var_list.
    var_list : list | tuple
        Variables to preserve (typically irradiance and sza).

    Returns
    -------
    df : pd.DataFrame
        Filtered data with zero or negative irradiance replaced with missing
        flag.
    """

    df = df[list(var_list)]
    df = df.sort_index()

    for var in var_list:
        # convert all data to float
        df[var] = df[var].astype(float)

    for var in var_list:
        # No data can be negative
        mask = df[var] < 0
        df.loc[mask, var] = MISSING

        if var in ("dhi", "ghi"):
            # dhi and ghi cannot be negative or zero during the day
            mask = (df[var] <= 0) & (df["sza"] < SZA_LIM)
            df.loc[mask, var] = MISSING

    return df


def get_dat_table(d, flist):
    """Get one year of data from the directory for .dat files.

    Parameters
    ----------
    d : str
        Directory containing surfrad data files.
    flist : list
        List of dat filenames to extract.

    Returns
    -------
    annual_table : list
        List of data rows from all files in d. First entry is the list of
        column headers.
    """

    cols = ()

    # iterate through data files
    for i, fname in enumerate(flist):
        table = []

        # get readlines iterator
        with open(os.path.join(d, fname)) as f:
            lines = f.readlines()

        # iterate through lines
        for line in lines:
            # reduce multiple spaces to a single space, split columns
            while "  " in line:
                line = line.replace("  ", " ")
            cols = line.strip(" ").split(" ")

            # Set table header or append data to table
            if len(cols) > len(DAT_COLS):
                table.append(cols[0: len(DAT_COLS)])

        # upon finishing table concatenation, initialize annual table
        # or append to annual table
        if i == 0:
            annual_table = table
        else:
            annual_table += table

    df = pd.DataFrame(annual_table, columns=DAT_COLS)
    df = df.rename(DAT_MAPPING, axis="columns")
    df["time_string"] = (
        df["year"]
        + df["month"].str.zfill(2)
        + df["day"].str.zfill(2)
        + df["hour"].str.zfill(2)
        + df["min"].str.zfill(2)
    )

    ti = pd.to_datetime(df["time_string"], format="%Y%m%d%H%M")
    df.index = ti
    df = df.sort_index()
    return df


def get_lw1_table(d, flist):
    """Get one year of data from the directory for .lw1 files.

    Parameters
    ----------
    d : str
        Directory containing surfrad data files.
    flist : list
        List of lw1 filenames to extract.

    Returns
    -------
    annual_table : list
        List of data rows from all files in d. First entry is the list of
        column headers.
    """

    # iterate through data files
    for i, fname in enumerate(flist):
        # get readlines iterator
        with open(os.path.join(d, fname)) as f:
            lines = f.readlines()

        # iterate through lines
        for j, line in enumerate(lines):
            # reduce multiple spaces to a single space, split columns
            while "  " in line:
                line = line.replace("  ", " ")
            cols = line.strip(" ").split(" ")

            # Set table header or append data to table
            if j == 0:
                table = [cols]
            else:
                table.append(cols)

        # upon finishing table concatenation, initialize annual table
        # or append to annual table
        if i == 0:
            annual_table = table
        else:
            if table[0] == annual_table[0]:
                # make sure headers are the same
                annual_table += table[1:]
            else:
                msg = (
                    'Headers for "{}" does not match annual table '
                    "headers: {}".format(
                        os.path.join(d, fname), annual_table[0]
                    )
                )
                raise ValueError(msg)

    headers = [h.lower() for h in annual_table[0]]
    df = pd.DataFrame(annual_table[1:], columns=headers)
    df = df[["zdate", "ztim", "cosz", "swdn", "dirsw", "difsw"]]
    df["sza"] = np.arccos(df["cosz"])
    df = df.rename(LW1_MAPPING, axis="columns")
    df["time_string"] = df["zdate"] + " " + df["ztim"].str.zfill(4)
    ti = pd.to_datetime(df["time_string"], format="%Y%m%d %H%M")
    df.index = ti
    df = df.sort_index()
    return df


def surfrad_to_h5(df, fout, dir_out):
    """Save annual surfrad dataframe to output .h5 file.

    Parameters
    ----------
    df : pd.DataFrame
        Timeseries dataframe for ground measurement irradiance. DF index must
        be pandas datetime index. Must have columns for dhi, dni, ghi.
    fout : str
        Output file name.
    dir_out : str
        Location to save output file.
    """

    with h5py.File(os.path.join(dir_out, fout), "w") as f:
        # write time index
        time_index = np.array(df.index.astype(str), dtype="S20")
        ds = f.create_dataset(
            "time_index",
            shape=time_index.shape,
            dtype=time_index.dtype,
            chunks=None,
        )
        ds[...] = time_index

        # write solar zenith angle
        ds = f.create_dataset(
            "solar_zenith_angle",
            shape=df["sza"].shape,
            dtype=np.float16,
            chunks=None,
        )
        ds[...] = df["sza"].values

        # write irraidance variables
        for dset in ["dhi", "dni", "ghi"]:
            df[dset] = np.round(df[dset].astype(float)).astype(np.int16)
            ds = f.create_dataset(
                dset, shape=df[dset].shape, dtype=df[dset].dtype, chunks=None
            )
            ds[...] = df[dset].values


def extract_all(
    root_dir,
    dir_out,
    years=range(1998, 2018),
    file_flag=".dat",
    site_codes=("bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl"),
):
    """Extract all surfrad measurement data into h5 files.

    Parameters
    ----------
    root_dir : str
        Root directory containing surfrad ground measurement data.
        Directory containing data files is:
            /root_dir/site_code/year/
    dir_out : str
        Target path to save surfrad h5 data.
    years : iterable
        Years to process for.
    site_codes : list | tuple
        Sites codes that makeup directory names.
    """

    bad_dirs = []

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    for site in site_codes:
        for year in years:
            # look for target data directory
            d = os.path.join(root_dir, site, str(year))

            # set target output filename
            fout = "{}_{}.h5".format(site, year)

            if not os.path.exists(d):
                print(
                    'Skipping: "{}" for {}. Path does not exist: {}'.format(
                        site, year, d
                    )
                )
                bad_dirs.append(d)
            elif os.path.exists(os.path.join(dir_out, fout)):
                print("Skipping file, already exists: {}".format(fout))
            else:
                # get number of valid files in dir
                flist = [f for f in os.listdir(d) if file_flag in f]

                print('Processing "{}" for {}'.format(site, year))
                if "dat" in file_flag:
                    df = get_dat_table(d, flist)
                elif "lw1" in file_flag:
                    df = get_lw1_table(d, flist)
                else:
                    raise (
                        "Did not recongize user-specified file flag: "
                        '"{}"'.format(file_flag)
                    )

                df = filter_measurement_df(df)

                surfrad_to_h5(df, fout, dir_out)

    print(
        "The following directories did not have valid datasets:\n{}".format(
            bad_dirs
        )
    )
    return df


if __name__ == "__main__":
    root_dir = "/projects/pxs/surfrad/raw"
    dir_out = "/projects/pxs/surfrad/h5"
    site_codes = ("bon", "dra", "fpk", "gwn", "psu", "sxf", "tbl")
    extract_all(root_dir, dir_out, site_codes=site_codes)
