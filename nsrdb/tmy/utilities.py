"""Shared TMY methods."""


def make_time_masks(time_index):
    """Make a time index mask lookup dict.

    Parameters
    ----------
    time_index : pd.datetimeindex
        Time index to mask.

    Returns
    -------
    masks : dict
        Lookup of boolean masks keyed by month and year integer.
    """
    masks = {}
    years = time_index.year.unique()
    months = time_index.month.unique()
    for y in years:
        masks[y] = time_index.year == y
    for m in months:
        masks[m] = time_index.month == m
    return masks


def drop_leap(arr, time_index):
    """Make 365-day timeseries (TMY does not have leap days).

    Parameters
    ----------
    arr : np.ndarray
        Timeseries array (time, sites) for one variable.
    time_index : pd.datetimeindex
        Datetime index corresponding to the rows in arr. May have leap days

    Returns
    -------
    arr : np.ndarray
        Timeseries array (time, sites) for one variable, n_rows is a
        multiple of 8760.
    time_index : pd.datetimeindex
        Datetime index corresponding to the rows in arr, without leap days.
    """
    if len(arr) % 8760 != 0:
        leap_day = (time_index.month == 2) & (time_index.day == 29)
        if any(leap_day):
            arr = arr[~leap_day]
            time_index = time_index[~leap_day]
    return arr, time_index
