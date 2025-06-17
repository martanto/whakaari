from __future__ import annotations
from datetime import datetime
from numpy import ndarray
from pandas import Timestamp
from obspy import UTCDateTime
import pandas as pd
import pickle
import numpy as np
from typing import Tuple, List


def to_datetime(datetime_str) -> datetime:
    """Return datetime object corresponding to input string.

    Args:
        datetime_str: Input date time string

    Returns:
        datetime object
    """
    if type(datetime_str) in [datetime, Timestamp]:
        return datetime_str

    if type(datetime_str) is UTCDateTime:
        return datetime_str.datetime

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y %m %d %H %M %S",
        "%Y-%m-%d",
        "%Y%m%d:%H%M",
    ]

    for _format in formats:
        try:
            return datetime.strptime(datetime_str, _format)
        except ValueError:
            pass

    raise ValueError(f"Time data '{datetime_str}' not a recognized format")


def load_dataframe(
    filename,
    index_col=None,
    parse_dates: bool = False,
    usecols=None,
    infer_datetime_format=False,
    nrows=None,
    header="infer",
    skiprows=None,
) -> pd.DataFrame:
    if filename.endswith(".csv"):
        print(f"📖 Reading from CSV file: {filename}")
        return pd.read_csv(
            filename,
            index_col=index_col,
            parse_dates=parse_dates,
            usecols=usecols,
            infer_datetime_format=infer_datetime_format,
            nrows=nrows,
            header=header,
            skiprows=skiprows,
        )

    # Handle pickle and HDF file
    if filename.endswith(".pkl"):
        fp = open(filename, "rb")
        df = pickle.load(fp)
    elif filename.endswith(".hdf"):
        df = pd.read_hdf(filename, "test")
    else:
        raise ValueError("Only CSV and pkl file formats supported")

    if usecols is not None:
        if len(usecols) == 1 and usecols[0] == df.index.name:
            df = df[df.columns[0]]
        else:
            df = df[usecols]
    if nrows is not None:
        if skiprows is None:
            skiprows = range(1, 1)
        skiprows = list(skiprows)
        inds = sorted(set(range(len(skiprows) + nrows)) - set(skiprows))
        df = df.iloc[inds]
    elif skiprows is not None:
        df = df.iloc[skiprows:]
    return df


def save_dataframe(
    df: pd.DataFrame, filename, index: bool = True, index_label: str = None
) -> bool:
    if filename.endswith(".csv"):
        df.to_csv(filename, index=index, index_label=index_label)
        return True

    if filename.endswith(".pkl"):
        fp = open(filename, "wb")
        pickle.dump(df, fp)
        return True

    if filename.endswith(".hdf"):
        df.to_hdf(filename, "test", format="fixed", mode="w")
        return True

    raise ValueError("only csv, hdf and pkl file formats supported")


def outlier_detection(data: List, outlier_degree: float = 0.5) -> Tuple[bool, int]:
    """Determines whether a given data interval requires earthquake filtering

    Args:
        data (ndarray): 10 minute interval of a processed datastream (rsam, mf, hf, mfd, hfd).
        outlier_degree (float, optional): Exponent (base 10) which determines the Z-score required to
                                        be considered an outlier.

    Returns:
        outlier (bool): Is the maximum of the data considered an outlier?
        max_idx (int): The index of the maximum of the data considered an outlier.
    """
    mean = np.mean(data)
    std = np.std(data)
    max_idx = np.argmax(data)  # get index of maximum value

    # Compute Z-score
    z_score = (data[max_idx] - mean) / std

    # Determine if an outlier
    if z_score > 10**outlier_degree:
        outlier = True
    else:
        outlier = False

    return outlier, int(max_idx)


def find_outliers(data: List, n: int, m: int) -> Tuple[List[bool], List[int]]:
    """Finds outlier indices for a given data interval.

    Args:
        data (list): Data
        n (int): Number windows
        m (int): Number of data

    Returns:
        outliers (list[bool]): List of indices of outliers.
        max_idxs (list[int]): List of indices of the maximum of the data considered an outlier.
    """
    outliers = []
    max_idxs = []
    for _m in range(m):
        outlier, max_idx = outlier_detection(data[2][_m * n : (_m + 1) * n])
        outliers.append(outlier)
        max_idxs.append(max_idx)

    return outliers, max_idxs


def wrapped_indices(
    max_idx: int, asymmetry_factor: float, sub_domain_range: int, n: int
) -> List[int]:
    """Wrapped indices based on asymmetry factor and subdomain range.

    Args:
        max_idx (int): The index of the maximum of the data considered an outlier.:
        asymmetry_factor (float): The asymmetry factor
        sub_domain_range (int): The subdomain range.
        n (int): The number data.

    Returns:
        list[int]: The indices of the maximum of the data.
    """

    # Compute the index of the domain where the subdomain centered on the peak begins
    start_idx = int(max_idx - np.floor(asymmetry_factor * sub_domain_range))

    # Find the end index of the subdomain
    end_idx = start_idx + sub_domain_range

    # If end index exceeds data range
    if end_idx >= n:
        # Wrap domain so continues from beginning of data range
        idx = list(range(end_idx - n))
        end = list(range(start_idx, n))
        idx.extend(end)

    # If starting index exceeds data range
    elif start_idx < 0:
        idx = list(range(end_idx))

        # Wrap domains so continues at end of data range
        end = list(range(n + start_idx, n))
        idx.extend(end)
    else:
        idx = list(range(start_idx, end_idx))
    return idx


def compute_rsam(
    data: List,
    band_names: List[str],
    m: int,
    n: int,
    outliers: List[bool],
    max_idxs: List[int],
    asymmetry_factor: float,
    sub_domain_range: int,
) -> Tuple[list, list]:
    """Calculates RSAM (w/ EQ filter).

    Args:
        data (list): Data
        band_names (list): Band names
        m (int): Number of data
        n (int): Number windows
        outliers (list[bool]): List of indices of outliers.
        max_idxs (list[int]): List of indices of the maximum of the data considered an outlier.
        asymmetry_factor (float): The asymmetry factor
        sub_domain_range (int): The subdomain range.

    Returns:
        datas (list): RSAM data
        columns (list): RSAM column names
    """
    datas = []
    columns = []

    for _data, band_name in zip(data, band_names):
        dr = []
        df = []
        for k, outlier, max_idx in zip(range(m), outliers, max_idxs):
            domain = _data[k * n : (k + 1) * n]
            dr.append(np.mean(domain))

            # Filter data when outlier found
            if outlier:
                _indices = wrapped_indices(
                    max_idx, asymmetry_factor, sub_domain_range, n
                )

                # Remove the subdomain with the largest peak
                domain = np.delete(domain, _indices)
            df.append(np.mean(domain))
        datas.append(np.array(dr))
        columns.append(band_name)

        datas.append(np.array(df))
        columns.append(band_name + "F")

    return datas, columns


def compute_dsar(
    data_is: list,
    ratio_names: List[str],
    m: int,
    n: int,
    outliers: List[bool],
    max_idxs: List[int],
    asymmetry_factor: float,
    sub_domain_range: int,
) -> Tuple[list, list]:
    """Compute DSAR (w/ EQ filter)"""
    datas = []
    columns = []

    for index_ratio, ratio_name in enumerate(ratio_names):
        dr = []
        df = []
        for k, outlier, maxIdx in zip(range(m), outliers, max_idxs):
            domain_mf = data_is[index_ratio][k * n : (k + 1) * n]
            domain_hf = data_is[index_ratio + 1][k * n : (k + 1) * n]
            dr.append(np.mean(domain_mf) / np.mean(domain_hf))

            # Filter data when outlier found
            if outlier:
                idx = wrapped_indices(maxIdx, asymmetry_factor, sub_domain_range, n)
                domain_mf = np.delete(domain_mf, idx)
                domain_hf = np.delete(domain_hf, idx)
            df.append(np.mean(domain_mf) / np.mean(domain_hf))
        datas.append(np.array(dr))
        columns.append(ratio_name)
        datas.append(np.array(df))
        columns.append(ratio_name + "F")

    return datas, columns
