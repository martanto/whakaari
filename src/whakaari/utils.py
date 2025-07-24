from __future__ import annotations

import pandas as pd
import numpy as np
import os, joblib, sys, pickle


from .const import CLASSIFIERS
from datetime import datetime
from dateutil import tz
from numpy import ndarray
from pandas import Timestamp
from obspy import UTCDateTime
from typing import Tuple, List, Any, Dict
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, ShuffleSplit
from tsfresh.transformers import FeatureSelector
from imblearn.under_sampling import RandomUnderSampler


def progress_bar(
    current_iteration: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    decimals: int = 1,
    length: int = 25,
    fill: str = "█",
) -> None:
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (current_iteration / float(total))
    )
    filled_length = int(length * current_iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    sys.stdout.write(f"\r{prefix} |{bar}| {percent}% {suffix}")
    sys.stdout.flush()


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
    parse_dates: Any = False,
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


def get_classifier(classifier: str) -> Tuple[Any, Dict]:
    """Return scikit-learn ML classifiers and search grids for input strings.
    Parameters:
    -----------
    classifier : str
        String designating which classifier to return.
    Returns:
    --------
    model :
        Scikit-learn classifier object.
    grid : dict
        Scikit-learn hyperparameter grid dictionarie.
    Classifier options:
    -------------------
    SVM - Support Vector Machine.
    KNN - k-Nearest Neighbors
    DT - Decision Tree
    RF - Random Forest
    NN - Neural Network
    NB - Naive Bayes
    LR - Logistic Regression
    """
    if classifier == "SVM":  # support vector machine
        model = SVC(class_weight="balanced")
        grid = {
            "C": [0.001, 0.01, 0.1, 1, 10],
            "kernel": ["poly", "rbf", "sigmoid"],
            "degree": [2, 3, 4, 5],
            "decision_function_shape": ["ovo", "ovr"],
        }
    elif classifier == "KNN":  # k nearest neighbour
        model = KNeighborsClassifier()
        grid = {
            "n_neighbors": [3, 6, 12, 24],
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3],
        }
    elif classifier == "DT":  # decision tree
        model = DecisionTreeClassifier(class_weight="balanced")
        grid = {
            "max_depth": [3, 5, 7],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2", None],
        }
    elif classifier == "RF":  # random forest
        model = RandomForestClassifier(class_weight="balanced")
        grid = {
            "n_estimators": [10, 30, 100],
            "max_depth": [3, 5, 7],
            "criterion": ["gini", "entropy"],
            "max_features": ["auto", "sqrt", "log2", None],
        }
    elif classifier == "NN":  # neural network
        model = MLPClassifier(alpha=1, max_iter=1000)
        grid = {
            "activation": ["identity", "logistic", "tanh", "relu"],
            "hidden_layer_sizes": [10, 100],
        }
    elif classifier == "NB":  # naive bayes
        model = GaussianNB()
        grid = {"var_smoothing": [1.0e-9]}
    elif classifier == "LR":  # logistic regression
        model = LogisticRegression(class_weight="balanced")
        grid = {"penalty": ["l2", "l1", "elasticnet"], "C": [0.001, 0.01, 0.1, 1, 10]}
    else:
        raise ValueError(f"❌ Classifier '{classifier}' not recognised")

    return model, grid


def train_one_model(
    feature_matrix,
    label_vector,
    number_of_significant_features,
    model_dir,
    classifier,
    retrain,
    random_seed,
    method,
    random_state,
):
    # undersample data
    rus = RandomUnderSampler(method, random_state=random_state + random_seed)
    fmt, yst = rus.fit_resample(feature_matrix, label_vector)
    yst = pd.Series(yst > 0, index=range(len(yst)))
    fmt.index = yst.index

    # find significant features
    select = FeatureSelector(n_jobs=0, ml_task="classification")
    select.fit_transform(fmt, yst)
    fts = select.features[:number_of_significant_features]
    pvs = select.p_values[:number_of_significant_features]
    fmt = fmt[fts]
    with open("{:s}/{:04d}.fts".format(model_dir, random_state), "w") as fp:
        for f, pv in zip(fts, pvs):
            fp.write("{:4.3e} {:s}\n".format(pv, f))

    # get sklearn training objects
    ss = ShuffleSplit(
        n_splits=5, test_size=0.25, random_state=random_state + random_seed
    )
    model, grid = get_classifier(classifier)

    # check if model has already been trained
    pref = type(model).__name__
    fl = "{:s}/{:s}_{:04d}.pkl".format(model_dir, pref, random_state)
    if os.path.isfile(fl) and not retrain:
        return

    # train and save classifier
    model_cv = GridSearchCV(
        model, grid, cv=ss, scoring="balanced_accuracy", error_score=np.nan
    )
    model_cv.fit(fmt, yst)
    _ = joblib.dump(model_cv.best_estimator_, fl, compress=3)


def predict_one_model(feature_matrix, model_path, _flp):
    file_model, file_prediction = _flp

    # print(f"File Model : {file_model}")
    # print(f"File Prediction : {file_prediction}")

    number = file_model.split(os.sep)[-1].split(".")[0].split("_")[-1]
    model = joblib.load(file_model)

    feature_path = os.path.join(model_path, f"{number}.fts")
    with open(feature_path) as fp:
        lns = fp.readlines()

    fts = [" ".join(ln.rstrip().split()[1:]) for ln in lns]

    if not os.path.isfile(file_prediction):
        # simulate predicton period
        yp = model.predict(feature_matrix[fts])
        # save prediction
        ypdf = pd.DataFrame(
            yp, columns=["pred{:s}".format(number)], index=feature_matrix.index
        )
    else:
        ypdf0 = load_dataframe(
            file_prediction,
            index_col="time",
            infer_datetime_format=True,
            parse_dates=["time"],
        )

        fm2 = feature_matrix.loc[feature_matrix.index > ypdf0.index[-1], fts]
        if fm2.shape[0] == 0:
            ypdf = ypdf0
        else:
            yp = model.predict(fm2)
            ypdf = pd.DataFrame(
                yp, columns=["pred{:s}".format(number)], index=fm2.index
            )
            ypdf = pd.concat([ypdf0, ypdf])

    # ypdf.to_csv(fl, index=True, index_label='time')
    save_dataframe(ypdf, file_prediction, index=True, index_label="time")
    return ypdf


def to_nz_timezone(t):
    """Routine to convert UTC to NZ time zone."""
    utc_tz = tz.gettz("UTC")
    nz_tz = tz.gettz("Pacific/Auckland")
    return [ti.replace(tzinfo=utc_tz).astimezone(nz_tz) for ti in pd.to_datetime(t)]


def classifier_codes():
    codes = []
    for classifier in CLASSIFIERS:
        codes.append(classifier["code"])

    return codes
