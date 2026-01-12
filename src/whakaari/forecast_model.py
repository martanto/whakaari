from __future__ import annotations

import os, gc, pathlib
import pandas as pd
import numpy as np
import sys
from matplotlib import pyplot as plt
from fnmatch import fnmatch
from glob import glob
from .tremor_data import TremorData
from .utils import (
    to_datetime,
    get_classifier,
    load_dataframe,
    save_dataframe,
    train_one_model,
    predict_one_model,
    to_nz_timezone,
    classifier_codes,
)
from datetime import datetime, timedelta
from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction.settings import ComprehensiveFCParameters
from functools import partial
from multiprocessing import Pool
import matplotlib.dates as mdates


class ForecastModel:
    """
    Object for train and running forecast models.

    Attributes
    ----------
    station : str
        Station name.
    start_date : str
        Beginning of analysis period. If not given, will default to beginning of tremor data.
    end_date : str
        End of analysis period. If not given, will default to end of tremor data.
    window : int
        Length of data window in days.
    overlap : float
        Fraction of overlap between adjacent windows. Set this to 1. For overlap of entire window minus 1 data point.
    look_forward : float
        Length of look-forward in days.
    data_streams : list[str], optional
        Data streams and transforms from which to extract features. Options are 'X', 'diff_X', 'log_X', 'inv_X', and 'stft_X'
        where X is one of 'rsam', 'mf', 'hf', or 'dsar'.
    verbose : bool
        Enable additional print statements.
    """

    def __init__(
        self,
        station: str,
        start_date: str,
        end_date: str,
        window: float,
        overlap: float,
        look_forward: float,
        eruptive_file: str,
        tremor_data_file: str = None,
        data_streams: list = None,
        exclude_dates: list = None,
        root: str = None,
        feature_root: str = None,
        feature_dir: str = None,
        n_jobs: int = 2,
        savefile_type="pkl",
        show_warnings: bool = False,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Object for train and running forecast models.

        Parameters
        ----------
        station : str
            Station name
        start_date : str
            Start date
        end_date : str
            End date
        window : float
            Length of data window in days.
        overlap : float
            Fraction of overlap between adjacent windows. Set this to 1.
            For overlap of entire window minus 1 data point.
        """
        if data_streams is None:
            data_streams = ["rsam", "mf", "hf", "dsar"]

        self.verbose = verbose
        self.debug = debug
        self.show_warnings = show_warnings
        self.station = station
        self.data_streams = data_streams
        self.exclude_dates = exclude_dates

        # Length of look-forward in days
        self.look_forward: float = look_forward

        self.eruptive_file = eruptive_file
        self.savefile_type = savefile_type
        self.tremor_data_file = tremor_data_file

        self.data: TremorData = TremorData(
            station=station,
            parent=self,
            data_dir=os.getcwd(),
            eruptive_file=eruptive_file,
            tremor_data_file=tremor_data_file,
        )

        if any([column not in self.data.df.columns for column in data_streams]):
            raise ValueError(f"❌ Data restricted to any of {self.data.df.columns}")

        if any(["_" in column for column in data_streams]):
            self.data.compute_transforms()

        if start_date is None:
            start_date = self.data.datetime_start
        self.start_date = start_date

        if end_date is None:
            end_date = self.data.datetime_end
        self.end_date = end_date

        # Originally self.ti_model and self.tf_model
        self.start_date_model: datetime = to_datetime(start_date)
        self.end_date_model: datetime = to_datetime(end_date)

        if self.end_date_model > self.data.datetime_end:
            t0, t1 = [
                self.end_date_model.strftime("%Y-%m-%d %H:%M"),
                self.data.datetime_end.strftime("%Y-%m-%d %H:%M"),
            ]
            raise ValueError(
                "Model end date '{:s}' beyond data range '{:s}'".format(t0, t1)
            )
        if self.start_date_model < self.data.datetime_start:
            t0, t1 = [
                self.start_date_model.strftime("%Y-%m-%d %H:%M"),
                self.data.datetime_start.strftime("%Y-%m-%d %H:%M"),
            ]
            raise ValueError(
                "Model start date '{:s}' predates data range '{:s}'".format(t0, t1)
            )

        # Length of look-forward.
        self.dtf = timedelta(days=look_forward)

        # Length between data samples (10 minutes).
        self.dt = timedelta(seconds=600)

        # Number of samples in window.
        self.iw = int(window * 6 * 24)

        # Number of samples in overlapping section of window.
        self.io = int(overlap * self.iw)
        if self.io == self.iw:
            self.io -= 1

        # Length of data window in days.
        self.window = self.iw * 1.0 / (6 * 24)

        # Length of window.
        self.dtw = timedelta(days=self.window)

        if self.start_date_model - self.dtw < self.data.datetime_start:
            self.start_date_model = self.data.datetime_start + self.dtw

        # Fraction of overlap between adjacent windows. Set this to 1.
        # For overlap of entire window minus 1 data point.
        self.overlap = self.io * 1.0 / self.iw

        # Length of non-overlapping section of window
        self.dto = (1.0 - self.overlap) * self.dtw

        self.drop_features = []
        self.exclude_dates = []
        self.use_only_features = []
        self.compute_only_features = []
        self.update_feature_matrix = True
        self.n_jobs = n_jobs

        # naming convention and file system attributes
        if root is None:
            root = (
                f"fm_{self.window:3.2f}_{self.overlap:3.2f}_{self.look_forward:3.2f}_"
            )
            root += (("{:s}-" * len(self.data_streams))[:-1]).format(
                *sorted(self.data_streams)
            )
        self.root = root

        self.feature_root = feature_root
        self.root_dir = os.getcwd()

        self.output_dir = os.path.join(self.root_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.plot_dir = os.path.join(self.output_dir, "plots", self.root)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.model_dir = os.path.join(self.output_dir, "models", self.root)
        os.makedirs(self.model_dir, exist_ok=True)

        if feature_dir is None:
            feature_dir = os.path.join(self.output_dir, "features")
        self.feature_dir = feature_dir
        os.makedirs(self.feature_dir, exist_ok=True)

        self.feature_file = lambda feature_name, data_stream: os.path.join(
            self.feature_dir,
            f"fm_{window:3.2f}w_{data_stream}_{station}{feature_name}.{savefile_type}",
        )

        self.prediction_dir = os.path.join(self.output_dir, "predictions", self.root)
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Defining outside init
        self.classifier = "DT"
        self.use_features = None
        self.number_of_classifiers = 500
        self.start_date_train = None
        self.end_date_train = None
        self.start_date_previous = None
        self.end_date_previous = None
        self.feature_matrix = pd.DataFrame()
        self.label_vector = pd.DataFrame()
        self.start_date_forecast = None
        self.end_date_forecast = None
        self.classifier_codes = classifier_codes()

        if verbose:
            print(f"Start date: {self.start_date_model.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"End date: {self.end_date_model.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Tremor data: {self.data.tremor_file}")
            print(f"Root dir: {self.root_dir}")
            print(f"Output dir: {self.output_dir}")
            print(f"Plot dir: {self.plot_dir}")
            print(f"Model dir: {self.model_dir}")
            print(f"Feature dir: {self.feature_dir}")
            print(f"Prediction dir: {self.prediction_dir}")

    def train(
        self,
        start_date: str,
        end_date: str,
        number_of_significant_features: int = 20,
        number_of_classifiers: int = 500,
        retrain_model: bool = False,
        classifier: str = "DT",
        random_seed: int = 0,
        drop_features: list = None,
        n_jobs: int = 2,
        exclude_dates: list = None,
        method: float = 0.75,
        use_features: list = None,
        show_warnings: bool = False,
    ):
        """Construct classifier models.

        Args:
            start_date (str): Start date of the model.
            end_date (str): End date of the model.
            number_of_significant_features (int): Number of significant features.
            number_of_classifiers (int): Number of classifiers.
            retrain_model (bool): Whether to keep the model or not. Defaults to False.
            classifier (str): Classifier name. Defaults to "DT".
            random_seed (int): Random seed. Defaults to 0.
            drop_features (list): list of feature names to drop. Defaults to None.
            n_jobs (int): Number of jobs. Defaults to 2.
            exclude_dates (list[list[str]): list of dates to exclude features. Example: [['2012-06-01','2012-08-01'],
                ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis. Defaults to None.
            method (float): Method to handle undersample data. Defaults to 0.75.
            use_features (list[str]): list of features to use for feature selection. Defaults to None.
            show_warnings (bool): Whether to show the warnings message. Defaults to False.

            Classifier options:
            -------------------
            SVM - Support Vector Machine.
            KNN - k-Nearest Neighbors
            DT - Decision Tree
            RF - Random Forest
            NN - Neural Network
            NB - Naive Bayes
            LR - Logistic Regression

        Returns:
            self (Self): Self
        """
        assert (
            classifier.upper() in self.classifier_codes
        ), f"❌ Classifier must be one of {self.classifier_codes}"

        self.classifier = classifier.upper()
        self.exclude_dates = exclude_dates
        self.use_features = use_features
        self.n_jobs = n_jobs
        self.show_warnings = show_warnings

        # Originally named as Ncl
        self.number_of_classifiers = number_of_classifiers

        # initialise training interval
        self.start_date_train: datetime = (
            self.start_date_model if start_date is None else to_datetime(start_date)
        )
        self.end_date_train: datetime = (
            self.end_date_model if end_date is None else to_datetime(end_date)
        )
        if self.start_date_train - self.dtw < self.data.datetime_start:
            self.start_date_train = self.data.datetime_start + self.dtw

        # check if any model training required
        run_models = False
        if not retrain_model:
            model, _grids = get_classifier(self.classifier)
            prefix = type(model).__name__
            for classifier_index in range(number_of_classifiers):
                model_path = os.path.join(
                    self.model_dir, f"{prefix}_{classifier_index:04d}.pkl"
                )
                if not os.path.isfile(model_path):
                    run_models = True
            if not run_models:
                return self
        else:
            # delete old model files
            _ = [os.remove(fl) for fl in glob("{:s}/*".format(self.model_dir))]

        # get feature matrix and label vector
        feature_matrix, label_vector = self._load_data(
            start_date=start_date,
            end_date=end_date,
        )

        if self.verbose:
            print(f"Feature Matrix dimension : {feature_matrix.shape}")
            print(f"Label Vector dimension : {label_vector.shape}")

        # manually drop features (columns)
        feature_matrix = self._drop_features(feature_matrix, drop_features)

        # manually select features (columns)
        if len(self.use_only_features) != 0:
            if self.verbose:
                print(f"Use only features: {self.use_only_features}")

            use_only_features = [
                df for df in self.use_only_features if df in feature_matrix.columns
            ]

            feature_matrix = feature_matrix[use_only_features]
            number_of_significant_features = len(use_only_features) + 1

        # manually drop windows (rows)
        feature_matrix, label_vector = self._exclude_dates(
            feature_matrix, label_vector, exclude_dates
        )

        if label_vector.shape[0] != feature_matrix.shape[0]:
            raise ValueError(
                f"❌ Dimensions of feature matrix and label vector do not match.\n"
                f"feature_matrix: {feature_matrix.shape[0]}\n"
                f"label_vector: {label_vector.shape[0]}\n"
            )

        # select training subset
        indices = (label_vector.index >= self.start_date_train) & (
            label_vector.index < self.end_date_train
        )

        feature_matrix = feature_matrix.loc[indices]
        label_vector = label_vector["label"].loc[indices]

        # set up model training
        p = Pool(self.n_jobs)
        mapper = p.imap

        if self.n_jobs == 1:
            mapper = map

        _train_one_model = partial(
            train_one_model,
            feature_matrix,
            label_vector,
            number_of_significant_features,
            self.model_dir,
            self.classifier,
            retrain_model,
            random_seed,
            method,
        )

        # train models with glorious progress bar
        # _train_one_model(0)
        for i, _ in enumerate(mapper(_train_one_model, range(number_of_classifiers))):
            _classifier = (i + 1) / number_of_classifiers
            print(
                f'building models: [{"#" * round(50 * _classifier) + "-" * round(50 * (1 - _classifier))}] {100. * _classifier:.2f}%\r',
                end="",
            )

        if self.n_jobs > 1:
            p.close()
            p.join()

        # free memory
        # del feature_matrix
        gc.collect()
        self._collect_features()

        return self

    def _detect_model(self, model_path: str = None):
        """Checks whether and what models have already been run."""
        if model_path is None:
            model_path = self.model_dir

        files = glob(os.path.join(model_path, "*.fts"))
        if len(files) == 0:
            raise ValueError(f"❌ Features not found at {model_path}")

        indices = [
            int(pathlib.Path(file).stem) for file in files if ("all.fts" not in file)
        ]

        if max(indices) != (len(indices) - 1):
            raise ValueError(
                f"❌ Feature file numbering in '{model_path}' appears not consecutive"
            )

        self.classifier = []
        for classifier in self.classifier_codes:
            model = get_classifier(classifier)[0]
            prefix = type(model).__name__
            if all(
                [
                    os.path.isfile(
                        os.path.join(model_path, f"{prefix}_{classifier_index:04d}.pkl")
                    )
                    for classifier_index in indices
                ]
            ):
                self.classifier = classifier
                return True

        raise ValueError(f"❌ Did not recognise models in '{model_path}'")

    def _load_data(
        self, start_date: str = None, end_date: str = None, year: int = None
    ):
        # Return pre loaded
        if start_date is None:
            start_date: datetime = self.start_date_train

        if end_date is None:
            end_date: datetime = self.end_date_train

        # Ensuring datetime object
        start_date = to_datetime(start_date)
        end_date = to_datetime(end_date)

        try:
            if (
                start_date == self.start_date_previous
                and end_date == self.end_date_previous
            ):
                return self.feature_matrix, self.label_vector
        except AttributeError:
            pass

        # range checking
        if end_date > self.data.datetime_end:
            raise ValueError(
                "❌ Model end date '{:s}' beyond data range '{:s}'".format(
                    end_date, self.data.datetime_end
                )
            )

        if start_date < self.data.datetime_start:
            raise ValueError(
                "❌ Model start date '{:s}' predates data range '{:s}'".format(
                    start_date, self.data.datetime_start
                )
            )

        date_range = [start_date, end_date]
        if year is None:
            date_range = [
                datetime(*[_year, 1, 1, 0, 0, 0])
                for _year in list(range(start_date.year + 1, end_date.year + 1))
            ]
            if start_date - self.dtw < self.data.datetime_start:
                start_date = self.data.datetime_start + self.dtw
            date_range.insert(0, start_date)
            date_range.append(end_date)

        _feature_matrices = []
        ysa = []

        for data_stream in self.data_streams:
            i = 0
            fma = []
            ysa = []
            for time_start, time_end in zip(date_range[:-1], date_range[1:]):
                fmi, ysi = self._extract_features(
                    time_start, time_end - self.dt, data_stream
                )
                i += 1
                if i == 2:
                    pass

                fma.append(fmi)
                ysa.append(ysi)
            fma_df = pd.concat(fma)
            _feature_matrices.append(fma_df)

            if self.debug:
                _feature_matrix_dir = os.path.join(self.output_dir, "_feature_matrix")
                os.makedirs(_feature_matrix_dir, exist_ok=True)

                _feature_matrix_filename = f"{data_stream}.csv"
                _feature_matrix_path = os.path.join(
                    _feature_matrix_dir, _feature_matrix_filename
                )

                fma_df.to_csv(_feature_matrix_path)

                print(f"🔨 Feature Matrix saved to {_feature_matrix_path}")

        # concat list of fM and ysa
        if len(ysa) == 0:
            raise ValueError(f"_load_data > ysa : ysa value is {len(ysa)}")

        if self.debug:
            print(_feature_matrices)

        _label_vector = pd.concat(ysa)
        _feature_matrices = pd.concat(_feature_matrices, axis=1, sort=False)

        if len(_feature_matrices) == 0:
            raise ValueError(
                f"❌ Feature matrix is empty. Tremor data start date is {start_date}. "
                f"Training period is {self.start_date_train} - {self.end_date_train}"
            )

        # del fmi, ysi, fma, ysa
        self.start_date_previous = start_date
        self.end_date_previous = end_date
        self.feature_matrix = _feature_matrices
        self.label_vector = _label_vector

        if self.debug:
            print(f"🔨 Length of feature matrix: {len(_feature_matrices)}")
            print(f"🔨 Length of label vector: {len(_label_vector)}")

        return _feature_matrices, _label_vector

    def forecast(
        self,
        start_date: str = None,
        end_date: str = None,
        recalculate=False,
        model_path: str = None,
        n_jobs: int = None,
        year=None,
    ):
        """Use classifier models to forecast eruption likelihood.
        Parameters:
        -----------
        ti : str, datetime.datetime
            Beginning of forecast period (default is beginning of model analysis period).
        tf : str, datetime.datetime
            End of forecast period (default is end of model analysis period).
        recalculate : bool
            Flag indicating forecast should be recalculated, otherwise forecast will be
            loaded from previous save file (if it exists).
        model_path : None or str
            Optionally pass path to pre-trained model directory in 'models'.
        n_jobs : int
            Number of cores to use.
        Returns:
        --------
        consensus : pd.DataFrame
            The model consensus, indexed by window date.
        """
        year_str = "_{:d}".format(year) if year is not None else ""
        consensus_file = os.path.join(
            self.prediction_dir, f"consensus{year_str}.{self.savefile_type}"
        )
        # confl = "{:s}/consensus{:s}".format(
        #     self.prediction_dir, "{:s}.{:s}".format(year_str, self.savefile_type)
        # )

        if os.path.isfile(consensus_file) and not recalculate:
            return load_dataframe(consensus_file)

        if n_jobs is None:
            n_jobs = self.n_jobs

        self.start_date_forecast = (
            self.start_date_model if start_date is None else to_datetime(start_date)
        )
        self.end_date_forecast = (
            self.end_date_model if end_date is None else to_datetime(end_date)
        )

        if self.end_date_forecast > self.data.datetime_end:
            self.end_date_forecast = self.data.datetime_end
        if self.start_date_forecast - self.dtw < self.data.datetime_start:
            self.start_date_forecast = self.data.datetime_start + self.dtw

        if model_path is not None:
            self._detect_model(model_path=model_path)

        model, classifier = get_classifier(self.classifier)

        if self.debug:
            print(f"🔨 Model: {model}")
            print(f"🔨 Classifier: {classifier}")

        # logic to determine which models need to be run and which to be
        # read from disk
        prefix = type(model).__name__
        models = glob(os.path.join(model_path, f"{prefix}_*.pkl"))

        if self.debug:
            print(f"🔨 Models found: {len(models)}", end="\n\n")

        run_predictions = []
        ys = []
        start_date_list = []

        # create a prediction for each model
        for model in models:
            # change location
            prediction_model = model.replace(model_path, self.prediction_dir)
            # update filetype
            prediction_model_path = prediction_model.replace(
                ".pkl", "{:s}.{:s}".format(year_str, self.savefile_type)
            )

            prediction_model_path = os.path.join(prediction_model_path)

            # check if prediction already exists
            if os.path.isfile(prediction_model_path):
                if self.debug:
                    print(f"🔨 Prediction model found: {prediction_model_path}")

                if recalculate:
                    # delete predictions to be recalculated
                    os.remove(prediction_model_path)
                    run_predictions.append([model, prediction_model_path])
                    start_date_list.append(self.start_date_forecast)
                else:
                    # load an existing prediction
                    y = load_dataframe(
                        prediction_model_path,
                        index_col=0,
                        parse_dates=["time"],
                        infer_datetime_format=True,
                    )
                    # check if prediction spans the requested interval
                    if y.index[-1] < self.start_date_forecast:
                        run_predictions.append([model, prediction_model_path])
                        start_date_list.append(y.index[-1])

                        if self.debug:
                            print(f"🔨 Predictions found: {len(run_predictions)}")
                    else:
                        ys.append(y)
            else:
                # if self.debug:
                #     print(f"🔨 Prediction model not found: {prediction_model_path}")

                run_predictions.append([model, prediction_model_path])
                start_date_list.append(self.start_date_forecast)

        if len(start_date_list) > 0:
            start_date = np.min(start_date_list)

        # generate new predictions
        if len(run_predictions) > 0:
            # load feature matrix
            feature_matrix, _ = self._load_data(
                start_date, self.end_date_forecast, year
            )

            # setup predictor
            p = Pool(n_jobs)
            if n_jobs > 1:
                mapper = p.imap
            else:
                mapper = map

            predict_function = partial(predict_one_model, feature_matrix, model_path)

            # run models with glorious progress bar
            predict_function(run_predictions[0])

            print(f"Length of run_predictions: {len(run_predictions)}")

            for i, y in enumerate(mapper(predict_function, run_predictions)):
                cf = (i + 1) / len(run_predictions)
                if year is None:
                    print(
                        f'forecasting: [{"#" * round(50 * cf) + "-" * round(50 * (1 - cf))}] {100. * cf:.2f}%\r',
                        end="",
                    )
                else:
                    print(
                        f'forecasting {year:d}: [{"#" * round(50 * cf) + "-" * round(50 * (1 - cf))}] {100. * cf:.2f}%\r',
                        end="",
                    )
                ys.append(y)

            if n_jobs > 1:
                p.close()
                p.join()

        # condense data frames and write output
        ys = pd.concat(ys, axis=1, sort=False)
        consensus = np.mean(
            [ys[col].values for col in ys.columns if "pred" in col], axis=0
        )

        # save consensus file
        forecast = pd.DataFrame(consensus, columns=["consensus"], index=ys.index)
        save_dataframe(forecast, consensus_file, index=True, index_label="time")

        # memory management
        if len(run_predictions) > 0:
            # del feature_matrix
            gc.collect()

        return forecast

    def hires_forecast(
        self,
        start_date: str,
        end_date: str,
        recalculate: bool = True,
        save: str = None,
        root: str = None,
        overlap: float = 1.0,
        nz_timezone: bool = False,
        n_jobs: int = None,
        threshold: float = 0.8,
        x_lim: list = None,
    ):
        """Construct forecast at resolution of data.

        Parameters:
        -----------
        start_date : str, datetime.datetime
            Beginning of forecast period.
        end_date : str, datetime.datetime
            End of forecast period.
        recalculate : bool
            Flag indicating forecast should be recalculated, otherwise forecast will be
            loaded from previous save file (if it exists).
        save : None or str
            If given, plot forecast and save to filename.
        root : None or str
            Naming convention for saving feature matrix.
        nz_timezone : bool
            Flag to plot forecast using NZ time zone instead of UTC.
        n_jobs : int
            CPUs to use when forecasting in parallel.
        threshold: float
            Threshold for forecasting.
        x_lim : list
            X-axis limits.

        Notes:
        ------
        Requires to be trained model.
        """
        # error checking
        if self.start_date_train is None:
            raise ValueError("❌ Train model before constructing hi-res forecast.")

        if save is None:
            save = os.path.join(
                self.plot_dir,
                f"hires_forecast_{self.classifier}_{self.station}_{start_date}-{end_date}.png",
            )

        if n_jobs is not None:
            self.n_jobs = n_jobs

        # calculate hires feature matrix
        if root is None:
            root = f"{self.root}_hires"

        _fm = ForecastModel(
            station=self.station,
            start_date=start_date,
            end_date=end_date,
            window=self.window,
            overlap=overlap,
            look_forward=self.look_forward,
            eruptive_file=self.eruptive_file,
            tremor_data_file=self.tremor_data_file,
            data_streams=self.data_streams,
            root=root,
            feature_root=root,
            feature_dir=self.feature_dir,
            n_jobs=self.n_jobs,
            savefile_type=self.savefile_type,
            verbose=self.verbose,
            debug=self.debug,
        )

        _fm.compute_only_features = list(
            set([ft.split("__")[1] for ft in self._collect_features()[0]])
        )

        for data_stream in self.data_streams:
            _fm._extract_features(
                to_datetime(start_date), to_datetime(end_date), data_stream
            )

        # predict on hires features
        # load from consensus file
        label_vector = _fm.forecast(
            start_date, end_date, recalculate, model_path=self.model_dir, n_jobs=n_jobs
        )

        if save is not None:
            self._plot_hires_forecast(
                label_vector,
                save,
                threshold,
                nz_timezone=nz_timezone,
                xlim=x_lim,
            )

        return label_vector

    def _new_plot_hires_forecast(self, df, filename):
        new_df = df.resample("1h").mean()
        ci = self._compute_confidence_interval(df["consensus"])

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 2.5))

        ax.plot(
            df.index,
            df["consensus"],
            color="black",
            linewidth=1,
            label="Consensus",
            alpha=0.2,
        )
        ax.plot(
            new_df.index,
            new_df["consensus"],
            color="k",
            linewidth=1,
            label="Resampled 1D",
        )

        ax.fill_between(
            df.index,
            (df["consensus"] - ci),
            (df["consensus"] + ci),
            color="gray",
            alpha=0.2,
        )
        ax.fill_between(
            df.index,
            0,
            1.0,
            where=(df["consensus"] > 0.7),
            color="red",
            alpha=0.3,
            label="Forecast",
            zorder=-3,
        )

        ax.set_ylim(0, 1.0)
        ax.set_xlim(df.index[0], df.index[-1])

        ax.axhline(y=0.7, color="k", linestyle="--", linewidth=1.5, label="Threshold")
        ax.set_ylabel("Consensus", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

        ax.axvline(
            datetime.strptime("2025-05-18", "%Y-%m-%d"),
            color="red",
            linestyle="--",
            label="Eruption",
        )
        ax.legend(loc="upper left", ncol=2, fontsize=8, frameon=False)

        for label in ax.get_xticklabels(which="major"):
            label.set(rotation=15, horizontalalignment="right")

        plt.tight_layout()
        plt.savefig(filename, dpi=300)

    def _plot_hires_forecast(
        self,
        label_vector,
        save,
        threshold=0.75,
        nz_timezone=False,
        xlim=None,
    ):
        """Plot model hires version of model forecast (single axes).
        Parameters:
        -----------
        label_vector : pandas.DataFrame
            Model forecast returned by ForecastModel.forecast.
        threshold : float
            Threshold consensus to declare alert.
        save : str
            File name to save figure.
        """
        # set up figures and axes
        f = plt.figure(figsize=(8, 4))
        ax = plt.axes([0.1, 0.08, 0.8, 0.8])

        time_index = pd.to_datetime(label_vector.index.values)
        time_index_start = time_index[0]
        time_index_end = time_index[-1]

        # if True:  # plot filtered data
        if "zsc_rsamF" in self.data_streams and "rsamF" not in self.data_streams:
            rsam = self.data.get_data(time_index_start, time_index_end)["zsc_rsamF"]
        else:
            rsam = self.data.get_data(time_index_start, time_index_end)["rsamF"]
        # else:
        #     if "zsc_rsam" in self.data_streams and "rsam" not in self.data_streams:
        #         rsam = self.data.get_data(t[0], t[-1])["zsc_rsam"]
        #     else:
        #         rsam = self.data.get_data(t[0], t[-1])["rsam"]

        time_index_rsam = rsam.index
        time_index_rsam_end = time_index_rsam[-1]
        time_index_rsam_start = time_index_rsam[0]

        if nz_timezone:
            time_index = to_nz_timezone(time_index)
            time_index_rsam = to_nz_timezone(time_index_rsam)
            ax.set_xlabel("Local time")
        else:
            ax.set_xlabel("UTC")

        y = np.mean(
            np.array([label_vector[col] for col in label_vector.columns]), axis=0
        )

        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0, 0.25, 0.50, 0.75, 1.00])
        ax.set_ylabel("ensemble mean")

        # consensus threshold
        ax.axhline(
            threshold, color="k", linestyle=":", label="alert threshold", zorder=4
        )

        # modelled alert
        ax.plot(time_index, y, "c-", label="ensemble mean", zorder=4, lw=0.75)
        ci = self._compute_confidence_interval(y)

        ax.fill_between(time_index, (y - ci), (y + ci), color="c", zorder=5, alpha=0.3)
        # ax_ = ax.twinx()
        # ax_.set_ylabel("RSAM [$\mu$m s$^{-1}$]")
        # ax_.set_ylim([0, 5])
        # ax_.plot(time_index_rsam, rsam.values * 1.0e-3, "k-", lw=0.75)

        for tii, yi in zip(time_index, y):
            if yi > threshold:
                ax.fill_between(
                    [tii, tii + self.dtf], [0, 0], [100, 100], color="y", zorder=3
                )

        for te in self.data.tes:
            ax.axvline(te, color="r", linestyle="--", zorder=10)

        ax.plot([], [], "r--", label="eruption")
        ax.fill_between([], [], [], color="y", label="eruption forecast")
        # ax.plot([], [], "k-", lw=0.75, label="RSAM")

        ax.legend(loc=2, ncol=2)

        time_index_max = (
            time_index_end
            if time_index_end > time_index_rsam_end
            else time_index_rsam_end
        )
        time_index_min = (
            time_index_start
            if time_index_start < time_index_rsam_start
            else time_index_rsam_start
        )

        if xlim is None:
            xlim = [time_index_min, time_index_max]

        tf = time_index_max
        t0 = tf.replace(hour=0, minute=0, second=0)
        dt = (time_index_max - time_index_min).total_seconds()

        if self.verbose:
            print(f"dt values = {dt}")

        if dt < 10.0 * 24 * 3600:
            n_days = int(np.ceil(dt / (24 * 3600)))
            xts = [t0 - timedelta(days=i) for i in range(n_days)][::-1]
            label_xts = [xt.strftime("%d %b") for xt in xts]
        elif dt < 20.0 * 24 * 3600:
            n_days = int(np.ceil(dt / (24 * 3600)) / 2)
            xts = [t0 - timedelta(days=2 * i) for i in range(n_days)][::-1]
            label_xts = [xt.strftime("%d %b") for xt in xts]
        elif dt < 70.0 * 24 * 3600:
            n_days = int(np.ceil(dt / (24 * 3600)) / 7)
            xts = [t0 - timedelta(days=7 * i) for i in range(n_days)][::-1]
            label_xts = [xt.strftime("%d %b") for xt in xts]
        elif dt < 365.25 * 24 * 3600:
            t0 = tf.replace(day=1, hour=0, minute=0, second=0)
            n_months = int(np.ceil(dt / (24 * 3600 * 365.25 / 12)))
            xts = [t0 - timedelta(days=i * 365.25 / 12) for i in range(n_months)][::-1]
            label_xts = [xt.strftime("%b") for xt in xts]
        # elif dt < 2 * 365.25 * 24 * 3600:
        else:
            t0 = tf.replace(day=1, hour=0, minute=0, second=0)
            n_months = int(np.ceil(dt / (24 * 3600 * 365.25 / 12)) / 2)
            xts = [t0 - timedelta(days=2 * i * 365.25 / 12) for i in range(n_months)][
                ::-1
            ]
            label_xts = [xt.strftime("%b %Y") for xt in xts]

        ax.set_xticks(xts)
        ax.set_xticklabels(label_xts)

        ax.set_xlim(xlim)
        # ax_.set_xlim(xlim)

        bbox = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        ax.text(
            0.85,
            0.95,
            self.data.station + " " + label_vector.index[-1].strftime("%Y"),
            size=12,
            ha="left",
            va="top",
            transform=ax.transAxes,
            bbox=bbox,
        )

        plt.savefig(save, dpi=400)
        plt.close(f)

    def _compute_confidence_interval(self, y):
        """Computes a 95% confidence interval of the model consensus.
        Parameters:
        -----------
        y : numpy.array
            Model consensus returned by ForecastModel.forecast.

        Returns:
        --------
        ci : numpy.array
            95% confidence interval of the model consensus
        """
        confidence_interval = 1.96 * (np.sqrt(y * (1 - y) / self.number_of_classifiers))
        return confidence_interval

    def _extract_features(
        self,
        start_date: datetime,
        end_date: datetime,
        data_stream,
    ) -> (pd.DataFrame, pd.DataFrame):
        """
        Extract features from windowed data.

        Notes:
        ------
        Saves feature matrix to $rootdir/features/$root_features.csv to avoid recalculation.
        """
        # print(f"start_date = {start_date}")
        # print(f"end_date = {end_date}")
        # print(f"data_stream = {data_stream}", end="\n\n")

        # number of windows in feature request
        # orginally Nw
        ceiled = np.ceil(((end_date - start_date) / self.dt) / (self.iw - self.io))
        number_of_windows = int(ceiled) + 1

        if self.debug:
            print("---- Extracting features ---")
            print(f"data_stream = {data_stream}")
            print(f"start_date = {start_date}")
            print(f"end_date = {end_date}")
            print(f"end_date - start_date : {end_date-start_date}")
            print(f"self.dt : {self.dt}")
            print(f"Nw floored : {ceiled}")
            print(f"number_of_windows : {number_of_windows}")
            print("---- Features Extracted ---", end="\n\n")

        # max number of construct windows per iteration (6*24*30 windows: ~ a month of hires, overlap of 1.)
        max_number_of_windows = 6 * 24 * 31

        # file naming convention
        year = start_date.year
        feature_file = self._feature_file(data_stream, year)

        if self.debug:
            print(f"🔨 Feature matrix: {feature_file}")

        feature_matrix = pd.DataFrame()

        # condition on the existence of fm save for the year requested
        if os.path.isfile(feature_file):
            # check if feature matrix file exists
            # load existing feature matrix
            if self.debug:
                print(f"🔨 Feature file exists in : {feature_file}")

            feature_matrix_preloaded = load_dataframe(
                feature_file,
                index_col=0,
                parse_dates=["time"],
                infer_datetime_format=True,
                header=0,
            )

            if self.debug:
                print(
                    f"🔨 Length of preloaded feature matrix  : {len(feature_matrix_preloaded)}",
                    end="\n\n",
                )

            # request for features, labeled by index
            label_1 = [
                np.datetime64(start_date + index_number_of_windows * self.dto)
                for index_number_of_windows in range(number_of_windows)
            ]

            # read the existing feature matrix file (index column only) for current computed features
            # identify new features for calculation
            # alternative to last to commands
            label_2 = load_dataframe(
                feature_file,
                index_col=0,
                parse_dates=["time"],
                usecols=["time"],
                infer_datetime_format=True,
            ).index.values

            label_3 = []
            [
                label_3.append(index_label_1.astype(datetime))
                for index_label_1 in label_1
                if index_label_1 not in label_2
            ]

            # end testing
            # check is new features need to be calculated (windows)
            if len(label_3) == 0:  # all features requested already calculated
                # load requested features (by index) and generate fm
                feature_matrix = feature_matrix_preloaded[
                    feature_matrix_preloaded.index.isin(label_1, level=0)
                ]
                # del feature_matrix_preloaded, label_1, label_2, label_3

            else:
                # calculate new features and add to existing saved feature matrix
                # note: if len(l3) too large for max number of construct windows (say Nmax) l3 is chunked
                # into subsets smaller of Nmax and call construct_windows/extract_features on these subsets
                if len(label_3) >= max_number_of_windows:
                    # condition on length of requested windows
                    # divide l3 in subsets

                    if self.debug:
                        print(f">>> L3 >= Nmax: {len(label_3)}")

                    n_sbs = int(number_of_windows / max_number_of_windows) + 1

                    def chunks(_list, n):
                        # Yield successive n-sized chunks from lst
                        for i in range(0, len(_list), n):
                            yield _list[i : i + n]

                    label_3_subsets = chunks(label_3, int(number_of_windows / n_sbs))

                    # copy existing feature matrix (to be filled and save)
                    _feature_matrix_preloaded = pd.concat([feature_matrix_preloaded])

                    # loop over subsets
                    for label_3_subset in label_3_subsets:
                        # generate dataframe for subset
                        _feature_matrix_new = self._construct_windows_extract_feature(
                            number_of_windows,
                            start_date,
                            data_stream,
                            index=label_3_subset,
                        )

                        # concatenate subset with existing feature matrix
                        feature_matrix: pd.DataFrame = pd.concat(
                            [_feature_matrix_preloaded, _feature_matrix_new]
                        )

                        # del _feature_matrix_new

                        # sort new updated feature matrix and save (replace existing one)
                        feature_matrix.sort_index(inplace=True)
                        save_dataframe(
                            feature_matrix, feature_file, index=True, index_label="time"
                        )
                else:
                    if self.debug:
                        print(f"<<< L3 < Nmax: {len(label_3)}")

                    # generate dataframe
                    _feature_matrix_new = self._construct_windows_extract_feature(
                        number_of_windows, start_date, data_stream, index=label_3
                    )
                    feature_matrix = pd.concat(
                        [feature_matrix_preloaded, _feature_matrix_new]
                    )

                    # sort new updated feature matrix and save (replace existing one)
                    feature_matrix.sort_index(inplace=True)
                    save_dataframe(
                        feature_matrix, feature_file, index=True, index_label="time"
                    )

                # keep in feature matrix (in memory) only the requested windows
                feature_matrix = feature_matrix[
                    feature_matrix.index.isin(label_1, level=0)
                ]
                #
                # del feature_matrix_preloaded, label_1, label_2, label_3

        else:
            ## create feature matrix from scratch
            if self.debug:
                print(f"🔨 Feature file not exists. Create feature matrix from scratch.")
            year = start_date.year
            feature_file = self._feature_file(data_stream, year)

            # note: if Nw is too large for max number of construct windows (say Nmax) the request is chunk
            # into subsets smaller of Nmax and call construct_windows/extract_features on these subsets
            if number_of_windows >= max_number_of_windows:
                # condition on length of requested windows
                # divide request in subsets
                number_of_subset = int(number_of_windows / max_number_of_windows) + 1

                def split_num(num, div):
                    # list of number of elements subsets of num divided by div
                    return [
                        num // div + (1 if x < num % div else 0) for x in range(div)
                    ]

                number_of_windows_list = split_num(number_of_windows, number_of_subset)
                ## fm for first subset
                # generate dataframe
                feature_matrix = self._construct_windows_extract_feature(
                    number_of_windows_list[0], start_date, data_stream
                )
                # aux intial time (vary for each subset)
                ti_aux = start_date + (number_of_windows_list[0]) * self.dto
                # loop over the rest subsets
                for index_number_of_windows in number_of_windows_list[1:]:
                    # generate dataframe
                    fm_new = self._construct_windows_extract_feature(
                        index_number_of_windows, ti_aux, data_stream
                    )
                    # concatenate
                    feature_matrix = pd.concat([feature_matrix, fm_new])
                    # increase aux ti
                    ti_aux = ti_aux + index_number_of_windows * self.dto

                save_dataframe(
                    feature_matrix, feature_file, index=True, index_label="time"
                )
                # end working section
                # del fm_new
            else:
                year = start_date.year
                feature_file = self._feature_file(data_stream, year)
                # generate dataframe
                fm = self._construct_windows_extract_feature(
                    number_of_windows, start_date, data_stream
                )
                save_dataframe(fm, feature_file, index=True, index_label="time")

        # Label vector corresponding to data windows
        if len(feature_matrix) > 0:
            label_vector = pd.DataFrame(
                self._get_label(feature_matrix.index.values),
                columns=["label"],
                index=feature_matrix.index,
            )

            vector_file = os.path.join(
                self.feature_dir, f"label_{data_stream}_{self.station}.csv"
            )
            label_vector.to_csv(vector_file)

            gc.collect()
            return feature_matrix, label_vector

        return pd.DataFrame(), pd.DataFrame()

    def _feature_file(self, data_stream, year):
        return self.feature_file("_{:d}".format(year), data_stream)

    def _construct_windows_extract_feature(
        self, number_of_windows, start_date, data_stream, index=None
    ) -> pd.DataFrame:
        """Construct windows, extract features and return dataframe"""
        cfp = ComprehensiveFCParameters()

        if self.compute_only_features:
            cfp = dict(
                [(k, cfp[k]) for k in cfp.keys() if k in self.compute_only_features]
            )
        else:
            # drop features if relevant
            _ = [cfp.pop(df) for df in self.drop_features if df in list(cfp.keys())]

        kw = {
            "column_id": "id",
            "n_jobs": self.n_jobs,
            "default_fc_parameters": cfp,
            "impute_function": impute,
            "show_warnings": self.show_warnings,
        }

        # construct_windows/extract_features for subsets
        df, wd = self._construct_windows(
            number_of_windows, start_date, data_stream, index=index
        )

        # extract features and generate feature matrixs
        feature_matrix = self._extract_features_x(df, **kw)
        feature_matrix.index = pd.Series(wd)
        feature_matrix.index.name = "time"

        if len(feature_matrix) > 0:
            _path = os.path.join(self.output_dir, "_extract_features")
            os.makedirs(_path, exist_ok=True)

            _start_date = start_date.strftime("%Y-%m-%d")
            _save_path = os.path.join(
                _path, f"{self.station}_{data_stream}_{_start_date}.xlsx"
            )
            feature_matrix.to_excel(_save_path)

        return feature_matrix

    def _construct_windows(
        self,
        number_of_windows,
        start_date: datetime,
        data_stream,
        i0=0,
        i1=None,
        index: list = None,
    ):
        if index is None:
            index = []
        """
        Create overlapping data windows for feature extraction.
        """
        if i1 is None:
            i1 = number_of_windows
        if len(index) == 0:
            # get data for windowing period
            df = self.data.get_data(
                start_date - self.dtw, start_date + (number_of_windows - 1) * self.dto
            )[
                [
                    data_stream,
                ]
            ]

            # create windows
            dfs = []
            for i in range(i0, i1):
                dfi = df[:].iloc[
                    i * (self.iw - self.io) : i * (self.iw - self.io) + self.iw
                ]

                try:
                    dfi["id"] = pd.Series(
                        np.ones(self.iw, dtype=int) * i, index=dfi.index
                    )
                except ValueError as e:
                    raise ValueError(
                        f"_construct_windows :: This shouldn't be happening: {e}"
                    )

                dfs.append(dfi)

            df = pd.concat(dfs)
            window_dates = [start_date + i * self.dto for i in range(number_of_windows)]
            return df, window_dates[i0:i1]
        else:
            # get data for windowing define in index
            dfs = []
            for i, ind in enumerate(index):  # loop over index
                ind = np.datetime64(ind).astype(datetime)
                dfi = self.data.get_data(ind - self.dtw, ind)[[data_stream]].iloc[:]

                try:
                    dfi["id"] = pd.Series(
                        np.ones(self.iw, dtype=int) * i, index=dfi.index
                    )
                except ValueError as e:
                    _, _, tb = sys.exc_info()
                    print(f"❌ _construct_windows :: This shouldn't be happening: {e}")

                dfs.append(dfi)
            df = pd.concat(dfs)
            window_dates = index
            return df, window_dates

    def _extract_features_x(self, df: pd.DataFrame, **kw) -> pd.DataFrame:
        t0 = df.index[0] + self.dtw
        t1 = df.index[-1] + self.dt

        if self.verbose:
            print(
                "{:s} feature extraction {:s} to {:s}".format(
                    df.columns[0], t0.strftime("%Y-%m-%d"), t1.strftime("%Y-%m-%d")
                )
            )

        # tsfresh extract features
        return extract_features(df, **kw)

    def _get_label(self, ts):
        """Compute label vector."""
        ys = [
            self.data.is_eruption_in(days=self.look_forward, from_time=t)
            for t in pd.to_datetime(ts)
        ]
        return ys

    def _drop_features(
        self, feature_matrix: pd.DataFrame, drop_features
    ) -> pd.DataFrame:
        """Drop columns from feature matrix.
        Parameters:
        -----------
        feature_matrix : pd.DataFrame
            Matrix to drop columns.
        drop_features : list
            tsfresh feature names or calculators to drop from matrix.
        Returns:
        --------
        feature_matrix : pd.DataFrame
            Reduced matrix.
        """
        self.drop_features = drop_features
        if len(self.drop_features) > 0:
            if self.verbose:
                print(f"dropping features : {self.drop_features}")

            comprehensive_features = ComprehensiveFCParameters()
            df2 = []
            for df in self.drop_features:
                if df in feature_matrix.columns:
                    df2.append(df)  # exact match
                else:
                    if df in comprehensive_features.keys() or df in [
                        "fft_coefficient_hann"
                    ]:
                        df = "*__{:s}__*".format(df)  # feature calculator
                    # wildcard match
                    df2 += [col for col in feature_matrix.columns if fnmatch(col, df)]
            feature_matrix = feature_matrix.drop(columns=df2)
        return feature_matrix

    def _exclude_dates(
        self,
        feature_matrix: pd.DataFrame,
        label_vector: pd.DataFrame,
        exclude_dates: None,
    ):
        """Drop rows from feature matrix and label vector.
        Parameters:
        -----------
        feature_matrix : pd.DataFrame
            Matrix to drop columns.
        y : pd.DataFrame
            Label vector.
        exclude_dates : list
            list of time windows to exclude during training. Facilitates dropping of eruption
            windows within analysis period. E.g., exclude_dates = [['2012-06-01','2012-08-01'],
            ['2015-01-01','2016-01-01']] will drop Jun-Aug 2012 and 2015-2016 from analysis.
        Returns:
        --------
        Xr : pd.DataFrame
            Reduced matrix.
        yr : pd.DataFrame
            Reduced label vector.
        """
        self.exclude_dates = exclude_dates
        if self.exclude_dates is not None:
            if len(self.exclude_dates) > 0:
                for exclude_date_range in self.exclude_dates:
                    t0, t1 = [to_datetime(dt) for dt in exclude_date_range]
                    indices = (label_vector.index < t0) | (label_vector.index >= t1)
                    feature_matrix = feature_matrix.loc[indices]
                    label_vector = label_vector.loc[indices]

        return feature_matrix, label_vector

    def _collect_features(self, save=None):
        """Aggregate features used to train classifiers by frequency.
        Parameters:
        -----------
        save : None or str
            If given, name of file to save feature frequencies. Defaults to all.fts
            if model directory.
        Returns:
        --------
        labels : list
            Feature names.
        freqs : list
            Frequency of feature appearance in classifier models.
        """
        if save is None:
            save = os.path.join(self.model_dir, "all.fts")

        features = []
        feature_files = glob(os.path.join(self.model_dir, "*.fts"))

        for index, feature_file in enumerate(feature_files):
            if feature_file.split(os.sep)[-1].split(".")[0] in ["all", "ranked"]:
                continue
            with open(feature_file) as fp:
                lines = fp.readlines()
            features += [" ".join(ln.rstrip().split()[1:]) for ln in lines]

        labels = list(set(features))
        frequencies = [features.count(label) for label in labels]
        labels = [label for _, label in sorted(zip(frequencies, labels))][::-1]
        frequencies = sorted(frequencies)[::-1]

        # write out feature frequencies
        with open(save, "w") as fp:
            _ = [
                fp.write("{:d},{:s}\n".format(freq, ft))
                for freq, ft in zip(frequencies, labels)
            ]
        return labels, frequencies
