#%%
from whakaari import ForecastModel
import warnings

warnings.simplefilter("ignore")


#%%
def main(
        start_date_train: str = None,
        end_date_train: str = None,
        start_date_forecast: str = None,
        end_date_forecast: str = None,
        station: str = None,
        classifier = "DT"
):
    data_streams = ["rsam", "mf", "hf", "dsar"]

    fm = ForecastModel(
        station="OJN",
        start_date=start_date_train,
        end_date=end_date_train,
        window=2.0,
        overlap=0.75,
        look_forward=2.0,
        eruptive_file=r"D:\Project\whakaari\input\OJN_eruptive_periods.txt",
        tremor_data_file=r"D:\Project\whakaari\output\OJN_tremor_data.csv",
        data_streams=data_streams,
        verbose=True,
    )

    drop_features = ["linear_trend_timewise", "agg_linear_trend"]

    fm.train(
        start_date=start_date_train,
        end_date=end_date_train,
        drop_features=drop_features,
        retrain_model=True,
        classifier=classifier,
        n_jobs=16,
    )

    filename = f"{station}_{classifier}_forecast_{start_date_forecast}_{end_date_forecast}.png"

    fm.hires_forecast(
        start_date=start_date_forecast,
        end_date=end_date_forecast,
        recalculate=True,
        save=filename,
        nz_timezone=False,
        threshold=0.7,
        n_jobs=16
    )


#%%
if __name__ == "__main__":
    main(
        start_date_train = "2025-01-01",
        end_date_train = "2025-03-31",
        start_date_forecast= "2025-04-01",
        end_date_forecast= "2025-08-22",
        station = "OJN",
        classifier = "RF"
    )
