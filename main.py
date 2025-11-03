from whakaari import TremorData
import os
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

def main():
    tremor_data = TremorData(
        station="OJN",
        data_dir=os.getcwd(),
        eruptive_file=r"D:\Project\whakaari\input\OJN_eruptive_periods.txt",
        n_jobs=16,
        verbose=True,
    )

    tremor_data.stations = {
        "OJN": {
            "client_name": "GEONET",
            "client_url": "https://service.geonet.org.nz",
            "channel": "EHN",
            "network": "VG",
            "location": "00",
        },
    }

    tremor_data.update(
        datetime_start="2025-01-01 00:00:00",
        datetime_end="2025-08-24 23:59:59",
        sds_dir=r"G:\OJN\Converted\SDS",
    )


if __name__ == "__main__":
    main()
