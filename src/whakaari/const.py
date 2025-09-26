STATIONS = {
    "WIZ": {
        "client_name": "GEONET",
        "client_url": "https://service.geonet.org.nz",
        "channel": "HHZ",
        "network": "NZ",
        "location": "*",
    },
    "KRVZ": {
        "client_name": "GEONET",
        "client_url": "https://service-nrt.geonet.org.nz",
        "channel": "EHZ",
        "network": "NZ",
        "location": "*",
    },
    "FWVZ": {
        "client_name": "GEONET",
        "client_url": "https://service-nrt.geonet.org.nz",
        "channel": "HHZ",
        "network": "NZ",
        "location": "*",
    },
    "PVV": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "PV6": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "OKWR": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "VNSS": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "SSLW": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "REF": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
    "BELO": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "HHZ",
        "network": "YC",
        "location": "*",
    },
    "CRPO": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "HHZ",
        "network": "OV",
        "location": "*",
    },
    "IVGP": {
        "client_name": "https://webservices.ingv.it",
        "client_url": "https://webservices.ingv.it",
        "channel": "HHZ",
        "network": "IV",
        "location": "*",
    },
    "AUS": {
        "client_name": "IRIS",
        "client_url": "https://service.iris.edu",
        "channel": "EHZ",
        "network": "AV",
        "location": "*",
    },
}

RATIO_NAMES = ["vlar", "lrar", "rmar", "dsar"]
BAND_NAMES = ["vlf", "lf", "rsam", "mf", "hf"]

FREQ_BANDS = [
    [0.01, 0.1],  # vlf
    [0.1, 2],  # lf
    [2, 5],  # rsam
    [4.5, 8],  # mf
    [8, 16],  # hf
]

"""
SVM - Support Vector Machine.
KNN - k-Nearest Neighbors
DT - Decision Tree
RF - Random Forest
NN - Neural Network
NB - Naive Bayes
LR - Logistic Regression
"""
CLASSIFIERS = [
    {
        "code": "SVM",
        "name": "Support Vector Machine",
    },
    {
        "code": "KNN",
        "name": "K-Nearest Neighbors",
    },
    {
        "code": "DT",
        "name": "Decision Tree",
    },
    {
        "code": "RF",
        "name": "Random Forest",
    },
    {
        "code": "NN",
        "name": "Neural Network",
    },
    {
        "code": "NB",
        "name": "Naive Bayes",
    },
    {
        "code": "LR",
        "name": "Logistic Regression",
    },
]
