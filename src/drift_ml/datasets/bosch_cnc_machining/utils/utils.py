import pandas as pd
import numpy as np
from scipy.signal import stft
from scipy.spatial.transform import Rotation
from tsfresh import extract_features


def sample_stft(sample, fs=2000):
    specs = []
    for c in range(sample.shape[1]):
        _, _, Zxx = stft(sample[:, c], fs=fs)
        specs.append(np.abs(Zxx))
    specs = np.stack(specs, axis=0)
    return specs


def augment_xyz_samples(data, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    rot_matrix = Rotation.from_euler(
        "zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True
    ).as_matrix()
    augmented_data = np.dot(data, rot_matrix.T)
    return augmented_data


def extract_tsfresh_features(samples, featureset):
    n_samples = samples.shape[0]
    sample_length = samples.shape[1]

    pd_X = samples.reshape((-1, 3))
    pd_X = pd.DataFrame(data=pd_X, columns=("X", "Y", "Z"))
    pd_X["time"] = np.tile(np.arange(sample_length), n_samples)
    pd_X["id"] = np.repeat(np.arange(n_samples), sample_length)
    extracted_features = extract_features(
        pd_X,
        column_id="id",
        column_sort="time",
        default_fc_parameters=featureset,
        chunksize=10,
    )
    return extracted_features.to_numpy()
