import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation
from tsfresh import extract_features


def augment_xyz_samples(data, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    rot_matrix = Rotation.from_euler(
        "zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True
    ).as_matrix()
    augmented_data = np.einsum("ii,jki->jki", rot_matrix, data)
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
        kind_to_fc_parameters=featureset,
    )
    return extracted_features
