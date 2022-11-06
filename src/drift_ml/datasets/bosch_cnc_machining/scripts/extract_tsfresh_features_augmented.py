import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from tsfresh import feature_extraction

from drift_ml.datasets.bosch_cnc_machining.utils.dataloader import NPYBoschCNCDataLoader
from drift_ml.datasets.bosch_cnc_machining.utils.utils import (
    augment_xyz_samples,
    extract_tsfresh_features,
)

if __name__ == "__main__":
    loader = NPYBoschCNCDataLoader(
        metadata_path="/home/tbiegel/nico_files/bosch_cnc_machining/features_and_data/metadata_ws4096.pkl"
    )
    loader.load_data(
        "/home/tbiegel/nico_files/bosch_cnc_machining/features_and_data/sample_data_x_raw_ws4096.npy",
        "/home/tbiegel/nico_files/bosch_cnc_machining/features_and_data/sample_data_y_raw_ws4096.npy",
    )

    features = pd.read_pickle(
        "/home/tbiegel/nico_files/bosch_cnc_machining/features_and_data/all_top_30_features.pkl"
    )
    feature_settings = feature_extraction.settings.from_columns(features)

    target_path = "/home/tbiegel/nico_files/drift_ml/src/drift_ml/datasets/bosch_cnc_machining/raw_data/augmented"

    min_degrees = 5.0
    max_degrees = 60.0
    step_degrees = 5.0

    for shift_degrees in np.arange(min_degrees, max_degrees, step_degrees):
        print(f"Processing features for a shift of {shift_degrees}...")

        # First, yaw
        filepath = os.path.join(
            target_path, f"tsfresh_top30_yaw_shift_{shift_degrees}_deg.pkl"
        )
        augmented_samples = augment_xyz_samples(
            loader.sample_data_X, yaw_deg=shift_degrees
        )
        extracted_features = extract_tsfresh_features(
            augmented_samples, feature_settings
        )
        extracted_features.to_pickle(filepath)

        # Second, pitch
        filepath = os.path.join(
            target_path, f"tsfresh_top30_pitch_shift_{shift_degrees}_deg.pkl"
        )
        augmented_samples = augment_xyz_samples(
            loader.sample_data_X, pitch_deg=shift_degrees
        )
        extracted_features = extract_tsfresh_features(
            augmented_samples, feature_settings
        )
        extracted_features.to_pickle(filepath)

        # Last, roll
        filepath = os.path.join(
            target_path, f"tsfresh_top30_roll_shift_{shift_degrees}_deg.pkl"
        )
        augmented_samples = augment_xyz_samples(
            loader.sample_data_X, roll_deg=shift_degrees
        )
        extracted_features = extract_tsfresh_features(
            augmented_samples, feature_settings
        )
        extracted_features.to_pickle(filepath)
