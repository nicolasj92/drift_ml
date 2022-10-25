from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

from .univariate import SimpleSTD


class PCADriftDetector:
    def __init__(
        self,
        min_explained_var: float = 0.65,
        score_drift_detector=SimpleSTD(),
        seed=None,
    ) -> None:
        self.score_drift_detector = score_drift_detector
        self.min_explained_var = min_explained_var
        self.seed = seed
        self.fitted = False

    @property
    def drift_detected(self) -> bool:
        return self.score_drift_detector.drift_detected

    def fit(self, reference_data) -> None:
        self.scaler = StandardScaler()
        scaled_reference_data = self.scaler.fit_transform(reference_data)

        self.pca = PCA(n_components=self.min_explained_var, random_state=self.seed)
        transformed_reference_data = self.pca.fit_transform(scaled_reference_data)

        # NOTE: It might be necessary to split the data into train/test to
        #       meaningfully train the score_drift_detector
        reconstructed_reference_data = self.pca.inverse_transform(
            transformed_reference_data
        )
        mse_reference_data = np.linalg.norm(
            (reconstructed_reference_data - scaled_reference_data), axis=1
        )
        self.score_drift_detector.fit(mse_reference_data)
        self.fitted = True

    def update(self, X) -> None:
        if not self.fitted:
            raise Exception
        if len(X.shape) == 1:
            X = np.expand_dims(X, axis=0)

        scaled_X = self.scaler.transform(X)
        transformed_X = self.pca.transform(scaled_X)
        reconstructed_X = self.pca.inverse_transform(transformed_X)
        mse_X = np.linalg.norm((reconstructed_X - scaled_X), axis=1)

        score = self.score_drift_detector.update(mse_X)
        return score
