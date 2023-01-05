import logging
import numpy as np
from scipy.special import entr
from scipy.stats import entropy
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)

from drift_ml.utils.utils import in_notebook


if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class DriftExperiment:
    def __init__(
        self,
        model,
        dataloader,
        fit_model=True,
        fit_kwargs={},
        name="test_run",
        result_dir="",
        length=None,
        drift_detector=None,
        drift_detector_update_freq=1,
        retrain_at_drift=False,
        retrain_new_samples=1000,
        retrain_with_train_set=False,
        window_size=100,
        chunksize=1000,
    ):
        self.name = name
        self.result_dir = result_dir

        self.model = model
        self.fit_model = fit_model
        self.fit_kwargs = fit_kwargs
        self.dataloader = dataloader

        if length is None:
            self.length = self.dataloader.max_drift_data_length
        else:
            self.length = length

        self.chunksize = min(chunksize, self.length)

        self.drift_detector = drift_detector
        self.drift_detector_update_freq = drift_detector_update_freq

        self.retrain_at_drift = retrain_at_drift
        self.retrain_new_samples = retrain_new_samples
        self.retrain_with_train_set = retrain_with_train_set

        self.window_size = window_size

        self.metrics_pred = {
            "accuracy": accuracy_score,
            "mcc": matthews_corrcoef,
            "f1": f1_score,
            "precision": precision_score,
            "recall": lambda x, y: recall_score(x, y, zero_division=0),
        }
        self.metrics_score = {
            # "auroc": roc_auc_score,
            # "ap": average_precision_score,
        }

    def _fit_model(self):
        X_train, y_train = self.dataloader.access_base_samples(dataset="train")
        self.model.fit(X_train, y_train, **self.fit_kwargs)

    def run(self):
        if self.fit_model:
            # Fit model initially on training data
            logging.info("Doing the initial training of the model.")
            self._fit_model()

        # Create arrays to hold history
        self.y_true = np.full((self.length), np.nan)
        self.y_pred = np.full((self.length), np.nan)
        self.y_pred_proba = np.full((self.length, 2), np.nan)
        self.y_pred_entr = np.full((self.length), np.nan)

        # Create arrays to hold metrics
        self.metric_results_pred = {
            name: np.full((self.length), np.nan) for name in self.metrics_pred
        }
        self.metric_results_score = {
            name: np.full((self.length), np.nan) for name in self.metrics_score
        }

        drift_detected_indices = []
        i_chunk_start = 0

        # Iterate through data in chunks
        with tqdm(total=self.length) as pbar:
            while i_chunk_start < self.length:

                # Load next chunk of data
                X_chunk, y_true_chunk = self.dataloader.access_test_drift_samples(
                    index=i_chunk_start, length=self.chunksize
                )

                # Get model output
                y_pred_chunk = self.model.predict(X_chunk)
                y_pred_proba_chunk = self.model.predict_proba(X_chunk)
                y_pred_entr_chunk = entropy(y_pred_proba_chunk, axis=1)

                # Append model output to history
                self.y_true[
                    i_chunk_start : i_chunk_start + self.chunksize
                ] = y_true_chunk
                self.y_pred[
                    i_chunk_start : i_chunk_start + self.chunksize
                ] = y_pred_chunk
                self.y_pred_proba[
                    i_chunk_start : i_chunk_start + self.chunksize
                ] = y_pred_proba_chunk
                self.y_pred_entr[
                    i_chunk_start : i_chunk_start + self.chunksize
                ] = y_pred_entr_chunk

                has_retrained = False

                # Go through each sample of the chunk and predicitons
                for i_chunk_sample in range(self.chunksize):
                    i_sample = i_chunk_start + i_chunk_sample
                    pbar.update(1)

                    if (
                        self.drift_detector is not None
                        and i_sample % self.drift_detector_update_freq == 0
                    ):
                        self.drift_detector.update()

                    # Check for drifts and update metrics
                    if i_sample >= self.window_size:
                        # Update metrics
                        for name in self.metrics_pred:
                            self.metric_results_pred[name][
                                i_sample
                            ] = self.metrics_pred[name](
                                self.y_true[i_sample - self.window_size : i_sample],
                                self.y_pred[i_sample - self.window_size : i_sample],
                            )
                        for name in self.metrics_score:
                            try:
                                self.metric_results_score[name][
                                    i_sample
                                ] = self.metrics_score[name](
                                    self.y_true[i_sample - self.window_size : i_sample],
                                    self.y_pred_proba[
                                        i_sample - self.window_size : i_sample, 1
                                    ],
                                )
                            except ValueError:
                                pass

                        # If a drift is detected, retrain if wanted
                        if (
                            self.drift_detector is not None
                            and self.drift_detector.drift_detected
                        ):
                            drift_detected_indices.append(i_sample)
                            self.drift_detector.reset()

                            logging.info(f"Drift detected at {i_sample}")

                            if self.retrain_at_drift:
                                logging.info(
                                    f"Retraining with new samples {i_sample} - {i_sample+self.retrain_new_samples}"
                                )
                                # Get new (future) samples for retraining
                                (
                                    X_retrain,
                                    y_retrain,
                                ) = self.dataloader.access_test_drift_samples(
                                    index=i_sample, length=self.retrain_new_samples
                                )

                                # Get old training samples if selected
                                if self.retrain_with_train_set:
                                    (
                                        X_train,
                                        y_train,
                                    ) = self.dataloader.access_base_samples(
                                        dataset="train"
                                    )
                                    X_retrain = np.concatenate([X_train, X_retrain])
                                    y_retrain = np.concatenate([y_train, y_retrain])

                                # Refit model
                                self.model.fit(X_retrain, y_retrain)

                                # Fast-forward index to after the new training samples
                                i_chunk_start = i_sample + self.retrain_new_samples
                                has_retrained = True
                                break  # Break chunk for-loop

                if not has_retrained:
                    i_chunk_start += self.chunksize
