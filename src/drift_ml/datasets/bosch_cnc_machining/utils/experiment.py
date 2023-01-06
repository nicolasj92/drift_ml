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
        val_model=True,
        fit_kwargs={},
        name="test_run",
        result_dir="",
        length=None,
        drift_detector=None,
        drift_detector_update_freq=1,
        retrain_at_drift=False,
        retrain_new_parts=10,
        retrain_with_train_set=False,
        window_size=100,
        chunksize=1000,
        random_seed=42,
    ):
        np.random.seed(seed=random_seed)
        self.name = name
        self.result_dir = result_dir

        self.model = model
        self.fit_model = fit_model
        self.val_model = val_model
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
        self.retrain_new_parts = retrain_new_parts
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
            "auroc": roc_auc_score,
            "ap": average_precision_score,
        }

        self.logger = logging.getLogger("driftexperiment")
        self.logger.setLevel(logging.INFO)

    def _fit_model(self, data=None):
        if data is None:
            X_train, y_train = self.dataloader.access_base_samples(dataset="train")
        else:
            X_train, y_train = data
        self.model.fit(X_train, y_train, **self.fit_kwargs)

    def _val_model(self):
        X_val, y_val = self.dataloader.access_base_samples(dataset="val")
        y_pred_val = self.model.predict(X_val)
        y_pred_proba_val = self.model.predict_proba(X_val)

        for name in self.metrics_pred:
            score = self.metrics_pred[name](y_val, y_pred_val)
            print(f"{name}: {score:.2f}")

    def _update_metrics(self, i_sample):
        for name in self.metrics_pred:
            self.metric_results_pred[name][i_sample] = self.metrics_pred[name](
                self.y_true[i_sample - self.window_size : i_sample],
                self.y_pred[i_sample - self.window_size : i_sample],
            )
        for name in self.metrics_score:
            try:
                self.metric_results_score[name][i_sample] = self.metrics_score[name](
                    self.y_true[i_sample - self.window_size : i_sample],
                    self.y_pred_proba[i_sample - self.window_size : i_sample, 1],
                )
            except ValueError:
                pass

    def _retrain(self, i_sample):
        # Get new (future) samples for retraining
        (X_retrain, y_retrain,) = self.dataloader.access_retrain_drift_samples(
            index=i_sample, n_parts=self.retrain_new_parts
        )

        # Get old training samples if selected
        if self.retrain_with_train_set:
            (
                X_train,
                y_train,
            ) = self.dataloader.access_base_samples(dataset="train")
            X_retrain_all = np.concatenate(
                [X_train, X_retrain, *self.X_additional_train]
            )
            y_retrain_all = np.concatenate(
                [y_train, y_retrain, *self.y_additional_train]
            )
        else:
            X_retrain_all = X_retrain
            y_retrain_all = y_retrain

        self.X_additional_train.append(X_retrain)
        self.y_additional_train.append(y_retrain)

        print(X_retrain_all.shape)
        print(y_retrain_all.shape)
        print(y_retrain_all.sum())

        # Refit model
        self._fit_model(data=(X_retrain_all, y_retrain_all))

        return len(X_retrain)

    def run(self):
        # self.dataloader._initialize()
        if self.fit_model:
            # Fit model initially on training data
            self.logger.warn("Doing the initial training of the model.")
            self._fit_model()

        if self.val_model:
            self.logger.warn("Val results:")
            self._val_model()

        self.drift_detector.reset()

        # Create arrays to hold history
        self.y_true = np.full((self.length), np.nan)
        self.y_pred = np.full((self.length), np.nan)
        self.y_pred_proba = np.full((self.length, 2), np.nan)
        self.y_pred_entr = np.full((self.length), np.nan)

        # Additional train data when drifts are detected
        self.X_additional_train = []
        self.y_additional_train = []

        # Create arrays to hold metrics
        self.metric_results_pred = {
            name: np.full((self.length), np.nan) for name in self.metrics_pred
        }
        self.metric_results_score = {
            name: np.full((self.length), np.nan) for name in self.metrics_score
        }

        self.drift_detected_indices = []

        i_chunk_start = 0

        self.logger.warn("Starting experiment queue...")

        # Iterate through data in chunks
        with tqdm(total=self.length) as pbar:
            while i_chunk_start < self.length:
                pbar.n = i_chunk_start
                pbar.refresh()

                chunklength = min(self.chunksize, (self.length - i_chunk_start))

                # Load next chunk of data
                X_chunk, y_true_chunk = self.dataloader.access_test_drift_samples(
                    index=i_chunk_start, length=chunklength
                )

                # Get model output
                y_pred_chunk = self.model.predict(X_chunk)
                y_pred_proba_chunk = self.model.predict_proba(X_chunk)
                y_pred_entr_chunk = entropy(y_pred_proba_chunk, axis=1)

                # Append model output to history
                self.y_true[i_chunk_start : i_chunk_start + chunklength] = y_true_chunk
                self.y_pred[i_chunk_start : i_chunk_start + chunklength] = y_pred_chunk
                self.y_pred_proba[
                    i_chunk_start : i_chunk_start + chunklength
                ] = y_pred_proba_chunk
                self.y_pred_entr[
                    i_chunk_start : i_chunk_start + chunklength
                ] = y_pred_entr_chunk

                has_retrained = False

                # Go through each sample of the chunk and predicitons
                for i_chunk_sample in range(chunklength):
                    i_sample = i_chunk_start + i_chunk_sample
                    pbar.update(1)

                    if (
                        self.drift_detector is not None
                        and i_sample % self.drift_detector_update_freq == 0
                    ):
                        self.drift_detector.update(
                            uncertainty=self.y_pred_entr[i_sample],
                            error=self.y_pred[i_sample] != self.y_true[i_sample],
                            features=X_chunk[i_chunk_sample],
                        )

                    # Check for drifts and update metrics
                    # if window size is filled
                    if i_sample >= self.window_size and (
                        not self.retrain_at_drift
                        or len(self.drift_detected_indices) == 0
                        or (
                            i_sample
                            - (self.drift_detected_indices[-1] + retrain_sample_count)
                        )
                        >= self.window_size
                    ):
                        self._update_metrics(i_sample)

                        # If a drift is detected, retrain if wanted
                        if (
                            self.drift_detector is not None
                            and self.drift_detector.drift_detected
                        ):
                            self.drift_detected_indices.append(i_sample)
                            self.drift_detector.reset()

                            self.logger.warn(f"Drift detected at {i_sample}")

                            if self.retrain_at_drift:
                                self.logger.warn(f"Retraining at index {i_sample}")

                                retrain_sample_count = self._retrain(i_sample)

                                # Fast-forward index to after the new training samples
                                i_chunk_start = i_sample + retrain_sample_count
                                has_retrained = retrain_sample_count
                                break  # Break chunk for-loop

                if not has_retrained:
                    i_chunk_start += chunklength
