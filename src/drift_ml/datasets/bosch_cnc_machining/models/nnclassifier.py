import torch
from torch.nn import BCEWithLogitsLoss
from torch import tensor
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryF1Score,
    BinaryAveragePrecision,
)

import functools
import numpy as np

from drift_ml.utils.utils import in_notebook
from drift_ml.datasets.bosch_cnc_machining.models.lenet import LeNet

if in_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

import logging

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NNClassifier:
    def __init__(self, model=LeNet, device="cuda"):
        self.model = model()
        self.device = device
        self.model.to(self.device)

    def predict(self, X, threshold=0.5, temperature=1.0, return_scores=False):
        y_probs = self.predict_proba(X, temperature=temperature)
        y = np.zeros_like(y_probs)
        y[y_probs > threshold] = 1
        if return_scores:
            return y, y_probs
        else:
            return y

    def predict_proba(self, X, temperature=1.0):
        return (
            self.model.predict_proba(
                tensor(X).to(self.device).float(),
                temperature=tensor(temperature).to(self.device),
            )
            .detach()
            .cpu()
            .numpy()
        )

    def fit(
        self,
        train_X,
        train_y,
        val_X,
        val_y,
        batch_size=64,
        lrate=1e-3,
        epochs=100,
        class_weighted_sampling=True,
        verbose=False,
        show_final_val_performance=True,
        return_self=True,
    ):
        if verbose:
            logging.debug(
                f"Starting training with batch size {batch_size}, lrate {lrate}, epochs {epochs}"
            )
        sgd = SGD(self.model.parameters(), lr=lrate)
        loss_fn = BCEWithLogitsLoss()

        if class_weighted_sampling:
            class_ratio = train_y.sum() / len(train_y)
            train_sample_weights = np.zeros_like(train_y)
            train_sample_weights[train_y == 0] = class_ratio
            train_sample_weights[train_y == 1] = 1 - class_ratio
            train_sample_weights = np.squeeze(train_sample_weights)
            train_sampler = WeightedRandomSampler(
                weights=train_sample_weights, num_samples=len(train_X), replacement=True
            )
        else:
            train_sampler = None

        train_set = TensorDataset(tensor(train_X), tensor(train_y))
        train_loader = DataLoader(
            train_set, sampler=train_sampler, batch_size=batch_size
        )

        val_X = tensor(val_X).to(self.device)
        val_y = tensor(val_y)

        auroc = BinaryAUROC()
        auprc = BinaryAveragePrecision()
        f1 = BinaryF1Score(threshold=0.5)

        auroc_score = np.nan
        auprc_score = np.nan
        f1_score = np.nan

        if verbose:
            pbar = tqdm(total=epochs, desc="Training NN")
        for epoch in range(epochs):
            self.model.train()

            for idx, (X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                sgd.zero_grad()
                predict_y = self.model(X.float())

                loss = loss_fn(predict_y, y.float())

                if verbose and idx % 10 == 0:
                    pbar.set_description(
                        f"Epoch {epoch}, Loss {loss:.5f} Val. AUROC {auroc_score:.2f}, AURPC {auprc_score:.2f}, F1 {f1_score:.2f}"
                    )

                loss.backward()
                sgd.step()

            self.model.eval()
            predict_probs = tensor(self.predict_proba(val_X.float()))
            auroc_score = auroc(predict_probs, val_y.float())
            auprc_score = auprc(predict_probs, val_y.float())
            f1_score = f1(predict_probs, val_y.float())

            if verbose:
                pbar.update(1)

        if verbose or show_final_val_performance:
            logging.debug(
                f"Final val. performance: AUROC {auroc_score:.2f}, AURPC {auprc_score:.2f}, F1 {f1_score:.2f}"
            )

        if return_self:
            return self


def call_fit_helper(model, args, kwargs):
    return model.fit(*args, **kwargs)


class NNEnsembleClassifier:
    def __init__(
        self, base_model=NNClassifier, n_ensemble=5, workers=2, base_model_params={}
    ):
        self.workers = workers
        self.models = [base_model(**base_model_params) for _ in range(n_ensemble)]

    def fit(self, fit_args, fit_kwargs):
        mp = torch.multiprocessing.get_context("spawn")
        pool = mp.Pool(self.workers)

        partial_fit = functools.partial(
            call_fit_helper, args=fit_args, kwargs=fit_kwargs
        )
        self.models = pool.map(partial_fit, self.models)

    def predict_proba(self, X, temperature=1.0):
        predictions = np.array(
            [model.predict_proba(X, temperature=temperature) for model in self.models]
        )
        return np.mean(predictions, axis=0)

    def predict(self, X, threshold=0.5, temperature=1.0, return_scores=False):
        y_probs = self.predict_proba(X, temperature=temperature)
        y = np.zeros_like(y_probs)
        y[y_probs > threshold] = 1
        if return_scores:
            return y, y_probs
        else:
            return y
