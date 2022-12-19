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
import numpy as np
import logging

from drift_ml.datasets.bosch_cnc_machining.models.lenet import LeNet

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


class NNClassifier:
    def __init__(self, model=LeNet):
        self.model = model()

    def predict_proba(self, X, temperature):
        return self.model.predict_proba(X, temperature=temperature)

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
        verbose=True,
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

        val_X = tensor(val_X)
        val_y = tensor(val_y)

        auroc = BinaryAUROC()
        auprc = BinaryAveragePrecision()
        f1 = BinaryF1Score(threshold=0.5)

        for epoch in range(epochs):
            self.model.train()

            for idx, (X, y) in enumerate(train_loader):
                sgd.zero_grad()
                predict_y = self.model(X.float())
                loss = loss_fn(predict_y, y.float())

                # if idx % 10 == 0:
                #     logging.debug(f"Idx: {idx}, loss: {loss.sum().item():.4f}")

                loss.backward()
                sgd.step()

            self.model.eval()
            predict_probs = self.model.predict_proba(val_X.float()).detach()

            auroc_score = auroc(predict_probs, val_y.float())
            auprc_score = auprc(predict_probs, val_y.float())
            f1_score = f1(predict_probs, val_y.float())
            if verbose:
                logging.debug(
                    f"Epoch {epoch}, Val. AUROC {auroc_score:.2f}, AURPC {auprc_score:.2f}, F1 {f1_score:.2f}"
                )

        logging.debug(
            f"Final val. performance: AUROC {auroc_score:.2f}, AURPC {auprc_score:.2f}, F1 {f1_score:.2f}"
        )


class NNEnsembleClassifier:
    def __init__(self, base_model=NNClassifier, n_ensemble=10, base_model_params={}):
        self.models = [base_model(**base_model_params) for _ in n_ensemble]

    def fit(self):
        pass

    def predict_proba(self):
        pass
