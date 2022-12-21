import pandas as pd
from torch import tensor, Tensor
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryCalibrationError,
    BinaryF1Score,
    BinaryMatthewsCorrCoef,
    BinaryPrecision,
    BinaryRecall,
)


class Metrics:
    def __init__(
        self,
        metrics=[
            BinaryAUROC,
            BinaryAveragePrecision,
            BinaryF1Score,
            BinaryMatthewsCorrCoef,
        ],
    ):
        self.metrics = [metric() for metric in metrics]

    def __call__(self, y_pred_scores, y_true):
        if not isinstance(y_pred_scores, Tensor):
            y_pred_scores = tensor(y_pred_scores)
        if not isinstance(y_true, Tensor):
            y_true = tensor(y_true)

        return {
            metric.__class__.__name__: metric(y_pred_scores, y_true).numpy().item()
            for metric in self.metrics
        }

    def print(self, y_pred_scores, y_true, name=0):
        metrics = self(y_pred_scores, y_true)
        df = pd.DataFrame(metrics, index=[name])
        print(df)

