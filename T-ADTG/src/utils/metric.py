from functools import partial
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics as sklearn_metrics


# Metrics computed from raw probability scores (threshold-free)
_PROBABILITY_METRICS = {'auroc', 'auprc'}

_METRIC_FNS = {
    'auroc': sklearn_metrics.roc_auc_score,
    'auprc': sklearn_metrics.average_precision_score,
    'acc': sklearn_metrics.accuracy_score,
    'micro-f1': partial(sklearn_metrics.f1_score, average='micro'),
    'macro-f1': partial(sklearn_metrics.f1_score, average='macro'),
}


class Metric(nn.Module):
    """Computes one or more evaluation metrics from model scores.

    Args:
        metrics: metric name(s) to compute, e.g. ["auroc", "auprc"].
            Supported: "auroc", "auprc", "acc", "micro-f1", "macro-f1".
    """

    def __init__(self, metrics: Union[str, List[str]]):
        super().__init__()
        if isinstance(metrics, str):
            metrics = [metrics]
        for m in metrics:
            if m not in _METRIC_FNS:
                raise ValueError(f'Unsupported metric {m}')
        self.metrics = metrics

    @torch.no_grad()
    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
        """Returns {"preds": ..., "labels": ..., <metric_name>: score, ...}
        for every metric requested at init.
        """
        y_true = y_true.squeeze()
        y_pred = y_pred.squeeze()

        # Probability-style score: used directly by auroc/auprc, and as
        # the basis for the thresholded prediction used by acc/f1 below.
        if y_pred.dim() == 1:
            proba = y_pred.sigmoid()
        else:
            proba = F.softmax(y_pred, dim=1)[:, 1]

        y_true_np = y_true.cpu().numpy()
        proba_np = proba.cpu().numpy()

        results = {"preds": proba_np, "labels": y_true_np}

        for name in self.metrics:
            metric_fn = _METRIC_FNS[name]

            if name in _PROBABILITY_METRICS:
                results[name] = metric_fn(y_true_np.astype('int64'), proba_np)
                continue

            # Discrete/thresholded prediction, only computed if a
            # non-probability metric (acc / f1 variants) was requested
            if y_true.dim() == 1:
                if y_pred.dim() == 1:
                    pred_discrete = (proba > 0.5).long()
                else:
                    pred_discrete = y_pred.argmax(-1)
            else:
                pred_discrete = (y_pred > 0).float()

            results[name] = metric_fn(y_true_np, pred_discrete.cpu().numpy())

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}({self.metrics})'