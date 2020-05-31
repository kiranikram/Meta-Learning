"""## Accuracy Calculated on NEW task after Meta Training"""

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from typing import Callable, List, Union

#from metrics import NAMED_METRICS


def evaluate(model: Module, dataloader: DataLoader, prepare_batch: Callable, metrics: List[Union[str, Callable]],
             loss_fn: Callable = None, prefix: str = 'val_', suffix: str = ''):
    logs = {}
    seen = 0
    totals = {m: 0 for m in metrics}
    if loss_fn is not None:
        totals['loss'] = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = prepare_batch(batch)
            y_pred = model(x)

            seen += x.shape[0]

            if loss_fn is not None:
                totals['loss'] += loss_fn(y_pred, y).item() * x.shape[0]

            for m in metrics:
                if isinstance(m, str):
                    v = NAMED_METRICS[m](y, y_pred)
                else:
                    # Assume metric is a callable function
                    v = m(y, y_pred)

                totals[m] += v * x.shape[0]

    for m in ['loss'] + metrics:
        logs[prefix + m + suffix] = totals[m] / seen

    return logs
