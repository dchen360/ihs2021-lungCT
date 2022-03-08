"""
Utility file for obtaining metrics for classifiers.

Peter Lais, 09/27/2021
"""

import torch
import warnings
from sklearn.metrics import matthews_corrcoef, roc_auc_score

def summary_statistics(y_true, y_pred):
    """
    Returns the accuracy, MCC, and AUC for a given set of ground-truth
    and prediction data.

    Parameters
    ----------
    y_true: a 1D array containing the ground-truth class labels.
    y_pred: a 1D/2D array.
            * With 1D arrays, each entry should be the probability of a positive
              class (represented by 1) or a negative class (represented by 0).
            * With 2D arrays, the number of columns correspond to the total number of
              classes. Each row's values should sum to one, meaning that each row
              entry represents the estimated probability of the class being that
              column.

    Returns
    -------
    ACC, MCC, and AUC (AUROC) organized into a tuple.
    """

    assert y_true.ndim == 1 and y_pred.ndim <= 2

    # This function cannot be used for multiclass classification.
    # if (y_pred.ndim == 2 and y_pred.shape[-1] > 2):
    #     raise NotImplementedError("summary_statistics does not support multiclass " \
    #      + "classification.")

    # Greedy classification, handles one-dimensional and two-dimensional y_preds
    y_greedy = y_pred.argmax(-1) if y_pred.ndim > 1 else y_pred.round()

    # Calculate the simple accuracy.
    acc = torch.sum(y_greedy == y_true) / len(y_true)

    # Calculate the MCC.
    with warnings.catch_warnings(record=True) as w:
        mcc = matthews_corrcoef(y_true, y_greedy)
        if w: print('Warning raised with MCC calculation. This can likely be ignored.')

    # Calculate the AUC with the predicted probabilities.
    auc = roc_auc_score(y_true, y_pred if y_pred.ndim > 1 else y_pred.max(1)[0], multi_class='ovr')

    return acc, mcc, auc
