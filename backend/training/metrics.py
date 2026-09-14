"""Metrics (plan §11): which scorers a model can produce, and the per-fold summary."""

import numpy as np
from sklearn.metrics import (average_precision_score, cohen_kappa_score, confusion_matrix, f1_score,
                             get_scorer, make_scorer, matthews_corrcoef, precision_score, recall_score)

METRICS = {  # key: (label, higher is better)
    "accuracy": ("Accuracy", True), "balanced_accuracy": ("Balanced accuracy", True),
    "precision": ("Precision", True), "recall": ("Recall", True), "specificity": ("Specificity", True),
    "f1": ("F1", True), "mcc": ("MCC", True), "kappa": ("Cohen's kappa", True),
    "roc_auc": ("ROC-AUC", True), "pr_auc": ("PR-AUC", True), "log_loss": ("Log loss", False),
    "mae": ("MAE", False), "mse": ("MSE", False), "rmse": ("RMSE", False),
    "r2": ("R²", True), "adj_r2": ("Adjusted R²", True), "mape": ("MAPE", False),
}
TASK_METRICS = {
    "classification": ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1",
                       "mcc", "kappa", "roc_auc", "pr_auc", "log_loss"],
    "regression": ["mae", "mse", "rmse", "r2", "adj_r2", "mape"],
}
_NEGATED = {"mae", "mse", "rmse", "mape", "log_loss"}  # scikit-learn scorers return these negated


def _specificity(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - cm.sum(axis=1) - fp
    return float(np.mean(tn / np.maximum(tn + fp, 1)))


def scorers(task, n_classes, pos_label, has_proba, target_has_zero):
    """metric -> scikit-learn scorer, only for metrics this model and data can
    produce (the rest show as n/a)."""
    if task == "regression":
        s = {"mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error",
             "rmse": "neg_root_mean_squared_error", "r2": "r2"}
        if not target_has_zero:  # MAPE divides by the true value
            s["mape"] = "neg_mean_absolute_percentage_error"
        return s
    s = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
         "mcc": make_scorer(matthews_corrcoef), "kappa": make_scorer(cohen_kappa_score)}
    if n_classes == 2:
        s.update(
            precision=make_scorer(precision_score, pos_label=pos_label, zero_division=0),
            recall=make_scorer(recall_score, pos_label=pos_label, zero_division=0),
            specificity=make_scorer(recall_score, pos_label=1 - pos_label, zero_division=0),
            f1=make_scorer(f1_score, pos_label=pos_label, zero_division=0),
            roc_auc="roc_auc",  # the same for either class
            pr_auc=make_scorer(average_precision_score, response_method=("decision_function", "predict_proba"),
                               pos_label=pos_label))
    else:
        labels = list(range(n_classes))
        s.update(
            precision=make_scorer(precision_score, average="macro", zero_division=0),
            recall=make_scorer(recall_score, average="macro", zero_division=0),
            specificity=make_scorer(_specificity, labels=labels),
            f1=make_scorer(f1_score, average="macro", zero_division=0))
        if has_proba:  # multiclass AUC scorers need predict_proba (verified 2026-09-14)
            s.update(roc_auc="roc_auc_ovr", pr_auc="average_precision")
    if has_proba:
        s["log_loss"] = "neg_log_loss"
    return s


def score(name, scorer, est, X, y):
    v = (get_scorer(scorer) if isinstance(scorer, str) else scorer)(est, X, y)
    v = -v if name in _NEGATED else v
    return float(v) if v is not None and np.isfinite(v) else None


def summary(values):
    vals = [v for v in values if v is not None]
    return {"mean": float(np.mean(vals)) if vals else None,
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,  # sample sd across folds
            "folds": values}


def adjusted_r2(r2, n, p):
    return None if r2 is None or n - p - 1 <= 0 else float(1 - (1 - r2) * (n - 1) / (n - p - 1))


