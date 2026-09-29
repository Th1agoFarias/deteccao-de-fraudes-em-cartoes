import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def evaluate_model(model, X_train, y_train, X_val, y_val):

    model_fit = clone(model)
    model_fit.fit(X_train, y_train)

    y_pred = model_fit.predict(X_val)
    y_proba = model_fit.predict_proba(X_val)[:, 1]

    results = {
        "accuracy": accuracy_score(y_val, y_pred),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred),
        "average_precision": average_precision_score(y_val, y_proba),
        "roc_auc": roc_auc_score(y_val, y_proba),
    }

    return pd.DataFrame([results])


def cross_validate_model(model, X_train, y_train, random_state=42, n_splits=5):

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    fold_scores = []

    for train_idx, val_idx in skf.split(X_train, y_train):

        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]

        y_fold_train = y_train.iloc[train_idx]
        y_fold_val = y_train.iloc[val_idx]

        fold_model = clone(model)
        fold_model.fit(X_fold_train, y_fold_train)

        y_proba = fold_model.predict_proba(X_fold_val)[:, 1]

        ap = average_precision_score(y_fold_val, y_proba)

        fold_scores.append(ap)

    return {
        "fold_scores": fold_scores,
        "mean": np.mean(fold_scores),
        "std": np.std(fold_scores),
    }


def evaluate_thresholds(y_true, y_proba, thresholds=None):

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    for threshold in thresholds:

        y_pred = (y_proba >= threshold).astype(int)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)

        results.append(
            {
                "Threshold": threshold,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

    return pd.DataFrame(results)


def plot_confusion_matrix(y_true, y_pred, classes, cmap=plt.cm.Blues):

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")

    return ax


def plot_precision_recall_curve(y_true, y_proba, ax=None, label=None):

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    curve_label = label or "Modelo"

    ax.plot(
        recall,
        precision,
        label=f"{curve_label} | AP = {ap:.4f}",
    )

    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid()

    return ax
