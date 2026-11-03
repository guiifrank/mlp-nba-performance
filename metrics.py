"""Métricas de classificação implementadas manualmente."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Matriz 2x2 para classes 0 (Bad) e 1 (Good):
    [[TN, FP], [FN, TP]]
    """
    y_true = y_true.ravel().astype(int)
    y_pred = y_pred.ravel().astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    return float(np.mean(y_true == y_pred))


def precision_recall_f1(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
    }


def format_metrics_report(cm: np.ndarray, acc: float) -> str:
    pr = precision_recall_f1(cm)
    lines = [
        f"Acurácia: {acc:.4f}",
        f"Precisão (classe Good): {pr['precision']:.4f}",
        f"Recall (classe Good): {pr['recall']:.4f}",
        f"F1 (classe Good): {pr['f1']:.4f}",
        f"Especificidade (classe Bad): {pr['specificity']:.4f}",
        "",
        "Matriz de confusão [ [TN, FP], [FN, TP] ]:",
        f"  {cm.tolist()}",
    ]
    return "\n".join(lines)
