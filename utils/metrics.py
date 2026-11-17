from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _macro_curve_metric(y_true, y_score, classes, metric_fn):
    values = []

    for class_idx, class_label in enumerate(classes):
        y_binary = (y_true == class_label).astype(int)
        if y_binary.min() == y_binary.max():
            continue
        values.append(metric_fn(y_binary, y_score[:, class_idx]))

    if not values:
        return float("nan")

    return sum(values) / len(values)


def compute_metrics_from_predictions(y_true, y_pred, y_score, classes):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_recall": recall_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_f1": f1_score(
            y_true, y_pred, labels=classes, average="macro", zero_division=0
        ),
        "macro_auroc": _macro_curve_metric(
            y_true, y_score, classes, roc_auc_score
        ),
        "macro_auprc": _macro_curve_metric(
            y_true, y_score, classes, average_precision_score
        ),
    }
