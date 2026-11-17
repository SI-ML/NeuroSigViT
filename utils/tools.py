from __future__ import annotations
import random
import hashlib
import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset
from utils.metrics import compute_metrics_from_predictions


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resize_mantis_input(X):
    X_scaled = F.interpolate(X, size=512, mode="linear", align_corners=False)
    return X_scaled


def _labels_to_indices(labels):
    classes = np.unique(labels)
    class_to_index = {label: index for index, label in enumerate(classes)}
    indices = np.asarray(
        [class_to_index[label] for label in labels], dtype=np.int64
    )
    return indices, classes, class_to_index


def _map_labels(labels, class_to_index):
    unknown = sorted(set(labels) - set(class_to_index))
    if unknown:
        raise ValueError(f"labels contain classes absent from training: {unknown}")
    return np.asarray([class_to_index[label] for label in labels], dtype=np.int64)


def _loader_sample_subject_ids(loader):
    """Return validated per-sample subject IDs, or ``None`` when unavailable."""
    dataset = loader.dataset
    subject_ids = getattr(dataset, "sample_subject_ids", None)
    if subject_ids is None and isinstance(dataset, Subset):
        parent_ids = getattr(dataset.dataset, "sample_subject_ids", None)
        if parent_ids is not None:
            subject_ids = np.asarray(parent_ids)[np.asarray(dataset.indices)]
    if subject_ids is None:
        return None
    subject_ids = np.asarray(subject_ids)
    if subject_ids.ndim != 1 or len(subject_ids) != len(dataset):
        raise ValueError(
            "loader.dataset.sample_subject_ids must be one-dimensional and "
            "match the dataset length"
        )
    if subject_ids.dtype == object:
        subject_ids = subject_ids.astype(str)
    return subject_ids.copy()


def _aggregate_subject_predictions(
    y_true,
    y_score,
    sample_indices,
    sample_subject_ids,
    classes,
):
    """Average window probabilities so every subject contributes exactly once."""
    # 输入窗口 y_score=[M,K]，sample_indices=[M] 索引源 split 受试者表；输出为 [U,K]。
    # U=本 split 的独立受试者数；每人对原始分类窗口概率求均值，之后 argmax，不是多数投票。
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    sample_indices = np.asarray(sample_indices, dtype=np.int64)
    sample_subject_ids = np.asarray(sample_subject_ids)
    if y_true.ndim != 1 or sample_indices.shape != y_true.shape:
        raise ValueError("subject aggregation requires aligned one-dimensional labels")
    if y_score.ndim != 2 or y_score.shape[0] != len(y_true):
        raise ValueError("subject aggregation requires [sample, class] probabilities")
    if y_score.shape[1] != len(classes):
        raise ValueError("subject probability class width does not match classes")
    if not np.isfinite(y_score).all():
        raise ValueError("subject aggregation probabilities must be finite")
    if np.any(sample_indices < 0) or np.any(sample_indices >= len(sample_subject_ids)):
        raise ValueError("subject aggregation sample indices are out of range")
    row_sums = y_score.sum(axis=-1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("subject aggregation rows must be probabilities")

    aligned_subject_ids = sample_subject_ids[sample_indices]
    subject_ids, inverse = np.unique(aligned_subject_ids, return_inverse=True)
    subject_y_true = np.empty(len(subject_ids), dtype=np.int64)
    subject_y_score = np.empty((len(subject_ids), y_score.shape[1]), dtype=np.float64)
    subject_window_count = np.empty(len(subject_ids), dtype=np.int64)
    for subject_index in range(len(subject_ids)):
        member = inverse == subject_index
        labels = np.unique(y_true[member])
        if len(labels) != 1:
            raise ValueError(
                f"subject {subject_ids[subject_index]!r} has inconsistent labels: "
                f"{labels.tolist()}"
            )
        subject_y_true[subject_index] = int(labels[0])
        subject_y_score[subject_index] = y_score[member].mean(axis=0)
        subject_window_count[subject_index] = int(member.sum())
    subject_y_pred = subject_y_score.argmax(axis=-1)
    subject_metrics = compute_metrics_from_predictions(
        subject_y_true,
        subject_y_pred,
        subject_y_score,
        np.arange(len(classes)),
    )
    per_subject_nll = -np.log(
        np.clip(
            subject_y_score[np.arange(len(subject_y_true)), subject_y_true],
            1e-12,
            1.0,
        )
    )
    class_nll = [
        per_subject_nll[subject_y_true == class_index].mean()
        for class_index in np.unique(subject_y_true)
    ]
    subject_metrics["macro_log_loss"] = float(np.mean(class_nll))
    return subject_metrics, {
        "sample_subject_id": aligned_subject_ids,
        "subject_id": subject_ids,
        "subject_y_true": subject_y_true,
        "subject_y_pred": subject_y_pred,
        "subject_y_score": subject_y_score,
        "subject_window_count": subject_window_count,
    }


def _merge_subject_metrics(window_metrics, subject_metrics):
    metrics = dict(window_metrics)
    if subject_metrics is not None:
        metrics.update(
            {
                f"subject_{name}": float(value)
                for name, value in subject_metrics.items()
            }
        )
    return metrics


def _resolve_checkpoint_metric(
    requested,
    train_subject_ids,
    validation_subject_ids,
    train_indices,
    validation_indices,
):
    if requested not in {"auto", "subject_macro_f1", "window_macro_f1"}:
        raise ValueError(
            "checkpoint_metric must be auto, subject_macro_f1, or window_macro_f1"
        )
    subject_ready = (
        train_subject_ids is not None and validation_subject_ids is not None
    )
    overlap = np.asarray([], dtype=np.int64)
    if subject_ready:
        train_used = np.asarray(train_subject_ids)[np.asarray(train_indices)]
        validation_used = np.asarray(validation_subject_ids)[
            np.asarray(validation_indices)
        ]
        overlap = np.intersect1d(train_used, validation_used)
        subject_ready = overlap.size == 0
    if requested == "subject_macro_f1" and not subject_ready:
        reason = (
            "subject IDs are unavailable"
            if train_subject_ids is None or validation_subject_ids is None
            else f"train/validation subjects overlap: {overlap.tolist()}"
        )
        raise ValueError(f"subject checkpoint selection is unsafe because {reason}")
    if requested == "window_macro_f1":
        return "window_macro_f1"
    if subject_ready:
        return "subject_macro_f1"
    print(
        "Checkpoint selection auto fallback: window_macro_f1 "
        "(validated disjoint subject IDs unavailable)"
    )
    return "window_macro_f1"


def _checkpoint_selection_key(metrics, effective_metric):
    if effective_metric == "subject_macro_f1":
        return (
            float(metrics["subject_macro_f1"]),
            -float(metrics["subject_macro_log_loss"]),
        )
    if effective_metric == "window_macro_f1":
        return (float(metrics["macro_f1"]),)
    raise ValueError(f"unsupported checkpoint metric: {effective_metric}")


class _EarlyStoppingMonitor:
    """Track a stopping signal without changing raw checkpoint selection.

    ``raw_selection_key`` exactly preserves the historical lexicographic
    monitor, including the subject log-loss tie-breaker. ``raw_primary``
    applies ``min_delta`` to the unsmoothed primary metric. ``ema_primary`` is
    intended for small validation cohorts: only the primary F1 component is
    smoothed. The caller always saves checkpoints from the raw selection key.
    """

    def __init__(
        self,
        strategy,
        patience,
        min_epochs=0,
        warmup_epochs=0,
        ema_decay=0.6,
        min_delta=0.0,
    ):
        if strategy not in {"raw_selection_key", "raw_primary", "ema_primary"}:
            raise ValueError(f"unsupported early-stop strategy: {strategy}")
        if patience < 0 or min_epochs < 0 or warmup_epochs < 0:
            raise ValueError(
                "early-stop patience, min_epochs, and warmup_epochs must be "
                "non-negative"
            )
        if not math.isfinite(ema_decay) or not 0.0 <= ema_decay < 1.0:
            raise ValueError("early-stop EMA decay must lie in [0, 1)")
        if not math.isfinite(min_delta) or min_delta < 0.0:
            raise ValueError("early-stop min_delta must be finite and non-negative")
        self.strategy = strategy
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.warmup_epochs = int(warmup_epochs)
        self.ema_decay = float(ema_decay)
        self.min_delta = float(min_delta)
        self.ema_score = None
        self.best_score = None
        self.best_raw_key = None
        self.epochs_without_improvement = 0

    def update(self, selection_key, epoch):
        epoch = int(epoch)
        if epoch <= 0:
            raise ValueError("early-stop epoch must be positive")
        raw_key = tuple(float(value) for value in selection_key)
        if not raw_key or not all(math.isfinite(value) for value in raw_key):
            raise ValueError("early-stop selection key must be finite and non-empty")
        raw_score = raw_key[0]

        # Warmup excludes validation observations from the stopping state.
        # Raw checkpoint selection remains independent and may still retain an
        # early epoch for restore_best_weights semantics.
        if epoch <= self.warmup_epochs:
            return {
                "raw_score": float(raw_score),
                "smoothed_score": float(raw_score),
                "best_score": (
                    float(self.best_score) if self.best_score is not None else None
                ),
                "improved": False,
                "epochs_without_improvement": int(
                    self.epochs_without_improvement
                ),
                "in_warmup": True,
                "eligible": False,
                "should_stop": False,
            }

        if self.strategy == "raw_selection_key":
            smoothed_score = raw_score
            improved = self.best_raw_key is None or raw_key > self.best_raw_key
            if improved:
                self.best_raw_key = raw_key
                self.best_score = raw_score
        elif self.strategy == "raw_primary":
            smoothed_score = raw_score
            improved = (
                self.best_score is None
                or raw_score > self.best_score + self.min_delta
            )
            if improved:
                self.best_score = raw_score
        else:
            if self.ema_score is None:
                self.ema_score = raw_score
            else:
                self.ema_score = (
                    self.ema_decay * self.ema_score
                    + (1.0 - self.ema_decay) * raw_score
                )
            smoothed_score = self.ema_score
            improved = (
                self.best_score is None
                or smoothed_score > self.best_score + self.min_delta
            )
            if improved:
                self.best_score = smoothed_score

        if improved:
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        eligible = self.patience > 0 and epoch >= max(1, self.min_epochs)
        should_stop = (
            eligible
            and self.epochs_without_improvement >= self.patience
        )
        return {
            "raw_score": float(raw_score),
            "smoothed_score": float(smoothed_score),
            "best_score": float(self.best_score),
            "improved": bool(improved),
            "epochs_without_improvement": int(
                self.epochs_without_improvement
            ),
            "in_warmup": False,
            "eligible": bool(eligible),
            "should_stop": bool(should_stop),
        }


def _build_patch_lr_scheduler(
    optimizer,
    scheduler_type,
    patience=4,
    factor=0.5,
    min_lr=1.0e-6,
):
    if scheduler_type == "none":
        return None
    if scheduler_type != "reduce_on_plateau":
        raise ValueError(f"unsupported learning-rate scheduler: {scheduler_type}")
    if patience < 0:
        raise ValueError("scheduler patience must be non-negative")
    if not math.isfinite(factor) or not 0.0 < factor < 1.0:
        raise ValueError("scheduler factor must lie in (0, 1)")
    if not math.isfinite(min_lr) or min_lr < 0.0:
        raise ValueError("scheduler min_lr must be finite and non-negative")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=int(patience),
        factor=float(factor),
        min_lr=float(min_lr),
    )


def _balanced_class_weights(label_indices, num_classes, device):
    counts = np.bincount(label_indices, minlength=num_classes)
    if np.any(counts == 0):
        raise ValueError(f"cannot balance empty classes: {counts.tolist()}")
    weights = len(label_indices) / (num_classes * counts)
    return torch.as_tensor(weights, dtype=torch.float32, device=device)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

