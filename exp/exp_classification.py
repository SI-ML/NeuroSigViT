import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from data_provider.data_loader import get_eeg_medformer_dataloaders, get_wearable_dataloaders
from exp.feature_cache import get_features
from models.NeuroSigVIA_ICIMH import NeuroSigVIA_ICIMH
from utils.metrics import compute_metrics_from_predictions
from utils.tools import (set_random_seed, _labels_to_indices, _map_labels,
                         _loader_sample_subject_ids, _aggregate_subject_predictions,
                         _merge_subject_metrics, _resolve_checkpoint_metric,
                         _checkpoint_selection_key, _EarlyStoppingMonitor,
                         _build_patch_lr_scheduler, _balanced_class_weights)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")


def static_loader(features, labels, batch_size, shuffle):
    dataset = TensorDataset(features["graph_tokens"], features["mantis_tokens"],
                            features["patch_mask"], features["valid_fraction"],
                            torch.as_tensor(labels, dtype=torch.long), torch.arange(len(labels)))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def forward_batch(model, batch, device, pretrain=False):
    graph, mantis, mask, fraction, labels, indices = batch
    logits, details = model(graph.to(device), mantis.to(device), mask.to(device),
                            fraction.to(device), pretrain=pretrain)
    return logits, details, labels.to(device), indices


def run_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total, count = 0.0, 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        logits, _, labels, _ = forward_batch(model, batch, device)
        # 参考的加权逐样本 CE 再取均值，不使用按权重和归一化的 reduction=mean。
        loss = criterion(logits, labels).mean()
        loss.backward()
        optimizer.step()
        total += float(loss.detach()) * len(labels)
        count += len(labels)
    return total / count


@torch.no_grad()
def evaluate(model, loader, classes, subject_ids, device):
    model.eval()
    truth, predictions, scores, indices = [], [], [], []
    for batch in loader:
        logits, _, labels, index = forward_batch(model, batch, device)
        probabilities = torch.softmax(logits.float(), dim=-1)
        truth.append(labels.cpu().numpy())
        predictions.append(probabilities.argmax(dim=-1).cpu().numpy())
        scores.append(probabilities.cpu().numpy())
        indices.append(index.numpy())
    details = dict(y_true=np.concatenate(truth), y_pred=np.concatenate(predictions),
                   y_score=np.concatenate(scores), sample_index=np.concatenate(indices))
    metrics = compute_metrics_from_predictions(details["y_true"], details["y_pred"],
                                                details["y_score"], np.arange(len(classes)))
    subject_metrics, subject_details = _aggregate_subject_predictions(
        details["y_true"], details["y_score"], details["sample_index"], subject_ids, classes)
    details.update(subject_details)
    return _merge_subject_metrics(metrics, subject_metrics), details


def pretrain_fusion(model, loader, args, device):
    # 仅训练集参与重建，仅融合层更新；投影目标按参考实现保留梯度。
    parameters = list(model.parameters())
    flags = [p.requires_grad for p in parameters]
    for p in parameters:
        p.requires_grad_(False)
    for p in model.fusion.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.AdamW(model.fusion.parameters(), lr=args.mlp_lr,
                                 weight_decay=args.mlp_weight_decay)
    history = []
    for epoch in range(args.pretrain_epochs):
        model.eval()
        model.fusion.train()
        total, updates, skipped = 0.0, 0, 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            _, details, _, _ = forward_batch(model, batch, device, pretrain=True)
            loss = details["reconstruction_loss"]
            if loss is None:
                skipped += 1
                continue
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            updates += 1
        history.append(dict(epoch=epoch+1, reconstruction_loss=total/updates if updates else 0.0,
                            updates=updates, skipped_batches=skipped))
    for p, flag in zip(parameters, flags):
        p.requires_grad_(flag)
        p.grad = None
    return history


def train(model, loaders, labels, subjects, classes, args, device, result_dir, checkpoint_dir):
    pretraining = []
    if args.icimh_branches == "both" and args.temporal_visual_fusion == "masked_pretrain":
        pretraining = pretrain_fusion(model, loaders["train"], args, device)
        write_json(result_dir / "masked_pretraining_history.json", pretraining)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.mlp_lr, weight_decay=args.mlp_weight_decay)
    scheduler = _build_patch_lr_scheduler(optimizer, args.mlp_lr_scheduler,
        patience=args.mlp_lr_scheduler_patience, factor=args.mlp_lr_scheduler_factor,
        min_lr=args.mlp_lr_scheduler_min_lr)
    weights = (_balanced_class_weights(labels["train"], len(classes), device)
               if args.mlp_class_weight == "balanced" else None)
    criterion = nn.CrossEntropyLoss(weight=weights, reduction="none")
    metric = _resolve_checkpoint_metric(args.patch_checkpoint_metric, subjects["train"],
        subjects["vali"], range(len(labels["train"])), range(len(labels["vali"])))
    monitor = _EarlyStoppingMonitor(strategy=args.mlp_early_stop_strategy,
        patience=args.mlp_early_stop_patience, min_epochs=args.mlp_early_stop_min_epochs,
        warmup_epochs=args.mlp_early_stop_warmup_epochs, ema_decay=args.mlp_early_stop_ema_decay,
        min_delta=args.mlp_early_stop_min_delta)
    best_key, history = None, []
    # 第 1 步：每轮训练和验证；最佳权重只依据验证指标选择。
    for epoch in range(args.mlp_epochs):
        # 学习率热身：前 mlp_lr_warmup_epochs 轮从 lr/W 线性升到 lr，之后交给原调度器（默认 0 不热身）。
        if epoch < args.mlp_lr_warmup_epochs:
            for group in optimizer.param_groups:
                group["lr"] = args.mlp_lr * (epoch + 1) / args.mlp_lr_warmup_epochs
        lr = float(optimizer.param_groups[0]["lr"])
        loss = run_epoch(model, loaders["train"], optimizer, criterion, device)
        validation, _ = evaluate(model, loaders["vali"], classes, subjects["vali"], device)
        key = _checkpoint_selection_key(validation, metric)
        improved = best_key is None or key > best_key
        stop = monitor.update(key, epoch+1)
        stepped = scheduler is not None and epoch+1 > args.mlp_early_stop_warmup_epochs
        if stepped:
            scheduler.step(float(key[0]))
        history.append(dict(epoch=epoch+1, loss=loss, task_loss=loss, validation_metrics=validation,
            learning_rate=lr, next_learning_rate=float(optimizer.param_groups[0]["lr"]),
            lr_scheduler_stepped=stepped, checkpoint_selection_key=list(key),
            checkpoint_improved=improved, early_stop=stop))
        write_json(result_dir / "history.json", history)
        print(f"Epoch {epoch+1}/{args.mlp_epochs} | CE={loss:.8f} | val_macro_f1={validation['macro_f1']:.8f}", flush=True)
        if improved:
            best_key, best_epoch = key, epoch+1
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if stop["should_stop"]:
            break
    # 第 2 步：恢复最佳权重，再验证一次、测试一次。
    model.load_state_dict(best_state)
    torch.save(dict(model=best_state, classes=classes.tolist(), best_epoch=best_epoch), checkpoint_dir / "best.pt")
    final = dict(best_epoch=best_epoch, checkpoint_metric_effective=metric,
                 trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    for split in ("vali", "test"):
        metrics, details = evaluate(model, loaders[split], classes, subjects[split], device)
        final["validation_metrics" if split == "vali" else "test_metrics"] = metrics
        np.savez(result_dir / f"{split}_predictions.npz", **details)
    write_json(result_dir / "metrics.json", final)
    return history, final


def main(args):
    device = torch.device("cuda")
    for dataset in args.dataset_names:
        set_random_seed(args.random_seed)
        load_data = get_eeg_medformer_dataloaders if args.datasets == "eeg" else get_wearable_dataloaders
        bundle = load_data(dataset, args)
        features = get_features(bundle, args, dataset, device)
        labels = {split: getattr(bundle, split + "_labels") for split in features}
        labels["train"], classes, mapping = _labels_to_indices(labels["train"])
        labels.update({split: _map_labels(labels[split], mapping) for split in ("vali", "test")})
        subjects = {split: _loader_sample_subject_ids(getattr(bundle, split + "_loader")) for split in features}
        loaders = {split: static_loader(features[split], labels[split], args.batch_size, split == "train")
                   for split in features}
        # 缓存命中与首次提取均从相同模型初始化和随机打乱状态开始。
        set_random_seed(args.random_seed)
        graph, mantis = features["train"]["graph_tokens"], features["train"]["mantis_tokens"]
        model = NeuroSigVIA_ICIMH(graph.shape[-1], mantis.shape[-1], mantis.shape[2], len(classes),
            branches=args.icimh_branches, fusion=args.temporal_visual_fusion,
            fusion_dim=args.fusion_dim, fusion_heads=args.fusion_heads, dropout=args.mlp_dropout,
            classifier_hidden_dim=args.mlp_hidden_dim, classifier_num_layers=args.mlp_num_layers,
            channel_hidden_dim=args.activity_graph_granularity_hidden_dim, mask_prob=args.mask_prob,
            numeric_reduce=getattr(args, "numeric_only_reduce", "pool")).to(device)
        result = Path(args.result_dir)
        checkpoint = result.parent.parent.parent / "checkpoints" / result.parent.name / result.name
        result.mkdir(parents=True, exist_ok=True)
        checkpoint.mkdir(parents=True, exist_ok=True)
        write_json(result / "args.json", vars(args))
        train(model, loaders, labels, subjects, classes, args, device, result, checkpoint)
