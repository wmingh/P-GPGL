import os
import random
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

try:
    from .dataset import load_dataloaders
    from .model import build_model
except ImportError:
    from dataset import load_dataloaders
    from model import build_model


@dataclass
class TrainingConfig:
    aircraft_id: str = "20"
    data_path: str = "data/12_07_2020_Faulty_Daredevil_onboard_log/20_07_12__06_58_20_SD.data"
    segment_start: int = 500
    segment_end: int = 1400

    seed: int = 42
    data_split_seed: Optional[int] = None
    num_random_seeds: int = 10
    device: str = "auto"
    num_workers: int = 0

    batch_size: int = 2
    n_step: int = 20
    sample_size: int = 360

    gpgl_pretrain_epochs: int = 50
    gpgl_batch_size: int = 16
    gpgl_n_hidden: int = 512
    gpgl_k: int = 3
    gpgl_lr: float = 0.001
    gpgl_beta: float = 0.8
    gpgl_bias: float = 2.0
    gpgl_init_std: float = 0.05
    gpgl_force_retrain: bool = False
    gpgl_cache_path: str = "A.npz"
    graph_mode: str = "ptcnet-full"
    graph_top_k: Optional[int] = None
    graph_source_cache_path: Optional[str] = None

    temporal_lr: float = 0.009146470591868487
    temporal_weight_decay: float = 0.00012695603154718826
    temporal_pretrain_epochs: int = 50
    tcn_hidden_dim: int = 32
    tcn_out_dim: int = 64
    aux_hidden_dim: int = 24
    phys_loss_weight: float = 0.0765409100626015
    phys_dt: Optional[float] = None
    use_relobralo: bool = True
    relobralo_alpha: float = 0.999
    relobralo_temperature: float = 0.1
    relobralo_rho: float = 0.999
    label_smoothing: float = 0.1
    num_classes: int = 2
    temporal_early_stopping_patience: int = 15
    temporal_early_stopping_delta: float = 0.0
    temporal_early_stopping_verbose: bool = True
    temporal_save_path: str = "PINN.pth"

    fusion_lr: float = 0.010735487708427595
    fusion_weight_decay: float = 0.0007577731862529718
    fusion_epochs: int = 120
    patch_size: int = 1
    embed_dim: int = 16
    depth: int = 12
    drop_path_rate: float = 0.2
    fusion_hidden_dim: int = 64
    fusion_dropout: float = 0.3068186188131839
    cliff_loss_weight: float = 1.0
    fusion_loss_weight: float = 1.5
    fusion_early_stopping_patience: int = 15
    fusion_early_stopping_delta: float = 0.0
    fusion_early_stopping_verbose: bool = True
    save_path: str = "MODEL.pth"


class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_acc = None
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_acc, val_loss):
        improved = (
            self.best_acc is None
            or val_acc > self.best_acc + self.delta
            or (abs(val_acc - self.best_acc) <= self.delta and val_loss < self.best_loss)
        )
        if improved:
            self.best_acc = val_acc
            self.best_loss = val_loss
            self.counter = 0
            return

        self.counter += 1
        if self.verbose:
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
        if self.counter >= self.patience:
            self.early_stop = True


class ReLoBRaLoBalancer:
    def __init__(self, num_losses=2, alpha=0.999, temperature=0.1, rho_probability=0.999):
        self.num_losses = num_losses
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.rho_probability = float(rho_probability)
        self.initial_losses = None
        self.previous_losses = None
        self.lambdas = None

    @staticmethod
    def sanitize(losses):
        return losses.detach().float().clamp_min(1e-12)

    def current(self, device=None):
        if self.lambdas is None:
            return torch.ones(self.num_losses, device=device)
        if device is None or self.lambdas.device == device:
            return self.lambdas.clone()
        return self.lambdas.to(device).clone()

    @torch.no_grad()
    def update(self, losses):
        if len(losses) != self.num_losses:
            raise ValueError(f"Expected {self.num_losses} losses, got {len(losses)}")

        losses_tensor = self.sanitize(torch.stack(list(losses)))
        device = losses_tensor.device
        if self.lambdas is None:
            self.lambdas = torch.ones(self.num_losses, device=device)
        else:
            self.lambdas = self.lambdas.to(device)

        if self.initial_losses is None:
            self.initial_losses = losses_tensor
            self.previous_losses = losses_tensor
            return self.lambdas.clone()

        prev_losses = self.previous_losses.to(device).clamp_min(1e-12)
        init_losses = self.initial_losses.to(device).clamp_min(1e-12)
        length = float(self.num_losses)

        lambs_hat = torch.softmax(losses_tensor / (prev_losses * self.temperature + 1e-12), dim=0) * length
        lambs0_hat = torch.softmax(losses_tensor / (init_losses * self.temperature + 1e-12), dim=0) * length
        rho = torch.bernoulli(torch.full((1,), self.rho_probability, device=device)).item()
        self.lambdas = (
            rho * self.alpha * self.lambdas
            + (1.0 - rho) * self.alpha * lambs0_hat
            + (1.0 - self.alpha) * lambs_hat
        )
        self.previous_losses = losses_tensor
        return self.lambdas.clone()


def clone_config(cfg):
    return replace(cfg)


def print_training_config(cfg):
    sections = [
        ("Data", ["aircraft_id", "data_path", "segment_start", "segment_end"]),
        ("Runtime", ["seed", "data_split_seed", "num_random_seeds", "device", "num_workers"]),
        ("Dataset", ["batch_size", "n_step", "sample_size"]),
        (
            "GPGL",
            [
                "gpgl_pretrain_epochs",
                "gpgl_batch_size",
                "gpgl_n_hidden",
                "gpgl_k",
                "gpgl_lr",
                "gpgl_beta",
                "gpgl_bias",
                "gpgl_init_std",
                "gpgl_cache_path",
                "gpgl_force_retrain",
            ],
        ),
        (
            "Graph",
            [
                "graph_mode",
                "graph_top_k",
                "graph_source_cache_path",
            ],
        ),
        (
            "Temporal",
            [
                "temporal_lr",
                "temporal_weight_decay",
                "temporal_pretrain_epochs",
                "tcn_hidden_dim",
                "tcn_out_dim",
                "aux_hidden_dim",
                "phys_loss_weight",
                "phys_dt",
                "use_relobralo",
                "relobralo_alpha",
                "relobralo_temperature",
                "relobralo_rho",
                "label_smoothing",
                "num_classes",
                "temporal_early_stopping_patience",
                "temporal_early_stopping_delta",
                "temporal_early_stopping_verbose",
                "temporal_save_path",
            ],
        ),
        (
            "Fusion",
            [
                "fusion_lr",
                "fusion_weight_decay",
                "fusion_epochs",
                "patch_size",
                "embed_dim",
                "depth",
                "drop_path_rate",
                "fusion_hidden_dim",
                "fusion_dropout",
                "cliff_loss_weight",
                "fusion_loss_weight",
                "fusion_early_stopping_patience",
                "fusion_early_stopping_delta",
                "fusion_early_stopping_verbose",
                "save_path",
            ],
        ),
    ]

    print("\n===== TrainingConfig =====")
    for title, keys in sections:
        print(f"[{title}]")
        for key in keys:
            print(f"{key}: {getattr(cfg, key)}")
        print()
    print("==========================\n")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_device(device_name):
    requested = str(device_name).strip().lower()
    cuda_available = torch.cuda.is_available()

    if requested in ("", "auto"):
        resolved = "cuda:0" if cuda_available else "cpu"
    elif requested.startswith("cuda") and not cuda_available:
        print("CUDA requested but not available. Falling back to CPU.")
        resolved = "cpu"
    else:
        resolved = requested

    device = torch.device(resolved)
    print(f"CUDA available: {cuda_available}")
    if device.type == "cuda":
        index = device.index if device.index is not None else 0
        print(f"Using Device: CUDA:{index} ({torch.cuda.get_device_name(index)})")
    else:
        print(f"Using Device: {device.type.upper()}")
    return device


def resolve_runtime_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return str(path)

    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    runtime_dir = Path(main_file).resolve().parent if main_file else Path.cwd().resolve()
    return str(runtime_dir / path)


def compute_binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    acc = (y_true == y_pred).mean()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return float(acc), float(precision), float(recall), float(f1), cm


def move_to_device(inputs, device):
    if isinstance(inputs, (list, tuple)):
        return tuple(move_to_device(item, device) for item in inputs)
    return inputs.to(device)


def compute_temporal_loss(model, seq_inputs, labels, criterion, cfg, loss_balancer=None, update_relobralo=False):
    temp_logits, temporal_outputs = model.forward_temporal_branch(seq_inputs)
    loss_cls = criterion(temp_logits, labels)
    loss_phys, phys_metrics = model.compute_physics_loss(temporal_outputs["h_temp"])

    if cfg.use_relobralo and loss_balancer is not None:
        lambdas = loss_balancer.update([loss_cls, loss_phys]) if update_relobralo else loss_balancer.current(loss_cls.device)
        lambda_phys = (lambdas[1] / lambdas[0].clamp_min(1e-12)).detach()
    else:
        lambda_phys = loss_cls.new_tensor(float(cfg.phys_loss_weight))

    total_loss = loss_cls + lambda_phys * loss_phys
    metrics = {
        "loss_total": total_loss.detach(),
        "loss_cls": loss_cls.detach(),
        "loss_phys": loss_phys.detach(),
        "lambda_phys": lambda_phys.detach(),
        "physics_residual_rms": phys_metrics["physics_residual_rms"].detach(),
    }
    return temp_logits, total_loss, metrics


def compute_fusion_loss(model, seq_inputs, gpgl_inputs, labels, criterion, cfg):
    with torch.no_grad():
        _, temporal_outputs = model.forward_temporal_branch(seq_inputs)
        z_temp = temporal_outputs["z_temp"].detach()

    cliff_logits, clifford_outputs = model.forward_clifford_branch(gpgl_inputs)
    fusion_logits = model.forward_fusion_head(z_temp, clifford_outputs["z_cliff"])
    loss_cliff = criterion(cliff_logits, labels)
    loss_fusion = criterion(fusion_logits, labels)
    total_loss = float(cfg.cliff_loss_weight) * loss_cliff + float(cfg.fusion_loss_weight) * loss_fusion
    metrics = {
        "loss_total": total_loss.detach(),
        "loss_cliff": loss_cliff.detach(),
        "loss_fusion": loss_fusion.detach(),
    }
    return fusion_logits, total_loss, metrics


def save_best_checkpoint(model, acc, avg_loss, best_acc, best_loss, save_path):
    improved = acc > best_acc or (acc == best_acc and avg_loss < best_loss)
    if improved:
        if acc > best_acc:
            print(f"New record! Accuracy improved from {best_acc * 100:.2f}% to {acc * 100:.2f}%")
        else:
            print(f"Accuracy matched best value {acc * 100:.2f}%, but loss improved from {best_loss:.4f} to {avg_loss:.4f}")
        best_acc = acc
        best_loss = avg_loss
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to: {save_path}")
    return best_acc, best_loss


@torch.no_grad()
def evaluate_temporal(model, loader, criterion, cfg, loss_balancer, device, best_acc, best_loss=float("inf"), save_path="best_temporal.pth"):
    model.eval()
    total = 0
    running = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_phys": 0.0,
        "lambda_phys": 0.0,
        "physics_residual_rms": 0.0,
    }
    batch_accs = []
    y_true_all = []
    y_pred_all = []

    for inputs, labels in loader:
        seq_inputs, _ = inputs
        seq_inputs = seq_inputs.to(device)
        labels = labels.to(device)
        logits, _, loss_metrics = compute_temporal_loss(
            model,
            seq_inputs,
            labels,
            criterion,
            cfg,
            loss_balancer=loss_balancer,
            update_relobralo=False,
        )
        batch_size = labels.size(0)
        for key in running:
            running[key] += loss_metrics[key].item() * batch_size

        predicted = logits.argmax(dim=1)
        batch_correct = predicted.eq(labels).sum().item()
        total += batch_size
        batch_accs.append(batch_correct / batch_size)
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_pred_all.extend(predicted.detach().cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Evaluation loader is empty. Please check data split and window size.")

    acc, precision, recall, f1, _ = compute_binary_metrics(y_true_all, y_pred_all)
    avg_metrics = {key: value / total for key, value in running.items()}
    best_acc, best_loss = save_best_checkpoint(model, acc, avg_metrics["loss_total"], best_acc, best_loss, save_path)
    avg_metrics.update(
        {
            "acc": acc,
            "acc_std": float(np.std(batch_accs)) if len(batch_accs) > 1 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return best_acc, best_loss, avg_metrics


def train_temporal_epoch(model, loader, criterion, cfg, loss_balancer, optimizer, device, epoch, total_epochs):
    model.train()
    running = {
        "loss_total": 0.0,
        "loss_cls": 0.0,
        "loss_phys": 0.0,
        "lambda_phys": 0.0,
        "physics_residual_rms": 0.0,
    }
    correct = 0
    total = 0
    batch_accs = []
    y_true_all = []
    y_pred_all = []

    progress = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs}", ncols=100)

    for inputs, labels in progress:
        seq_inputs, _ = inputs
        seq_inputs = seq_inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, total_loss, loss_metrics = compute_temporal_loss(
            model,
            seq_inputs,
            labels,
            criterion,
            cfg,
            loss_balancer=loss_balancer,
            update_relobralo=True,
        )
        total_loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        for key in running:
            running[key] += loss_metrics[key].item() * batch_size

        predicted = logits.argmax(dim=1)
        batch_correct = predicted.eq(labels).sum().item()
        total += batch_size
        correct += batch_correct
        batch_accs.append(batch_correct / batch_size)
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_pred_all.extend(predicted.detach().cpu().numpy().tolist())

        progress.set_postfix(
            loss=f"{running['loss_total'] / total:.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    if total == 0:
        raise ValueError("Training loader is empty. Please check data split and window size.")

    acc, precision, recall, f1, _ = compute_binary_metrics(y_true_all, y_pred_all)
    epoch_metrics = {key: value / total for key, value in running.items()}
    epoch_metrics.update(
        {
            "acc": acc,
            "acc_std": float(np.std(batch_accs)) if len(batch_accs) > 1 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return epoch_metrics


def run_temporal_pretraining(model, train_loader, val_loader, criterion, cfg, device):
    temporal_save_path = resolve_runtime_path(cfg.temporal_save_path)
    loss_balancer = ReLoBRaLoBalancer(
        num_losses=2,
        alpha=cfg.relobralo_alpha,
        temperature=cfg.relobralo_temperature,
        rho_probability=cfg.relobralo_rho,
    ) if cfg.use_relobralo else None

    optimizer = optim.AdamW(
        model.temporal_parameters(),
        lr=cfg.temporal_lr,
        weight_decay=cfg.temporal_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.temporal_pretrain_epochs, 1))
    early_stopping = EarlyStopping(
        patience=cfg.temporal_early_stopping_patience,
        verbose=cfg.temporal_early_stopping_verbose,
        delta=cfg.temporal_early_stopping_delta,
    )

    print(f"Start temporal pretraining for {cfg.temporal_pretrain_epochs} epochs...")
    best_acc = 0.0
    best_loss = float("inf")

    for epoch in range(1, cfg.temporal_pretrain_epochs + 1):
        train_metrics = train_temporal_epoch(
            model,
            train_loader,
            criterion,
            cfg,
            loss_balancer,
            optimizer,
            device,
            epoch,
            cfg.temporal_pretrain_epochs,
        )
        best_acc, best_loss, val_metrics = evaluate_temporal(
            model,
            val_loader,
            criterion,
            cfg,
            loss_balancer,
            device,
            best_acc,
            best_loss,
            save_path=temporal_save_path,
        )
        print(
            f"Temporal Epoch [{epoch}/{cfg.temporal_pretrain_epochs}] "
            f"Train: loss={train_metrics['loss_total']:.4f}, acc={train_metrics['acc'] * 100:.2f}%, "
            f"precision={train_metrics['precision'] * 100:.2f}%, recall={train_metrics['recall'] * 100:.2f}%, "
            f"f1={train_metrics['f1'] * 100:.2f}% | "
            f"Val: loss={val_metrics['loss_total']:.4f}, acc={val_metrics['acc'] * 100:.2f}%, "
            f"precision={val_metrics['precision'] * 100:.2f}%, recall={val_metrics['recall'] * 100:.2f}%, "
            f"f1={val_metrics['f1'] * 100:.2f}%"
        )
        early_stopping(val_metrics["acc"], val_metrics["loss_total"])
        if early_stopping.early_stop:
            print(f"Temporal pretraining early stopping triggered at epoch {epoch}.")
            break
        scheduler.step()

    if os.path.exists(temporal_save_path):
        model.load_state_dict(torch.load(temporal_save_path, map_location=device))


@torch.no_grad()
def evaluate_fusion(model, loader, criterion, cfg, device, best_acc, best_loss=float("inf"), save_path="best_model.pth"):
    model.eval()
    total = 0
    running = {
        "loss_total": 0.0,
        "loss_cliff": 0.0,
        "loss_fusion": 0.0,
    }
    batch_accs = []
    y_true_all = []
    y_pred_all = []

    for inputs, labels in loader:
        inputs = move_to_device(inputs, device)
        seq_inputs, gpgl_inputs = inputs
        labels = labels.to(device)
        logits, _, loss_metrics = compute_fusion_loss(model, seq_inputs, gpgl_inputs, labels, criterion, cfg)
        batch_size = labels.size(0)
        for key in running:
            running[key] += loss_metrics[key].item() * batch_size

        predicted = logits.argmax(dim=1)
        batch_correct = predicted.eq(labels).sum().item()
        total += batch_size
        batch_accs.append(batch_correct / batch_size)
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_pred_all.extend(predicted.detach().cpu().numpy().tolist())

    if total == 0:
        raise ValueError("Evaluation loader is empty. Please check data split and window size.")

    acc, precision, recall, f1, _ = compute_binary_metrics(y_true_all, y_pred_all)
    avg_metrics = {key: value / total for key, value in running.items()}
    best_acc, best_loss = save_best_checkpoint(model, acc, avg_metrics["loss_total"], best_acc, best_loss, save_path)
    avg_metrics.update(
        {
            "acc": acc,
            "acc_std": float(np.std(batch_accs)) if len(batch_accs) > 1 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return best_acc, best_loss, avg_metrics


def train_fusion_epoch(model, loader, criterion, cfg, optimizer, device, epoch, total_epochs):
    model.train()
    model.temporal_encoder.eval()
    model.temporal_classifier.eval()
    model.physics_regularizer.eval()

    running = {
        "loss_total": 0.0,
        "loss_cliff": 0.0,
        "loss_fusion": 0.0,
    }
    correct = 0
    total = 0
    batch_accs = []
    y_true_all = []
    y_pred_all = []

    progress = tqdm(loader, desc=f"Fusion Epoch {epoch}/{total_epochs}", ncols=100)

    for inputs, labels in progress:
        inputs = move_to_device(inputs, device)
        seq_inputs, gpgl_inputs = inputs
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, total_loss, loss_metrics = compute_fusion_loss(model, seq_inputs, gpgl_inputs, labels, criterion, cfg)
        total_loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        for key in running:
            running[key] += loss_metrics[key].item() * batch_size

        predicted = logits.argmax(dim=1)
        batch_correct = predicted.eq(labels).sum().item()
        total += batch_size
        correct += batch_correct
        batch_accs.append(batch_correct / batch_size)
        y_true_all.extend(labels.detach().cpu().numpy().tolist())
        y_pred_all.extend(predicted.detach().cpu().numpy().tolist())

        progress.set_postfix(
            loss=f"{running['loss_total'] / total:.4f}",
            acc=f"{100.0 * correct / total:.2f}%",
        )

    if total == 0:
        raise ValueError("Training loader is empty. Please check data split and window size.")

    acc, precision, recall, f1, _ = compute_binary_metrics(y_true_all, y_pred_all)
    epoch_metrics = {key: value / total for key, value in running.items()}
    epoch_metrics.update(
        {
            "acc": acc,
            "acc_std": float(np.std(batch_accs)) if len(batch_accs) > 1 else 0.0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    return epoch_metrics


@torch.no_grad()
def evaluate_test_set(model, loader, device):
    model.eval()
    y_true = []
    y_pred = []

    for inputs, labels in loader:
        inputs = move_to_device(inputs, device)
        logits = model(*inputs)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(labels.numpy().tolist())

    acc, precision, recall, f1, cm = compute_binary_metrics(y_true, y_pred)
    print("\n===== Final Test Metrics (Positive class = fault=1) =====")
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall   : {recall * 100:.2f}%")
    print(f"F1-Score : {f1 * 100:.2f}%")
    print("Final Confusion Matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "cm": cm,
    }


def run_training(cfg=None):
    cfg = TrainingConfig() if cfg is None else cfg

    seed_everything(cfg.seed)
    device = get_device(cfg.device)
    fusion_save_path = resolve_runtime_path(cfg.save_path)

    train_loader, val_loader, test_loader = load_dataloaders(cfg)
    sample_inputs, _ = next(iter(train_loader))
    seq_inputs, gpgl_inputs = sample_inputs
    seq_input_dim = int(seq_inputs.shape[-1])
    gpgl_input_channels = int(gpgl_inputs.shape[1])

    model = build_model(cfg, seq_input_dim, gpgl_input_channels).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    start_time = time.time()
    run_temporal_pretraining(model, train_loader, val_loader, criterion, cfg, device)

    model.freeze_temporal_branch()
    optimizer = optim.AdamW(
        list(model.clifford_parameters()) + list(model.fusion_parameters()),
        lr=cfg.fusion_lr,
        weight_decay=cfg.fusion_weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.fusion_epochs, 1))
    early_stopping = EarlyStopping(
        patience=cfg.fusion_early_stopping_patience,
        verbose=cfg.fusion_early_stopping_verbose,
        delta=cfg.fusion_early_stopping_delta,
    )

    print(f"Start fusion training for {cfg.fusion_epochs} epochs...")
    best_val_acc = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, cfg.fusion_epochs + 1):
        train_metrics = train_fusion_epoch(
            model,
            train_loader,
            criterion,
            cfg,
            optimizer,
            device,
            epoch,
            cfg.fusion_epochs,
        )
        best_val_acc, best_val_loss, val_metrics = evaluate_fusion(
            model,
            val_loader,
            criterion,
            cfg,
            device,
            best_val_acc,
            best_val_loss,
            save_path=fusion_save_path,
        )
        print(
            f"Fusion Epoch [{epoch}/{cfg.fusion_epochs}] "
            f"Train: loss={train_metrics['loss_total']:.4f}, acc={train_metrics['acc'] * 100:.2f}%, "
            f"precision={train_metrics['precision'] * 100:.2f}%, recall={train_metrics['recall'] * 100:.2f}%, "
            f"f1={train_metrics['f1'] * 100:.2f}% | "
            f"Val: loss={val_metrics['loss_total']:.4f}, acc={val_metrics['acc'] * 100:.2f}%, "
            f"precision={val_metrics['precision'] * 100:.2f}%, recall={val_metrics['recall'] * 100:.2f}%, "
            f"f1={val_metrics['f1'] * 100:.2f}%"
        )
        early_stopping(val_metrics["acc"], val_metrics["loss_total"])
        if early_stopping.early_stop:
            print(f"Fusion training early stopping triggered at epoch {epoch}.")
            break
        scheduler.step()

    if fusion_save_path and os.path.exists(fusion_save_path):
        model.load_state_dict(torch.load(fusion_save_path, map_location=device))

    test_metrics = evaluate_test_set(model, test_loader, device)
    total_minutes = (time.time() - start_time) / 60.0
    print(f"Training Finished. Total time: {total_minutes:.2f} mins")

    return {
        "best_acc": float(best_val_acc),
        "best_val_acc": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "test_acc": test_metrics["acc"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "time_minutes": float(total_minutes),
    }
