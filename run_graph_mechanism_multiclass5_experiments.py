import argparse
import copy
import importlib.util
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from .dataset import GPGLGraphLearner, build_uav_prior, compute_pearson_adjacency, compute_random_adjacency
    from .dataset import prune_adjacency_top_k, symmetrize_adjacency, to_gnn_windows, windows_to_gpgl_input
    from .trainer import TrainingConfig, get_device, seed_everything
    from .run_graph_mechanism_experiments import (
        DEFAULT_SEEDS,
        EDGE_KNOCKOUT_K,
        EXP1_K,
        EXP2_K,
        GRAPH_MODE_LABELS,
        GRAPH_MODE_ORDER,
        HEAD_LABELS,
        HEAD_ORDER,
        PRIOR_VARIANT_LABELS,
        PRIOR_VARIANT_ORDER,
        CliffordAuxClassifier,
        GraphAuxClassifier,
        active_edge_count,
        edge_set_from_adjacency,
        ensure_parent,
        format_pct,
        jaccard_edge_overlap,
        markdown_table,
        mean_std,
        prior_edge_mask,
        random_active_edges,
        random_prior_like,
        remove_edges,
        shuffled_prior,
        strongest_edges,
    )
except ImportError:
    from dataset import GPGLGraphLearner, build_uav_prior, compute_pearson_adjacency, compute_random_adjacency
    from dataset import prune_adjacency_top_k, symmetrize_adjacency, to_gnn_windows, windows_to_gpgl_input
    from trainer import TrainingConfig, get_device, seed_everything
    from run_graph_mechanism_experiments import (
        DEFAULT_SEEDS,
        EDGE_KNOCKOUT_K,
        EXP1_K,
        EXP2_K,
        GRAPH_MODE_LABELS,
        GRAPH_MODE_ORDER,
        HEAD_LABELS,
        HEAD_ORDER,
        PRIOR_VARIANT_LABELS,
        PRIOR_VARIANT_ORDER,
        CliffordAuxClassifier,
        GraphAuxClassifier,
        active_edge_count,
        edge_set_from_adjacency,
        ensure_parent,
        format_pct,
        jaccard_edge_overlap,
        markdown_table,
        mean_std,
        prior_edge_mask,
        random_active_edges,
        random_prior_like,
        remove_edges,
        shuffled_prior,
        strongest_edges,
    )


PROJECT_ROOT = Path(__file__).resolve().parent
MULTICLASS_SUPPORT_PATH = PROJECT_ROOT / "1" / "dataset.py"


def load_support_module(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load support module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MC_DATASET = load_support_module("mechanism_multiclass_dataset_support", MULTICLASS_SUPPORT_PATH)
DATA = MC_DATASET.DATA
FEATURE_COLUMNS = MC_DATASET.FEATURE_COLUMNS
MULTICLASS_LABELS = {
    0: "normal",
    1: "add1-only",
    2: "add2-only",
    3: "motor-only",
    4: "compound",
}


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run 5-class graph-mechanism experiments and export a single markdown report."
    )
    parser.add_argument(
        "--results-root",
        default="results/graph_mechanism_multiclass5",
        help="Output directory root.",
    )
    parser.add_argument("--sample-size", type=int, default=360, help="Training sample size from the pool.")
    parser.add_argument("--data-split-seed", type=int, default=42, help="Fixed split seed.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Training seeds.")
    parser.add_argument("--device", default=None, help="Device override, e.g. cuda:0 or cpu.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers.")
    parser.add_argument("--head-batch-size", type=int, default=16, help="Batch size for graph-only heads.")
    parser.add_argument("--head-epochs", type=int, default=40, help="Max training epochs for graph-only heads.")
    parser.add_argument("--head-patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument("--head-lr", type=float, default=None, help="Learning rate for graph-only heads.")
    parser.add_argument("--head-weight-decay", type=float, default=None, help="Weight decay for graph-only heads.")
    parser.add_argument(
        "--gpgl-pretrain-epochs",
        type=int,
        default=None,
        help="Override GPGL pretrain epochs for learned graphs.",
    )
    parser.add_argument(
        "--data-path",
        default=str(MC_DATASET.DEFAULT_DATA_PATH),
        help="Path to the multiclass data file.",
    )
    parser.add_argument("--aircraft-id", default=str(MC_DATASET.DEFAULT_AIRCRAFT_ID), help="Aircraft id.")
    parser.add_argument("--segment-start", type=int, default=int(MC_DATASET.DEFAULT_TIME_SLICE[0]), help="Slice start.")
    parser.add_argument("--segment-end", type=int, default=int(MC_DATASET.DEFAULT_TIME_SLICE[1]), help="Slice end.")
    parser.add_argument(
        "--mode-filter",
        type=float,
        default=float(MC_DATASET.DEFAULT_MODE_FILTER),
        help="Only keep rows with this mode value.",
    )
    parser.add_argument("--smoke", action="store_true", help="Run a tiny version for verification.")
    return parser


def build_experiment_root(args):
    return Path(args.results_root) / f"sample_{int(args.sample_size)}" / f"split_{int(args.data_split_seed)}"


def build_base_config(args):
    cfg = TrainingConfig()
    cfg.sample_size = int(args.sample_size)
    cfg.data_split_seed = int(args.data_split_seed)
    cfg.num_workers = int(args.num_workers)
    cfg.gpgl_k = 0
    cfg.gpgl_force_retrain = True
    cfg.data_path = str(args.data_path)
    cfg.aircraft_id = str(args.aircraft_id)
    cfg.segment_start = int(args.segment_start)
    cfg.segment_end = int(args.segment_end)

    cfg.label_smoothing = 0.01
    cfg.num_classes = 5
    cfg.tcn_out_dim = 48
    cfg.embed_dim = 4
    cfg.depth = 6
    cfg.fusion_hidden_dim = 16
    cfg.fusion_dropout = 0.05033906512729099
    cfg.fusion_lr = 0.0024811290920228805
    cfg.fusion_weight_decay = 3.8097150619702276e-05

    if args.device is not None:
        cfg.device = args.device
    if args.gpgl_pretrain_epochs is not None:
        cfg.gpgl_pretrain_epochs = int(args.gpgl_pretrain_epochs)
    if args.smoke:
        cfg.sample_size = min(int(args.sample_size), 90)
        cfg.gpgl_pretrain_epochs = min(int(cfg.gpgl_pretrain_epochs), 2)
    return cfg


def assign_multiclass5_fault_labels(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.assign(fault=0)

    add1 = (frame["add1"] > 0.005) | (frame["add1"] < -0.005)
    add2 = (frame["add2"] > 0.005) | (frame["add2"] < -0.005)
    motor = (frame["m1"] < 1.0) | (frame["m2"] < 1.0)

    compound = ((add1.astype(int) + add2.astype(int) + motor.astype(int)) >= 2)
    frame.loc[add1 & ~add2 & ~motor, "fault"] = 1
    frame.loc[add2 & ~add1 & ~motor, "fault"] = 2
    frame.loc[motor & ~add1 & ~add2, "fault"] = 3
    frame.loc[compound, "fault"] = 4
    return frame


def compute_multiclass_metrics(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(int(num_classes))),
        average="macro",
        zero_division=0,
    )
    acc = float((y_true == y_pred).mean())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(int(num_classes))))
    return {
        "acc": acc,
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "cm": cm,
    }


def make_loader(features, labels, batch_size, shuffle, num_workers):
    dataset = TensorDataset(torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def prepare_data_bundle(cfg, args):
    data_path = MC_DATASET.resolve_data_path(str(cfg.data_path), cfg.aircraft_id)
    raw_data = DATA(str(data_path), cfg.aircraft_id, data_type="flight", sample_period=0.1)
    frame = raw_data.get_labelled_data()
    frame = frame.loc[int(cfg.segment_start):int(cfg.segment_end)].copy()
    frame = assign_multiclass5_fault_labels(frame)
    frame = frame[frame["mode"] == float(args.mode_filter)].copy()

    X_raw = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_raw = frame["fault"].to_numpy(dtype=np.int64)
    X_all, y_all = MC_DATASET.build_time_windows(X_raw, y_raw, n_step=int(cfg.n_step))

    split_seed = int(cfg.data_split_seed if cfg.data_split_seed is not None else cfg.seed)
    stratify_labels = MC_DATASET.get_stratify_labels(y_all)
    X_gpgl_raw, _, y_gpgl, _ = train_test_split(
        X_all,
        y_all,
        train_size=0.8,
        random_state=split_seed,
        stratify=stratify_labels,
    )
    X_pool, X_test_raw, y_pool, y_test_raw = train_test_split(
        X_all,
        y_all,
        test_size=0.1,
        random_state=split_seed,
        stratify=stratify_labels,
    )
    X_sampled, y_sampled = MC_DATASET.sample_training_pool(
        X_pool,
        y_pool,
        sample_size=cfg.sample_size,
        random_state=split_seed,
    )
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_sampled,
        y_sampled,
        train_size=0.7,
        test_size=0.3,
        random_state=split_seed,
        stratify=MC_DATASET.get_stratify_labels(y_sampled),
    )

    n_features = X_all.shape[-1]
    gpgl_scaler = MC_DATASET.preprocessing.StandardScaler()
    X_gpgl_scaled = gpgl_scaler.fit_transform(X_gpgl_raw.reshape(-1, n_features)).reshape(X_gpgl_raw.shape)

    scaler = MC_DATASET.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
    X_val = scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

    return {
        "X_gpgl_raw": np.asarray(X_gpgl_raw, dtype=np.float32),
        "X_gpgl_scaled": np.asarray(X_gpgl_scaled, dtype=np.float32),
        "y_gpgl": np.asarray(y_gpgl, dtype=np.int64),
        "X_train": np.asarray(X_train, dtype=np.float32),
        "X_val": np.asarray(X_val, dtype=np.float32),
        "X_test": np.asarray(X_test, dtype=np.float32),
        "y_train": np.asarray(y_train_raw, dtype=np.int64),
        "y_val": np.asarray(y_val_raw, dtype=np.int64),
        "y_test": np.asarray(y_test_raw, dtype=np.int64),
    }


def fit_gpgl_adjacency_multiclass(X_train_scaled, y_train, cfg, device, prior_matrix, beta, seed):
    seed_everything(seed)
    X_train_gnn = to_gnn_windows(X_train_scaled)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_gnn), torch.from_numpy(np.asarray(y_train, dtype=np.int64))),
        batch_size=int(cfg.gpgl_batch_size),
        shuffle=True,
        num_workers=int(cfg.num_workers),
        pin_memory=torch.cuda.is_available(),
    )
    model = GPGLGraphLearner(
        n_nodes=X_train_gnn.shape[1],
        window_size=int(cfg.n_step),
        n_hidden=int(cfg.gpgl_n_hidden),
        k=0,
        beta=float(beta),
        bias=float(cfg.gpgl_bias),
        init_std=float(cfg.gpgl_init_std),
        device=device,
    ).to(device)
    model.fc = nn.Linear(int(cfg.gpgl_n_hidden), int(cfg.num_classes)).to(device)

    if prior_matrix is not None:
        prior_tensor = torch.as_tensor(symmetrize_adjacency(prior_matrix), dtype=torch.float32, device=device)
        with torch.no_grad():
            model.graph.P.copy_(prior_tensor)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.gpgl_lr))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))

    model.train()
    for _ in range(int(cfg.gpgl_pretrain_epochs)):
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = criterion(logits, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return symmetrize_adjacency(model.adjacency().detach().cpu().numpy())


def evaluate_classifier(model, loader, criterion, device, num_classes):
    model.eval()
    total = 0
    running_loss = 0.0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total += batch_size
            running_loss += loss.item() * batch_size
            y_true.extend(labels.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())

    metrics = compute_multiclass_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = running_loss / max(total, 1)
    return metrics


def train_classifier(model, train_loader, val_loader, test_loader, device, max_epochs, patience, lr, weight_decay, seed, num_classes, label_smoothing):
    seed_everything(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    optimizer = optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    best_state = copy.deepcopy(model.state_dict())
    best_acc = -1.0
    best_loss = float("inf")
    patience_counter = 0

    for _ in range(int(max_epochs)):
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate_classifier(model, val_loader, criterion, device, num_classes)
        improved = (
            val_metrics["acc"] > best_acc
            or (
                math.isclose(val_metrics["acc"], best_acc, rel_tol=0.0, abs_tol=1e-12)
                and val_metrics["loss"] < best_loss
            )
        )
        if improved:
            best_acc = val_metrics["acc"]
            best_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= int(patience):
                break

    model.load_state_dict(best_state)
    test_metrics = evaluate_classifier(model, test_loader, criterion, device, num_classes)
    return {
        "state_dict": best_state,
        "test_metrics": test_metrics,
    }


def build_clifford_arrays(bundle, adjacency):
    coords = MC_DATASET.adjacency_to_layout(adjacency, size=5)
    return {
        "train": windows_to_gpgl_input(bundle["X_train"], coords),
        "val": windows_to_gpgl_input(bundle["X_val"], coords),
        "test": windows_to_gpgl_input(bundle["X_test"], coords),
    }


def run_graph_aux_head(bundle, adjacency, cfg, args, device, train_seed):
    train_loader = make_loader(
        to_gnn_windows(bundle["X_train"]),
        bundle["y_train"],
        args.head_batch_size,
        True,
        args.num_workers,
    )
    val_loader = make_loader(
        to_gnn_windows(bundle["X_val"]),
        bundle["y_val"],
        args.head_batch_size,
        False,
        args.num_workers,
    )
    test_loader = make_loader(
        to_gnn_windows(bundle["X_test"]),
        bundle["y_test"],
        args.head_batch_size,
        False,
        args.num_workers,
    )
    model = GraphAuxClassifier(
        adjacency=adjacency,
        seq_len=int(bundle["X_train"].shape[1]),
        hidden_dim=max(int(cfg.tcn_out_dim), 32),
        dropout=float(cfg.fusion_dropout),
        num_classes=int(cfg.num_classes),
    )
    return train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=args.head_epochs,
        patience=args.head_patience,
        lr=args.head_lr,
        weight_decay=args.head_weight_decay,
        seed=train_seed,
        num_classes=int(cfg.num_classes),
        label_smoothing=float(cfg.label_smoothing),
    )


def run_clifford_aux_head(bundle, adjacency, cfg, args, device, train_seed):
    arrays = build_clifford_arrays(bundle, adjacency)
    train_loader = make_loader(arrays["train"], bundle["y_train"], args.head_batch_size, True, args.num_workers)
    val_loader = make_loader(arrays["val"], bundle["y_val"], args.head_batch_size, False, args.num_workers)
    test_loader = make_loader(arrays["test"], bundle["y_test"], args.head_batch_size, False, args.num_workers)
    model = CliffordAuxClassifier(gpgl_in_chans=arrays["train"].shape[1], cfg=cfg)
    return train_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        max_epochs=args.head_epochs,
        patience=args.head_patience,
        lr=args.head_lr,
        weight_decay=args.head_weight_decay,
        seed=train_seed,
        num_classes=int(cfg.num_classes),
        label_smoothing=float(cfg.label_smoothing),
    )


def get_learned_adjacency(kind, seed, bundle, cfg, device, cache):
    canonical_kind = "ptcnet-full" if kind == "correct-p" else kind
    key = (canonical_kind, int(seed))
    if key in cache:
        return cache[key]

    prior = build_uav_prior()
    if canonical_kind == "ptcnet-full":
        adjacency = fit_gpgl_adjacency_multiclass(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=prior,
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "w.o.p":
        adjacency = fit_gpgl_adjacency_multiclass(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=prior,
            beta=0.0,
            seed=int(seed),
        )
    elif canonical_kind == "shuffled-p":
        adjacency = fit_gpgl_adjacency_multiclass(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=shuffled_prior(prior, seed),
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "random-p":
        adjacency = fit_gpgl_adjacency_multiclass(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=random_prior_like(prior, seed),
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "no-p":
        adjacency = fit_gpgl_adjacency_multiclass(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=prior,
            beta=0.0,
            seed=int(seed),
        )
    else:
        raise ValueError(f"Unsupported learned adjacency kind: {kind}")

    cache[key] = adjacency
    return adjacency


def get_experiment1_adjacency(mode, seed, bundle, cfg, device, cache):
    key = (mode, int(seed))
    if key in cache:
        return cache[key]

    if mode in ("ptcnet-full", "w.o.p"):
        adjacency = get_learned_adjacency(mode, seed, bundle, cfg, device, cache)
    elif mode == "correlation-graph":
        adjacency = compute_pearson_adjacency(bundle["X_gpgl_raw"])
    elif mode == "random-graph":
        adjacency = compute_random_adjacency(len(FEATURE_COLUMNS), seed)
    else:
        raise ValueError(f"Unsupported graph mode: {mode}")

    adjacency = symmetrize_adjacency(adjacency)
    cache[key] = adjacency
    return adjacency


def aggregate_metric_rows(rows, group_keys):
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key in sorted(grouped):
        items = grouped[key]
        acc_mean, acc_std = mean_std(item["acc"] for item in items)
        macro_f1_mean, macro_f1_std = mean_std(item["macro_f1"] for item in items)
        payload = {name: value for name, value in zip(group_keys, key)}
        payload.update(
            {
                "acc": format_pct(acc_mean, acc_std),
                "macro_f1": format_pct(macro_f1_mean, macro_f1_std),
            }
        )
        aggregated.append(payload)
    return aggregated


def run_experiment_one(bundle, cfg, args, device, adjacency_cache):
    print("\n=== Experiment 1: Graph-only budget efficiency (5-class) ===")
    rows = []
    edge_model_cache = {}

    graph_modes = GRAPH_MODE_ORDER if not args.smoke else ["ptcnet-full", "correlation-graph"]
    budgets = EXP1_K if not args.smoke else [1, 3]
    seeds = args.seeds if not args.smoke else args.seeds[:1]

    for graph_mode in graph_modes:
        for seed in seeds:
            print(f"[Experiment 1] Building source graph: {GRAPH_MODE_LABELS[graph_mode]} | seed={seed}")
            base_A = get_experiment1_adjacency(graph_mode, seed, bundle, cfg, device, adjacency_cache)
            for top_k in budgets:
                final_A = prune_adjacency_top_k(base_A, top_k)
                print(f"[Experiment 1] {GRAPH_MODE_LABELS[graph_mode]} | k={top_k} | seed={seed}")

                graph_result = run_graph_aux_head(bundle, final_A, cfg, args, device, train_seed=int(seed) + 1000)
                rows.append(
                    {
                        "experiment": "exp1",
                        "head": "graph_aux",
                        "graph_mode": GRAPH_MODE_LABELS[graph_mode],
                        "top_k": int(top_k),
                        "seed": int(seed),
                        "acc": float(graph_result["test_metrics"]["acc"]),
                        "macro_f1": float(graph_result["test_metrics"]["macro_f1"]),
                    }
                )
                if top_k in EDGE_KNOCKOUT_K:
                    edge_model_cache[(graph_mode, int(seed), int(top_k))] = {
                        "state_dict": copy.deepcopy(graph_result["state_dict"]),
                        "adjacency": np.asarray(final_A, dtype=np.float32),
                        "baseline_metrics": {
                            "acc": float(graph_result["test_metrics"]["acc"]),
                            "macro_f1": float(graph_result["test_metrics"]["macro_f1"]),
                        },
                    }

                clifford_result = run_clifford_aux_head(bundle, final_A, cfg, args, device, train_seed=int(seed) + 2000)
                rows.append(
                    {
                        "experiment": "exp1",
                        "head": "clifford_aux",
                        "graph_mode": GRAPH_MODE_LABELS[graph_mode],
                        "top_k": int(top_k),
                        "seed": int(seed),
                        "acc": float(clifford_result["test_metrics"]["acc"]),
                        "macro_f1": float(clifford_result["test_metrics"]["macro_f1"]),
                    }
                )

    return rows, edge_model_cache


def run_experiment_two(bundle, cfg, args, device, adjacency_cache):
    print("\n=== Experiment 2: Correct / Shuffled / Random / No P (5-class) ===")
    rows = []
    variants = PRIOR_VARIANT_ORDER if not args.smoke else ["correct-p", "no-p"]
    budgets = EXP2_K if not args.smoke else [5]
    seeds = args.seeds if not args.smoke else args.seeds[:1]

    for variant in variants:
        for seed in seeds:
            print(f"[Experiment 2] Building source graph: {PRIOR_VARIANT_LABELS[variant]} | seed={seed}")
            base_A = get_learned_adjacency(variant, seed, bundle, cfg, device, adjacency_cache)
            for top_k in budgets:
                final_A = prune_adjacency_top_k(base_A, top_k)
                print(f"[Experiment 2] {PRIOR_VARIANT_LABELS[variant]} | k={top_k} | seed={seed}")

                graph_result = run_graph_aux_head(bundle, final_A, cfg, args, device, train_seed=int(seed) + 3000)
                rows.append(
                    {
                        "experiment": "exp2",
                        "head": "graph_aux",
                        "prior_variant": PRIOR_VARIANT_LABELS[variant],
                        "top_k": int(top_k),
                        "seed": int(seed),
                        "acc": float(graph_result["test_metrics"]["acc"]),
                        "macro_f1": float(graph_result["test_metrics"]["macro_f1"]),
                    }
                )

                clifford_result = run_clifford_aux_head(bundle, final_A, cfg, args, device, train_seed=int(seed) + 4000)
                rows.append(
                    {
                        "experiment": "exp2",
                        "head": "clifford_aux",
                        "prior_variant": PRIOR_VARIANT_LABELS[variant],
                        "top_k": int(top_k),
                        "seed": int(seed),
                        "acc": float(clifford_result["test_metrics"]["acc"]),
                        "macro_f1": float(clifford_result["test_metrics"]["macro_f1"]),
                    }
                )

    return rows


def experiment_three_overlap_rows(adjacency_cache, seeds, budgets):
    rows = []
    have_corr = all(("correlation-graph", int(seed)) in adjacency_cache for seed in seeds)
    have_wop = all(("w.o.p", int(seed)) in adjacency_cache for seed in seeds)
    for top_k in budgets:
        overlap_full_corr = []
        overlap_full_wop = []
        for seed in seeds:
            full_A = adjacency_cache[("ptcnet-full", int(seed))]
            if have_corr:
                corr_A = adjacency_cache[("correlation-graph", int(seed))]
                overlap_full_corr.append(jaccard_edge_overlap(full_A, corr_A, top_k))
            if have_wop:
                wop_A = adjacency_cache[("w.o.p", int(seed))]
                overlap_full_wop.append(jaccard_edge_overlap(full_A, wop_A, top_k))
        corr_mean, corr_std = mean_std(overlap_full_corr)
        wop_mean, wop_std = mean_std(overlap_full_wop)
        rows.append(
            {
                "top_k": int(top_k),
                "PTCNet-full vs Correlation": format_pct(corr_mean, corr_std) if have_corr else "-",
                "PTCNet-full vs w.o.P": format_pct(wop_mean, wop_std) if have_wop else "-",
            }
        )
    return rows


def experiment_three_prior_rows(adjacency_cache, seeds, budgets, modes):
    prior_edges = prior_edge_mask()
    rows = []
    for mode in modes:
        for top_k in budgets:
            values = []
            for seed in seeds:
                edges = edge_set_from_adjacency(adjacency_cache[(mode, int(seed))], top_k)
                values.append(len(edges & prior_edges) / max(len(edges), 1) if edges else 0.0)
            mean_value, std_value = mean_std(values)
            rows.append(
                {
                    "graph_mode": GRAPH_MODE_LABELS[mode],
                    "top_k": int(top_k),
                    "prior_hit_ratio": format_pct(mean_value, std_value),
                }
            )
    return rows


def experiment_three_seed_rows(adjacency_cache, seeds, budgets, modes):
    rows = []
    seed_pairs = [(seeds[i], seeds[j]) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    for mode in modes:
        for top_k in budgets:
            values = []
            for seed_a, seed_b in seed_pairs:
                values.append(
                    jaccard_edge_overlap(
                        adjacency_cache[(mode, int(seed_a))],
                        adjacency_cache[(mode, int(seed_b))],
                        top_k,
                    )
                )
            mean_value, std_value = mean_std(values)
            rows.append(
                {
                    "graph_mode": GRAPH_MODE_LABELS[mode],
                    "top_k": int(top_k),
                    "seed_stability": format_pct(mean_value, std_value),
                }
            )
    return rows


def run_edge_knockout(bundle, cfg, args, device, edge_model_cache):
    rows = []
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg.label_smoothing))
    test_loader = make_loader(
        to_gnn_windows(bundle["X_test"]),
        bundle["y_test"],
        args.head_batch_size,
        False,
        args.num_workers,
    )

    for (graph_mode, seed, top_k), payload in sorted(edge_model_cache.items()):
        adjacency = payload["adjacency"]
        active_edges = active_edge_count(adjacency)
        if active_edges <= 0:
            continue

        drop_count = max(1, int(round(0.2 * min(top_k, active_edges))))
        top_edges = strongest_edges(adjacency, drop_count)
        random_deltas = []

        model = GraphAuxClassifier(
            adjacency=adjacency,
            seq_len=int(bundle["X_train"].shape[1]),
            hidden_dim=max(int(cfg.tcn_out_dim), 32),
            dropout=float(cfg.fusion_dropout),
            num_classes=int(cfg.num_classes),
        )
        model.load_state_dict(payload["state_dict"])
        model = model.to(device)

        baseline_acc = float(payload["baseline_metrics"]["acc"])
        baseline_macro_f1 = float(payload["baseline_metrics"]["macro_f1"])

        model.set_adjacency(remove_edges(adjacency, top_edges))
        top_metrics = evaluate_classifier(model, test_loader, criterion, device, int(cfg.num_classes))
        delta_top = baseline_macro_f1 - float(top_metrics["macro_f1"])

        for repeat_idx in range(5):
            random_edges = random_active_edges(adjacency, drop_count, seed=int(seed) + 7000 + repeat_idx)
            model.set_adjacency(remove_edges(adjacency, random_edges))
            random_metrics = evaluate_classifier(model, test_loader, criterion, device, int(cfg.num_classes))
            random_deltas.append(baseline_macro_f1 - float(random_metrics["macro_f1"]))

        rows.append(
            {
                "graph_mode": GRAPH_MODE_LABELS[graph_mode],
                "top_k": int(top_k),
                "seed": int(seed),
                "baseline_acc": baseline_acc,
                "baseline_macro_f1": baseline_macro_f1,
                "delta_top": float(delta_top),
                "delta_random": float(np.mean(random_deltas)),
            }
        )

    grouped = {}
    for row in rows:
        grouped.setdefault((row["graph_mode"], row["top_k"]), []).append(row)

    aggregated = []
    for key in sorted(grouped):
        items = grouped[key]
        baseline_acc_mean, baseline_acc_std = mean_std(item["baseline_acc"] for item in items)
        baseline_f1_mean, baseline_f1_std = mean_std(item["baseline_macro_f1"] for item in items)
        top_mean, top_std = mean_std(item["delta_top"] for item in items)
        random_mean, random_std = mean_std(item["delta_random"] for item in items)
        aggregated.append(
            {
                "graph_mode": key[0],
                "top_k": key[1],
                "baseline_acc": format_pct(baseline_acc_mean, baseline_acc_std),
                "baseline_macro_f1": format_pct(baseline_f1_mean, baseline_f1_std),
                "delta_top": format_pct(top_mean, top_std),
                "delta_random": format_pct(random_mean, random_std),
            }
        )
    return aggregated


def run_experiment_three(bundle, cfg, args, device, adjacency_cache, edge_model_cache):
    print("\n=== Experiment 3: Structural behavior analysis (5-class) ===")
    seeds = args.seeds if not args.smoke else args.seeds[:1]
    modes = [mode for mode in GRAPH_MODE_ORDER if all((mode, int(seed)) in adjacency_cache for seed in seeds)]
    budgets = EXP1_K if not args.smoke else [1, 3]
    return {
        "overlap": experiment_three_overlap_rows(adjacency_cache, seeds, budgets),
        "prior_concentration": experiment_three_prior_rows(adjacency_cache, seeds, budgets, modes),
        "seed_stability": experiment_three_seed_rows(adjacency_cache, seeds, budgets, modes),
        "edge_knockout": run_edge_knockout(bundle, cfg, args, device, edge_model_cache),
    }


def render_experiment_one(rows):
    sections = ["## Experiment 1: Graph-only Budget Efficiency (5-class)"]
    aggregated = aggregate_metric_rows(rows, ["head", "graph_mode", "top_k"])
    graph_order = {label: idx for idx, label in enumerate(GRAPH_MODE_LABELS.values())}
    for head in HEAD_ORDER:
        head_rows = sorted(
            [row for row in aggregated if row["head"] == head],
            key=lambda row: (graph_order[row["graph_mode"]], int(row["top_k"])),
        )
        table_rows = [[row["graph_mode"], row["top_k"], row["acc"], row["macro_f1"]] for row in head_rows]
        sections.append(f"\n### {HEAD_LABELS[head]}")
        sections.append(markdown_table(["Graph Mode", "top-k", "ACC", "Macro-F1"], table_rows))
    return "\n".join(sections)


def render_experiment_two(rows):
    sections = ["## Experiment 2: Correct / Shuffled / Random / No P (5-class)"]
    aggregated = aggregate_metric_rows(rows, ["head", "prior_variant", "top_k"])
    prior_order = {label: idx for idx, label in enumerate(PRIOR_VARIANT_LABELS.values())}
    for head in HEAD_ORDER:
        head_rows = sorted(
            [row for row in aggregated if row["head"] == head],
            key=lambda row: (prior_order[row["prior_variant"]], int(row["top_k"])),
        )
        table_rows = [[row["prior_variant"], row["top_k"], row["acc"], row["macro_f1"]] for row in head_rows]
        sections.append(f"\n### {HEAD_LABELS[head]}")
        sections.append(markdown_table(["Prior Variant", "top-k", "ACC", "Macro-F1"], table_rows))
    return "\n".join(sections)


def render_experiment_three(payload):
    sections = ["## Experiment 3: Structural Behavior Analysis (5-class)"]
    sections.append("\n### Overlap@k")
    sections.append(
        markdown_table(
            ["top-k", "PTCNet-full vs Correlation", "PTCNet-full vs w.o.P"],
            [[row["top_k"], row["PTCNet-full vs Correlation"], row["PTCNet-full vs w.o.P"]] for row in payload["overlap"]],
        )
    )
    sections.append("\n### Prior-edge Concentration")
    sections.append(
        markdown_table(
            ["Graph Mode", "top-k", "Prior Hit Ratio"],
            [[row["graph_mode"], row["top_k"], row["prior_hit_ratio"]] for row in payload["prior_concentration"]],
        )
    )
    sections.append("\n### Seed Stability")
    sections.append(
        markdown_table(
            ["Graph Mode", "top-k", "Seed Stability"],
            [[row["graph_mode"], row["top_k"], row["seed_stability"]] for row in payload["seed_stability"]],
        )
    )
    sections.append("\n### Edge Knockout (Graph Auxiliary Head)")
    sections.append(
        markdown_table(
            ["Graph Mode", "top-k", "Baseline ACC", "Baseline Macro-F1", "Delta Macro-F1 (Top)", "Delta Macro-F1 (Random)"],
            [
                [row["graph_mode"], row["top_k"], row["baseline_acc"], row["baseline_macro_f1"], row["delta_top"], row["delta_random"]]
                for row in payload["edge_knockout"]
            ],
        )
    )
    return "\n".join(sections)


def write_markdown_report(path, exp1_rows, exp2_rows, exp3_payload):
    lines = ["# Graph Mechanism Experiment Results (5-class)", ""]
    lines.append(render_experiment_one(exp1_rows))
    lines.append("")
    lines.append(render_experiment_two(exp2_rows))
    lines.append("")
    lines.append(render_experiment_three(exp3_payload))

    ensure_parent(path)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = build_parser().parse_args()
    cfg = build_base_config(args)
    args.sample_size = int(cfg.sample_size)
    experiment_root = build_experiment_root(args)
    markdown_path = experiment_root / "graph_mechanism_multiclass5_results.md"

    if args.head_lr is None:
        args.head_lr = float(cfg.fusion_lr)
    if args.head_weight_decay is None:
        args.head_weight_decay = float(cfg.fusion_weight_decay)

    print("\n===== Graph Mechanism Multiclass-5 Experiment Config =====")
    print(f"results_root: {experiment_root}")
    print(f"data_path: {cfg.data_path}")
    print(f"aircraft_id: {cfg.aircraft_id}")
    print(f"segment: [{cfg.segment_start}, {cfg.segment_end}]")
    print(f"mode_filter: {args.mode_filter}")
    print(f"sample_size: {cfg.sample_size}")
    print(f"num_classes: {cfg.num_classes}")
    print(f"data_split_seed: {cfg.data_split_seed}")
    print(f"seeds: {args.seeds}")
    print(f"head_batch_size: {args.head_batch_size}")
    print(f"head_epochs: {args.head_epochs}")
    print(f"head_patience: {args.head_patience}")
    print(f"head_lr: {args.head_lr}")
    print(f"head_weight_decay: {args.head_weight_decay}")
    print(f"gpgl_pretrain_epochs: {cfg.gpgl_pretrain_epochs}")
    print("===========================================\n")

    device = get_device(cfg.device)
    start_time = time.time()

    print("Preparing shared 5-class data split and scaled windows...")
    bundle = prepare_data_bundle(cfg, args)
    adjacency_cache = {}

    exp1_rows, edge_model_cache = run_experiment_one(bundle, cfg, args, device, adjacency_cache)
    exp2_rows = run_experiment_two(bundle, cfg, args, device, adjacency_cache)
    exp3_payload = run_experiment_three(bundle, cfg, args, device, adjacency_cache, edge_model_cache)

    write_markdown_report(markdown_path, exp1_rows, exp2_rows, exp3_payload)
    elapsed_minutes = (time.time() - start_time) / 60.0
    print(f"\nAll experiments completed in {elapsed_minutes:.2f} minutes.")
    print(f"Markdown report saved to: {markdown_path}")


if __name__ == "__main__":
    main()
