import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

try:
    from .dataset import (
        FEATURE_COLUMNS,
        GPGLGraphLearner,
        add_fault_label,
        adjacency_to_layout,
        build_time_windows,
        build_uav_prior,
        compute_pearson_adjacency,
        compute_random_adjacency,
        prune_adjacency_top_k,
        resolve_task_path,
        scale_windows,
        symmetrize_adjacency,
        to_gnn_windows,
        windows_to_gpgl_input,
    )
    from .model import CliffordNet
    from .trainer import TrainingConfig, compute_binary_metrics, get_device, seed_everything
except ImportError:
    from dataset import (
        FEATURE_COLUMNS,
        GPGLGraphLearner,
        add_fault_label,
        adjacency_to_layout,
        build_time_windows,
        build_uav_prior,
        compute_pearson_adjacency,
        compute_random_adjacency,
        prune_adjacency_top_k,
        resolve_task_path,
        scale_windows,
        symmetrize_adjacency,
        to_gnn_windows,
        windows_to_gpgl_input,
    )
    from model import CliffordNet
    from trainer import TrainingConfig, compute_binary_metrics, get_device, seed_everything

from pprz_data.pprz_data import DATA


GRAPH_MODE_ORDER = ["ptcnet-full", "w.o.p", "correlation-graph", "random-graph"]
GRAPH_MODE_LABELS = {
    "ptcnet-full": "PTCNet-full",
    "w.o.p": "w.o.P",
    "correlation-graph": "Correlation-graph",
    "random-graph": "Random-graph",
}
PRIOR_VARIANT_ORDER = ["correct-p", "shuffled-p", "random-p", "no-p"]
PRIOR_VARIANT_LABELS = {
    "correct-p": "Correct P",
    "shuffled-p": "Shuffled P",
    "random-p": "Random P",
    "no-p": "No P",
}
HEAD_ORDER = ["graph_aux", "clifford_aux"]
HEAD_LABELS = {
    "graph_aux": "Graph Auxiliary Head",
    "clifford_aux": "Clifford Auxiliary Head",
}
EXP1_K = [1, 3, 5, 7, 10, 15, 20]
EXP2_K = [5, 10]
EDGE_KNOCKOUT_K = [5, 10]
DEFAULT_SEEDS = [42, 52, 62]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run graph-mechanism experiments and export a single markdown report."
    )
    parser.add_argument("--results-root", default="results/graph_mechanism", help="Output directory root.")
    parser.add_argument("--sample-size", type=int, default=360, help="TrainingConfig.sample_size.")
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
        help="Override TrainingConfig.gpgl_pretrain_epochs for learned graphs.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny version of the experiments for verification.",
    )
    return parser


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def build_experiment_root(args):
    return Path(args.results_root) / f"sample_{int(args.sample_size)}" / f"split_{int(args.data_split_seed)}"


def build_base_config(args):
    cfg = TrainingConfig()
    cfg.sample_size = int(args.sample_size)
    cfg.data_split_seed = int(args.data_split_seed)
    cfg.num_workers = int(args.num_workers)
    cfg.gpgl_k = 0
    cfg.gpgl_force_retrain = True
    if args.device is not None:
        cfg.device = args.device
    if args.gpgl_pretrain_epochs is not None:
        cfg.gpgl_pretrain_epochs = int(args.gpgl_pretrain_epochs)
    if args.smoke:
        cfg.sample_size = min(int(args.sample_size), 90)
        cfg.gpgl_pretrain_epochs = min(int(cfg.gpgl_pretrain_epochs), 2)
    return cfg


def format_pct(mean_value, std_value):
    return f"{mean_value * 100:.2f}+/-{std_value * 100:.2f}"


def format_num(mean_value, std_value):
    return f"{mean_value:.4f}+/-{std_value:.4f}"


def mean_std(values):
    values = np.asarray(list(values), dtype=np.float64)
    if values.size == 0:
        return 0.0, 0.0
    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0)) if values.size > 1 else 0.0
    return mean_value, std_value


def edge_set_from_adjacency(A, top_k):
    A = symmetrize_adjacency(A)
    n_nodes = A.shape[0]
    rows, cols = np.triu_indices(n_nodes, k=1)
    values = A[rows, cols]
    if top_k is None:
        selected = np.where(values > 1e-8)[0]
    else:
        top_k = max(0, min(int(top_k), len(values)))
        if top_k == 0:
            return set()
        selected = np.argsort(values)[::-1][:top_k]
    return {(int(rows[idx]), int(cols[idx])) for idx in selected if values[idx] > 1e-8}


def jaccard_edge_overlap(A, B, top_k):
    edges_a = edge_set_from_adjacency(A, top_k)
    edges_b = edge_set_from_adjacency(B, top_k)
    if not edges_a and not edges_b:
        return 1.0
    union = edges_a | edges_b
    return len(edges_a & edges_b) / max(len(union), 1)


def prior_edge_mask():
    return edge_set_from_adjacency(build_uav_prior(), None)


def shuffled_prior(prior, seed):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(prior.shape[0])
    shuffled = prior[np.ix_(perm, perm)]
    return symmetrize_adjacency(shuffled)


def random_prior_like(prior, seed):
    prior = symmetrize_adjacency(prior)
    rng = np.random.default_rng(int(seed))
    rows, cols = np.triu_indices(prior.shape[0], k=1)
    weights = prior[rows, cols]
    active_weights = weights[weights > 1e-8]
    selected = rng.choice(rows.size, size=active_weights.size, replace=False)
    shuffled_weights = rng.permutation(active_weights)
    random_prior = np.zeros_like(prior)
    for idx, weight in zip(selected, shuffled_weights):
        i = int(rows[idx])
        j = int(cols[idx])
        random_prior[i, j] = float(weight)
        random_prior[j, i] = float(weight)
    return symmetrize_adjacency(random_prior)


def normalize_adjacency_with_self_loops(A):
    A = symmetrize_adjacency(A)
    A = A + np.eye(A.shape[0], dtype=np.float32)
    degree = np.clip(A.sum(axis=1), a_min=1e-8, a_max=None)
    inv_sqrt = np.power(degree, -0.5)
    return (inv_sqrt[:, None] * A * inv_sqrt[None, :]).astype(np.float32)


def prepare_data_bundle(cfg):
    data_path = resolve_task_path(cfg.data_path)
    raw_data = DATA(str(data_path), cfg.aircraft_id, data_type="flight", sample_period=0.1)
    frame = raw_data.get_labelled_data()
    frame = frame.loc[cfg.segment_start:cfg.segment_end].copy()
    frame = add_fault_label(frame)

    X_raw = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_raw = frame["fault"].to_numpy(dtype=np.int64)
    X_all, y_all = build_time_windows(X_raw, y_raw, n_step=int(cfg.n_step))

    split_seed = int(cfg.data_split_seed if cfg.data_split_seed is not None else cfg.seed)
    X_gpgl_raw, _, y_gpgl, _ = train_test_split(
        X_all,
        y_all,
        train_size=0.8,
        random_state=split_seed,
        stratify=y_all,
    )
    X_pool, X_test_raw, y_pool, y_test_raw = train_test_split(
        X_all,
        y_all,
        test_size=0.1,
        random_state=split_seed,
        stratify=y_all,
    )
    X_sampled, _, y_sampled, _ = train_test_split(
        X_pool,
        y_pool,
        train_size=int(cfg.sample_size),
        random_state=split_seed,
        stratify=y_pool,
    )
    X_train_raw, X_val_raw, y_train_raw, y_val_raw = train_test_split(
        X_sampled,
        y_sampled,
        train_size=0.7,
        test_size=0.3,
        random_state=split_seed,
        stratify=y_sampled,
    )

    X_train, X_val, X_test = scale_windows(X_train_raw, X_val_raw, X_test_raw)
    X_gpgl_scaled = scale_windows(X_gpgl_raw)[0]

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


def fit_gpgl_adjacency_with_prior(X_train_scaled, y_train, cfg, device, prior_matrix, beta, seed):
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
    if prior_matrix is not None:
        prior_tensor = torch.as_tensor(symmetrize_adjacency(prior_matrix), dtype=torch.float32, device=device)
        with torch.no_grad():
            model.graph.P.copy_(prior_tensor)

    optimizer = optim.Adam(model.parameters(), lr=float(cfg.gpgl_lr))
    model.train()
    for _ in range(int(cfg.gpgl_pretrain_epochs)):
        for batch_x, batch_y in train_loader:
            logits = model(batch_x)
            loss = nn.functional.cross_entropy(logits, batch_y.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return symmetrize_adjacency(model.adjacency().detach().cpu().numpy())


class GraphAuxClassifier(nn.Module):
    def __init__(self, adjacency, seq_len, hidden_dim=64, dropout=0.2, num_classes=2):
        super().__init__()
        self.register_buffer("adj_norm", torch.zeros(adjacency.shape[0], adjacency.shape[1]))
        self.set_adjacency(adjacency)
        self.input_proj = nn.Linear(seq_len, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def set_adjacency(self, adjacency):
        adj_norm = normalize_adjacency_with_self_loops(adjacency)
        self.adj_norm.copy_(torch.as_tensor(adj_norm, dtype=torch.float32, device=self.adj_norm.device))

    def graph_step(self, x):
        return torch.matmul(self.adj_norm.unsqueeze(0), x)

    def forward(self, x):
        hidden = self.input_proj(x)
        hidden = self.norm1(hidden)
        hidden = F.silu(self.graph_step(hidden))
        hidden = self.dropout(hidden)
        hidden = self.graph_proj(hidden)
        hidden = self.norm2(hidden)
        hidden = F.silu(self.graph_step(hidden))
        hidden = self.dropout(hidden)
        pooled = hidden.mean(dim=1)
        return self.classifier(pooled)


class CliffordAuxClassifier(nn.Module):
    def __init__(self, gpgl_in_chans, cfg):
        super().__init__()
        self.encoder = CliffordNet(
            num_classes=int(cfg.num_classes),
            patch_size=int(cfg.patch_size),
            embed_dim=int(cfg.embed_dim),
            depth=int(cfg.depth),
            drop_path_rate=float(cfg.drop_path_rate),
            in_chans=int(gpgl_in_chans),
        )
        self.encoder.head = nn.Identity()
        self.classifier = nn.Sequential(
            nn.Linear(int(cfg.embed_dim), int(cfg.fusion_hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(cfg.fusion_dropout)),
            nn.Linear(int(cfg.fusion_hidden_dim), int(cfg.num_classes)),
        )

    def forward(self, x):
        return self.classifier(self.encoder.forward_features(x))


def make_loader(features, labels, batch_size, shuffle, num_workers):
    dataset = TensorDataset(torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.long))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_classifier(model, loader, criterion, device):
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

    avg_loss = running_loss / max(total, 1)
    acc, precision, recall, f1, _ = compute_binary_metrics(y_true, y_pred)
    return {
        "loss": float(avg_loss),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train_classifier(model, train_loader, val_loader, test_loader, device, max_epochs, patience, lr, weight_decay, seed):
    seed_everything(seed)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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

        val_metrics = evaluate_classifier(model, val_loader, criterion, device)
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
    test_metrics = evaluate_classifier(model, test_loader, criterion, device)
    return {
        "model": model.cpu(),
        "state_dict": best_state,
        "test_metrics": test_metrics,
    }


def build_clifford_arrays(bundle, adjacency):
    coords = adjacency_to_layout(adjacency, size=5)
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
    )


def active_edge_count(A):
    rows, cols = np.triu_indices(A.shape[0], k=1)
    return int(np.sum(np.asarray(A[rows, cols]) > 1e-8))


def remove_edges(A, edges):
    pruned = np.asarray(A, dtype=np.float32).copy()
    for i, j in edges:
        pruned[i, j] = 0.0
        pruned[j, i] = 0.0
    return symmetrize_adjacency(pruned)


def strongest_edges(A, count):
    rows, cols = np.triu_indices(A.shape[0], k=1)
    values = np.asarray(A[rows, cols], dtype=np.float32)
    active_idx = np.where(values > 1e-8)[0]
    if active_idx.size == 0:
        return []
    order = active_idx[np.argsort(values[active_idx])[::-1]]
    return [(int(rows[idx]), int(cols[idx])) for idx in order[:count]]


def random_active_edges(A, count, seed):
    rows, cols = np.triu_indices(A.shape[0], k=1)
    values = np.asarray(A[rows, cols], dtype=np.float32)
    active_idx = np.where(values > 1e-8)[0]
    if active_idx.size == 0:
        return []
    rng = np.random.default_rng(int(seed))
    selected = rng.choice(active_idx, size=min(count, active_idx.size), replace=False)
    return [(int(rows[idx]), int(cols[idx])) for idx in selected]


def get_learned_adjacency(kind, seed, bundle, cfg, device, cache):
    canonical_kind = "ptcnet-full" if kind == "correct-p" else kind
    key = (canonical_kind, int(seed))
    if key in cache:
        return cache[key]

    prior = build_uav_prior()
    if canonical_kind == "ptcnet-full":
        adjacency = fit_gpgl_adjacency_with_prior(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=prior,
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "w.o.p":
        adjacency = fit_gpgl_adjacency_with_prior(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=prior,
            beta=0.0,
            seed=int(seed),
        )
    elif canonical_kind == "shuffled-p":
        adjacency = fit_gpgl_adjacency_with_prior(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=shuffled_prior(prior, seed),
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "random-p":
        adjacency = fit_gpgl_adjacency_with_prior(
            bundle["X_gpgl_scaled"],
            bundle["y_gpgl"],
            cfg,
            device,
            prior_matrix=random_prior_like(prior, seed),
            beta=float(cfg.gpgl_beta),
            seed=int(seed),
        )
    elif canonical_kind == "no-p":
        adjacency = fit_gpgl_adjacency_with_prior(
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
        key = tuple(row[key_name] for key_name in group_keys)
        grouped.setdefault(key, []).append(row)

    aggregated = []
    for key in sorted(grouped):
        items = grouped[key]
        acc_mean, acc_std = mean_std(item["acc"] for item in items)
        precision_mean, precision_std = mean_std(item["precision"] for item in items)
        recall_mean, recall_std = mean_std(item["recall"] for item in items)
        f1_mean, f1_std = mean_std(item["f1"] for item in items)

        payload = {name: value for name, value in zip(group_keys, key)}
        payload.update(
            {
                "acc": format_pct(acc_mean, acc_std),
                "precision": format_pct(precision_mean, precision_std),
                "recall": format_pct(recall_mean, recall_std),
                "f1": format_pct(f1_mean, f1_std),
            }
        )
        aggregated.append(payload)
    return aggregated


def markdown_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def run_experiment_one(bundle, cfg, args, device, adjacency_cache):
    print("\n=== Experiment 1: Graph-only budget efficiency ===")
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
                        **graph_result["test_metrics"],
                    }
                )
                if top_k in EDGE_KNOCKOUT_K:
                    edge_model_cache[(graph_mode, int(seed), int(top_k))] = {
                        "state_dict": copy.deepcopy(graph_result["state_dict"]),
                        "adjacency": np.asarray(final_A, dtype=np.float32),
                        "baseline_metrics": dict(graph_result["test_metrics"]),
                    }

                clifford_result = run_clifford_aux_head(bundle, final_A, cfg, args, device, train_seed=int(seed) + 2000)
                rows.append(
                    {
                        "experiment": "exp1",
                        "head": "clifford_aux",
                        "graph_mode": GRAPH_MODE_LABELS[graph_mode],
                        "top_k": int(top_k),
                        "seed": int(seed),
                        **clifford_result["test_metrics"],
                    }
                )

    return rows, edge_model_cache


def run_experiment_two(bundle, cfg, args, device, adjacency_cache):
    print("\n=== Experiment 2: Correct / Shuffled / Random / No P ===")
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
                        **graph_result["test_metrics"],
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
                        **clifford_result["test_metrics"],
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
                if not edges:
                    values.append(0.0)
                else:
                    values.append(len(edges & prior_edges) / max(len(edges), 1))
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
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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

        baseline_f1 = float(payload["baseline_metrics"]["f1"])
        knockout_top = remove_edges(adjacency, top_edges)
        model.set_adjacency(knockout_top)
        top_metrics = evaluate_classifier(model, test_loader, criterion, device)
        delta_top = baseline_f1 - float(top_metrics["f1"])

        for repeat_idx in range(5):
            random_edges = random_active_edges(adjacency, drop_count, seed=int(seed) + 7000 + repeat_idx)
            model.set_adjacency(remove_edges(adjacency, random_edges))
            random_metrics = evaluate_classifier(model, test_loader, criterion, device)
            random_deltas.append(baseline_f1 - float(random_metrics["f1"]))

        rows.append(
            {
                "graph_mode": GRAPH_MODE_LABELS[graph_mode],
                "top_k": int(top_k),
                "seed": int(seed),
                "baseline_f1": baseline_f1,
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
        baseline_mean, baseline_std = mean_std(item["baseline_f1"] for item in items)
        top_mean, top_std = mean_std(item["delta_top"] for item in items)
        random_mean, random_std = mean_std(item["delta_random"] for item in items)
        aggregated.append(
            {
                "graph_mode": key[0],
                "top_k": key[1],
                "baseline_f1": format_pct(baseline_mean, baseline_std),
                "delta_top": format_pct(top_mean, top_std),
                "delta_random": format_pct(random_mean, random_std),
            }
        )
    return aggregated


def run_experiment_three(bundle, cfg, args, device, adjacency_cache, edge_model_cache):
    print("\n=== Experiment 3: Structural behavior analysis ===")
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
    sections = ["## Experiment 1: Graph-only Budget Efficiency"]
    aggregated = aggregate_metric_rows(rows, ["head", "graph_mode", "top_k"])
    graph_order = {label: idx for idx, label in enumerate(GRAPH_MODE_LABELS.values())}
    for head in HEAD_ORDER:
        head_rows = sorted(
            [row for row in aggregated if row["head"] == head],
            key=lambda row: (graph_order[row["graph_mode"]], int(row["top_k"])),
        )
        table_rows = [
            [row["graph_mode"], row["top_k"], row["acc"], row["precision"], row["recall"], row["f1"]]
            for row in head_rows
        ]
        sections.append(f"\n### {HEAD_LABELS[head]}")
        sections.append(markdown_table(["Graph Mode", "top-k", "ACC", "Precision", "Recall", "F1"], table_rows))
    return "\n".join(sections)


def render_experiment_two(rows):
    sections = ["## Experiment 2: Correct / Shuffled / Random / No P"]
    aggregated = aggregate_metric_rows(rows, ["head", "prior_variant", "top_k"])
    prior_order = {label: idx for idx, label in enumerate(PRIOR_VARIANT_LABELS.values())}
    for head in HEAD_ORDER:
        head_rows = sorted(
            [row for row in aggregated if row["head"] == head],
            key=lambda row: (prior_order[row["prior_variant"]], int(row["top_k"])),
        )
        table_rows = [
            [row["prior_variant"], row["top_k"], row["acc"], row["precision"], row["recall"], row["f1"]]
            for row in head_rows
        ]
        sections.append(f"\n### {HEAD_LABELS[head]}")
        sections.append(markdown_table(["Prior Variant", "top-k", "ACC", "Precision", "Recall", "F1"], table_rows))
    return "\n".join(sections)


def render_experiment_three(payload):
    sections = ["## Experiment 3: Structural Behavior Analysis"]
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
            ["Graph Mode", "top-k", "Baseline F1", "Delta F1 (Top)", "Delta F1 (Random)"],
            [
                [row["graph_mode"], row["top_k"], row["baseline_f1"], row["delta_top"], row["delta_random"]]
                for row in payload["edge_knockout"]
            ],
        )
    )
    return "\n".join(sections)


def write_markdown_report(path, exp1_rows, exp2_rows, exp3_payload):
    lines = ["# Graph Mechanism Experiment Results", ""]
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
    markdown_path = experiment_root / "graph_mechanism_results.md"
    if args.head_lr is None:
        args.head_lr = float(cfg.fusion_lr)
    if args.head_weight_decay is None:
        args.head_weight_decay = float(cfg.fusion_weight_decay)

    print("\n===== Graph Mechanism Experiment Config =====")
    print(f"results_root: {experiment_root}")
    print(f"sample_size: {cfg.sample_size}")
    print(f"data_split_seed: {cfg.data_split_seed}")
    print(f"seeds: {args.seeds}")
    print(f"head_batch_size: {args.head_batch_size}")
    print(f"head_epochs: {args.head_epochs}")
    print(f"head_patience: {args.head_patience}")
    print(f"head_lr: {args.head_lr}")
    print(f"head_weight_decay: {args.head_weight_decay}")
    print(f"gpgl_pretrain_epochs: {cfg.gpgl_pretrain_epochs}")
    print("==============================\n")

    device = get_device(cfg.device)
    start_time = time.time()

    print("Preparing shared data split and scaled windows...")
    bundle = prepare_data_bundle(cfg)

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
