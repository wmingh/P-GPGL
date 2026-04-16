import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist, pdist
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pprz_data.pprz_data import DATA


FEATURE_COLUMNS = [
    "airspeed",
    "phi",
    "psi",
    "theta",
    "Ax",
    "Ay",
    "Az",
    "Gx",
    "Gy",
    "Gz",
    "C1",
    "C2",
]

GRAPH_MODE_ALIASES = {
    "ptcnet-full": "ptcnet-full",
    "learned_prior": "ptcnet-full",
    "full": "ptcnet-full",
    "w.o.p": "w.o.p",
    "wo-p": "w.o.p",
    "wop": "w.o.p",
    "learned_no_prior": "w.o.p",
    "correlation-graph": "correlation-graph",
    "correlation": "correlation-graph",
    "random-graph": "random-graph",
    "random": "random-graph",
}


class DualInputDataset(Dataset):
    def __init__(self, seq_inputs, gpgl_inputs, labels):
        self.seq_inputs = seq_inputs
        self.gpgl_inputs = gpgl_inputs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return (self.seq_inputs[index], self.gpgl_inputs[index]), self.labels[index]


def add_fault_label(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.assign(fault=0)

    cond1 = (frame["add1"] > 0.005) | (frame["add1"] < -0.005)
    cond2 = (frame["add2"] > 0.005) | (frame["add2"] < -0.005)
    cond3 = (frame["m1"] < 1.0) | (frame["m2"] < 1.0)
    frame.loc[cond1 | cond2 | cond3, "fault"] = 1
    return frame


def build_time_windows(X, y, n_step=20):
    time_len, feature_count = X.shape
    windows = np.zeros((time_len - n_step, n_step, feature_count), dtype=np.float32)

    for end in range(n_step, time_len):
        windows[end - n_step] = X[end - n_step:end]

    labels = y[n_step - 1:time_len - 1]
    return windows, labels


def scale_windows(train_windows, *other_sets):
    scaler = preprocessing.StandardScaler()
    feature_count = train_windows.shape[-1]

    train_scaled = scaler.fit_transform(train_windows.reshape(-1, feature_count)).reshape(train_windows.shape)
    scaled_sets = [train_scaled]
    for windows in other_sets:
        scaled = scaler.transform(windows.reshape(-1, feature_count)).reshape(windows.shape)
        scaled_sets.append(scaled)
    return scaled_sets


def resolve_task_path(path_text):
    path = Path(path_text)
    if path.is_absolute():
        return path

    module_dir = Path(__file__).resolve().parent
    main_file = getattr(sys.modules.get("__main__"), "__file__", None)
    runtime_dir = Path(main_file).resolve().parent if main_file else Path.cwd().resolve()
    cwd_dir = Path.cwd().resolve()

    search_roots = []
    for root in (runtime_dir, runtime_dir.parent, module_dir, module_dir.parent, cwd_dir, cwd_dir.parent):
        if root not in search_roots:
            search_roots.append(root)

    candidates = [root / path for root in search_roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return runtime_dir / path


def push_layout_to_grid(pos, size):
    dist_mat = pdist(pos)
    scale = min(1.0 / dist_mat.min(), (size - 2) / (pos.max() - pos.min()))
    pos_grid = np.round(pos * scale).astype(int)
    pos_grid = pos_grid - pos_grid.min(axis=0) + [1, 1]

    mask = np.zeros((size, size), dtype=int)
    for point in pos_grid:
        mask[point[0], point[1]] += 1

    for _ in range(50):
        unique_pos, counts = np.unique(pos_grid, axis=0, return_counts=True)
        if counts.max() <= 1:
            break

        for overlap in unique_pos[counts > 1]:
            idx = np.argmin(cdist(pos_grid, [overlap]))
            row_down = max(overlap[0] - 1, 0)
            row_up = min(overlap[0] + 2, size)
            col_left = max(overlap[1] - 1, 0)
            col_right = min(overlap[1] + 2, size)
            nearby = mask[row_down:row_up, col_left:col_right]

            if nearby.min() == 0:
                target = np.unravel_index(np.argmin(nearby), nearby.shape)
                next_pos = target + np.array([row_down, col_left])
            else:
                empty_pos = np.argwhere(mask == 0)
                target = empty_pos[np.argmin(cdist(empty_pos, [overlap]))]
                direction = target - overlap
                next_pos = overlap + np.round(direction / np.linalg.norm(direction)).astype(int)

            pos_grid[idx] = next_pos
            mask[overlap[0], overlap[1]] -= 1
            mask[next_pos[0], next_pos[1]] += 1

    return pos_grid.astype(np.int64)


def build_uav_prior():
    feature_to_idx = {name: idx for idx, name in enumerate(FEATURE_COLUMNS)}
    prior = np.zeros((len(FEATURE_COLUMNS), len(FEATURE_COLUMNS)), dtype=np.float32)

    def add_edge(a, b, weight):
        i = feature_to_idx[a]
        j = feature_to_idx[b]
        prior[i, j] = weight
        prior[j, i] = weight

    add_edge("phi", "Gz", 0.9)
    add_edge("theta", "C2", 0.8)
    add_edge("Ax", "Az", 0.7)
    add_edge("C1", "C2", 0.7)
    add_edge("Gx", "C1", 0.6)
    add_edge("theta", "C1", 0.6)
    add_edge("Az", "Gy", 0.4)
    add_edge("theta", "Gx", 0.4)
    add_edge("Gx", "C2", 0.4)
    add_edge("Ay", "Az", 0.3)
    add_edge("airspeed", "C2", 0.3)
    add_edge("airspeed", "Gy", 0.2)
    add_edge("phi", "Ay", 0.2)
    add_edge("Ay", "Gz", 0.2)
    add_edge("airspeed", "C1", 0.2)

    np.fill_diagonal(prior, 0.0)
    return prior


def normalize_graph_mode(graph_mode):
    key = str(graph_mode).strip().lower()
    if key not in GRAPH_MODE_ALIASES:
        choices = ", ".join(sorted(GRAPH_MODE_ALIASES))
        raise ValueError(f"Unsupported graph_mode: {graph_mode}. Available aliases: {choices}")
    return GRAPH_MODE_ALIASES[key]


def symmetrize_adjacency(A):
    A = np.asarray(A, dtype=np.float32)
    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    return A


def prune_adjacency_top_k(A, top_k):
    A = symmetrize_adjacency(A)
    if top_k is None:
        return A

    top_k = int(top_k)
    if top_k <= 0:
        return np.zeros_like(A)

    n_nodes = A.shape[0]
    total_edges = int(n_nodes * (n_nodes - 1) / 2)
    if top_k >= total_edges:
        return A

    upper_rows, upper_cols = np.triu_indices(n_nodes, k=1)
    upper_values = A[upper_rows, upper_cols]
    selected_idx = np.argsort(upper_values)[::-1][:top_k]

    pruned = np.zeros_like(A)
    rows = upper_rows[selected_idx]
    cols = upper_cols[selected_idx]
    pruned[rows, cols] = A[rows, cols]
    pruned[cols, rows] = A[rows, cols]
    return pruned


def compute_pearson_adjacency(X_windows):
    X_windows = np.asarray(X_windows, dtype=np.float32)
    if X_windows.ndim != 3:
        raise ValueError(f"Expected X_windows with shape [N, T, F], got {tuple(X_windows.shape)}")

    flattened = X_windows.reshape(-1, X_windows.shape[-1])
    if flattened.shape[0] < 2:
        return np.zeros((X_windows.shape[-1], X_windows.shape[-1]), dtype=np.float32)

    corr = np.corrcoef(flattened, rowvar=False)
    corr = np.abs(np.asarray(corr, dtype=np.float32))
    return symmetrize_adjacency(corr)


def compute_random_adjacency(n_nodes, seed):
    rng = np.random.default_rng(int(seed))
    upper_rows, upper_cols = np.triu_indices(int(n_nodes), k=1)
    upper_values = rng.random(upper_rows.size).astype(np.float32)

    A = np.zeros((int(n_nodes), int(n_nodes)), dtype=np.float32)
    A[upper_rows, upper_cols] = upper_values
    A[upper_cols, upper_rows] = upper_values
    return A


def load_cached_adjacency(cache_path):
    if not cache_path or not os.path.exists(cache_path):
        return None

    cache = np.load(cache_path)
    if "A" not in cache:
        raise ValueError(f"Graph cache at {cache_path} does not contain key 'A'.")
    A = symmetrize_adjacency(cache["A"])
    summarize_graph(A)
    return A


def save_graph_cache(cache_path, A, gpgl_coords=None):
    if not cache_path:
        return

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {"A": symmetrize_adjacency(A)}
    if gpgl_coords is not None:
        payload["gpgl_coords"] = np.asarray(gpgl_coords, dtype=np.int64)
    np.savez(str(cache_path), **payload)


class GraphPrior(nn.Module):
    def __init__(self, n_nodes, device, k=None, beta=0.8, bias=2.0, init_std=0.05):
        super().__init__()
        self.k = k
        self.beta = beta
        self.bias = bias
        self.W = nn.Parameter(torch.randn(n_nodes, n_nodes, device=device) * init_std)
        self.register_buffer("P", torch.tensor(build_uav_prior(), dtype=torch.float32, device=device))

    def symmetric_topk_mask(self, adj):
        if not self.k or self.k <= 0:
            return torch.ones_like(adj)

        k = min(self.k, adj.size(1))
        mask = torch.zeros_like(adj)
        _, indices = (adj + torch.rand_like(adj) * 0.01).topk(k, dim=1)
        mask.scatter_(1, indices, 1.0)
        mask = ((mask + mask.t()) > 0).float()
        return mask - torch.diag(torch.diag(mask))

    def forward(self):
        adj = 0.5 * (self.W + self.W.t())
        adj = F.softplus(adj + self.beta * self.P - self.bias)
        adj = adj - torch.diag(torch.diag(adj))

        if self.k and self.k > 0:
            adj = adj * self.symmetric_topk_mask(adj)
            adj = 0.5 * (adj + adj.t())
            adj = adj - torch.diag(torch.diag(adj))

        return adj


class GCLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, adj, X):
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        hidden = self.linear(X)
        norm = adj.sum(1).pow(-0.5)
        return norm[None, :] * adj * norm[:, None] @ hidden


class GPGLGraphLearner(nn.Module):
    def __init__(
        self,
        n_nodes,
        window_size,
        n_hidden=1024,
        k=None,
        beta=0.8,
        bias=2.0,
        init_std=0.05,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.graph = GraphPrior(
            n_nodes=n_nodes,
            device=self.device,
            k=k,
            beta=beta,
            bias=bias,
            init_std=init_std,
        )
        self.register_buffer("z", torch.ones(n_nodes, n_nodes, device=self.device) - torch.eye(n_nodes, device=self.device))
        self.conv1 = GCLayer(window_size, n_hidden)
        self.bn1 = nn.BatchNorm1d(n_nodes)
        self.conv2 = GCLayer(n_hidden, n_hidden)
        self.bn2 = nn.BatchNorm1d(n_nodes)
        self.fc = nn.Linear(n_hidden, 2)

    def adjacency(self):
        return self.graph() * self.z

    def forward(self, X):
        X = X.to(self.device)
        adj = self.adjacency()

        hidden = self.conv1(adj, X).relu()
        hidden = self.bn1(hidden)
        skip, _ = torch.min(hidden, dim=1)

        hidden = self.conv2(adj, hidden).relu()
        hidden = self.bn2(hidden)
        hidden, _ = torch.min(hidden, dim=1)

        return self.fc(hidden + skip)


def adjacency_to_layout(A, size=5):
    A = np.asarray(A, dtype=np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    if A.max() > 0:
        A = A / A.max()

    pos_dict = nx.spring_layout(nx.from_numpy_array(A), weight="weight", seed=42, dim=2)
    pos = np.array([pos_dict[i] for i in range(A.shape[0])], dtype=np.float32)
    pos = pos + 1e-6 * np.arange(len(pos), dtype=np.float32)[:, None]
    return push_layout_to_grid(pos.astype(np.float64), int(size))


def windows_to_gpgl_input(X, gpgl_coords, H=5, W=5):
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()

    X = np.asarray(X, dtype=np.float32)
    gpgl_coords = np.asarray(gpgl_coords, dtype=np.int64)
    if gpgl_coords.min() >= 1:
        gpgl_coords = gpgl_coords - 1

    grid = np.zeros((X.shape[0], X.shape[1], H, W), dtype=np.float32)
    for feature_idx, (row, col) in enumerate(gpgl_coords):
        grid[:, :, row, col] = X[:, :, feature_idx]

    mask = np.zeros((H, W), dtype=np.float32)
    mask[gpgl_coords[:, 0], gpgl_coords[:, 1]] = 1.0

    gpgl_input = np.zeros((X.shape[0], X.shape[1] + 1, H, W), dtype=np.float32)
    gpgl_input[:, :X.shape[1]] = grid
    gpgl_input[:, -1] = mask
    return gpgl_input


def to_gnn_windows(X):
    X = np.asarray(X, dtype=np.float32)
    return np.transpose(X, (0, 2, 1))


def summarize_graph(A, top_k_edges=5, active_threshold=1e-8):
    A = np.asarray(A, dtype=np.float32)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)

    n_nodes = A.shape[0]
    upper_mask = np.triu(np.ones_like(A, dtype=bool), k=1)
    upper_values = A[upper_mask]
    active_values = upper_values[upper_values > active_threshold]

    total_edges = int(n_nodes * (n_nodes - 1) / 2)
    active_edges = int(active_values.size)
    density = active_edges / total_edges if total_edges > 0 else 0.0

    if active_edges > 0:
        edge_mean = float(active_values.mean())
        edge_std = float(active_values.std())
        edge_min = float(active_values.min())
        edge_max = float(active_values.max())
    else:
        edge_mean = edge_std = edge_min = edge_max = 0.0

    degree = A.sum(axis=1)
    isolated_nodes = int(np.sum(degree <= active_threshold))
    symmetry_error = float(np.abs(A - A.T).max())
    spectral_radius = float(np.max(np.linalg.eigvalsh(A))) if n_nodes > 0 else 0.0

    graph = nx.from_numpy_array((A > active_threshold).astype(np.int32))
    connected_components = nx.number_connected_components(graph)

    edges = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if A[i, j] > active_threshold:
                edges.append((i, j, float(A[i, j])))
    edges.sort(key=lambda item: item[2], reverse=True)

    print("\n===== GPGL Graph A Summary =====")
    print(
        f"Nodes: {n_nodes} | Active edges: {active_edges}/{total_edges} | "
        f"Density: {density:.4f} | Connected components: {connected_components}"
    )
    print(
        f"Edge weights (active only): mean={edge_mean:.4f}, std={edge_std:.4f}, "
        f"min={edge_min:.4f}, max={edge_max:.4f}"
    )
    print(
        f"Node degree: mean={degree.mean():.4f}, std={degree.std():.4f}, "
        f"min={degree.min():.4f}, max={degree.max():.4f}, isolated={isolated_nodes}"
    )
    print(f"Symmetry error(max|A-A^T|): {symmetry_error:.6f} | Spectral radius: {spectral_radius:.4f}")
    if edges:
        print("Top edges:")
        for idx, (i, j, weight) in enumerate(edges[:top_k_edges], start=1):
            print(f"  {idx}. ({i}, {j}) -> {weight:.4f}")
    else:
        print("Top edges: none")


def fit_gpgl_adjacency(
    X_train,
    y_train,
    cache_path=None,
    force_retrain=False,
    window_size=20,
    n_epochs=40,
    batch_size=512,
    n_hidden=1024,
    k=None,
    lr=0.001,
    beta=0.8,
    bias=2.0,
    init_std=0.05,
    num_workers=0,
    device=None,
):
    if cache_path and os.path.exists(cache_path) and not force_retrain:
        print(f"Loading cached GPGL matrix from: {cache_path}")
        cached_A = load_cached_adjacency(cache_path)
        if cached_A is not None:
            return cached_A

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_gnn = to_gnn_windows(X_train)
    y_train = np.asarray(y_train, dtype=np.int64)
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_gnn), torch.from_numpy(y_train)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = GPGLGraphLearner(
        n_nodes=X_train_gnn.shape[1],
        window_size=window_size,
        n_hidden=n_hidden,
        k=k,
        beta=beta,
        bias=bias,
        init_std=init_std,
        device=device,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=lr)
    class_weight = torch.tensor([1, 1], dtype=torch.float32, device=device)

    model.train()
    for _ in range(n_epochs):
        for train_ts, train_label in train_loader:
            logits = model(train_ts)
            loss = F.cross_entropy(logits, train_label.to(device), weight=class_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    final_A = symmetrize_adjacency(model.adjacency().detach().cpu().numpy())
    summarize_graph(final_A)

    if cache_path:
        gpgl_coords = adjacency_to_layout(final_A, size=5)
        save_graph_cache(cache_path, final_A, gpgl_coords)
        print(f"Saved GPGL matrix to: {cache_path}")

    return final_A


def fit_gpgl_layout(
    X_train,
    y_train,
    cache_path=None,
    force_retrain=False,
    window_size=20,
    n_epochs=40,
    batch_size=512,
    n_hidden=1024,
    k=None,
    lr=0.001,
    beta=0.8,
    bias=2.0,
    init_std=0.05,
    num_workers=0,
    device=None,
):
    A = fit_gpgl_adjacency(
        X_train=X_train,
        y_train=y_train,
        cache_path=cache_path,
        force_retrain=force_retrain,
        window_size=window_size,
        n_epochs=n_epochs,
        batch_size=batch_size,
        n_hidden=n_hidden,
        k=k,
        lr=lr,
        beta=beta,
        bias=bias,
        init_std=init_std,
        num_workers=num_workers,
        device=device,
    )
    return adjacency_to_layout(A, size=5)


def build_gpgl_sets(
    X_train,
    y_train,
    X_val,
    X_test,
    gpgl_coords=None,
    cache_path=None,
    force_retrain=False,
    window_size=20,
    n_epochs=40,
    batch_size=512,
    n_hidden=1024,
    k=None,
    lr=0.001,
    beta=0.8,
    bias=2.0,
    init_std=0.05,
    num_workers=0,
    device=None,
):
    if gpgl_coords is None:
        gpgl_coords = fit_gpgl_layout(
            X_train=X_train,
            y_train=y_train,
            cache_path=cache_path,
            force_retrain=force_retrain,
            window_size=window_size,
            n_epochs=n_epochs,
            batch_size=batch_size,
            n_hidden=n_hidden,
            k=k,
            lr=lr,
            beta=beta,
            bias=bias,
            init_std=init_std,
            num_workers=num_workers,
            device=device,
        )

    return (
        windows_to_gpgl_input(X_train, gpgl_coords),
        windows_to_gpgl_input(X_val, gpgl_coords),
        windows_to_gpgl_input(X_test, gpgl_coords),
        gpgl_coords,
    )


def build_graph_layout_for_config(X_gpgl_raw, y_gpgl, cfg, runtime_cache_path):
    graph_mode = normalize_graph_mode(getattr(cfg, "graph_mode", "ptcnet-full"))
    source_cache_path = getattr(cfg, "graph_source_cache_path", None) or runtime_cache_path
    graph_top_k = getattr(cfg, "graph_top_k", None)

    print(
        f"Graph construction mode: {graph_mode} | "
        f"top_k={graph_top_k if graph_top_k is not None else 'full'}"
    )

    X_gpgl_scaled = scale_windows(X_gpgl_raw)[0]

    if graph_mode == "ptcnet-full":
        base_A = fit_gpgl_adjacency(
            X_train=X_gpgl_scaled,
            y_train=y_gpgl,
            cache_path=str(source_cache_path),
            force_retrain=bool(cfg.gpgl_force_retrain),
            window_size=int(cfg.n_step),
            n_epochs=int(cfg.gpgl_pretrain_epochs),
            batch_size=int(cfg.gpgl_batch_size),
            n_hidden=int(cfg.gpgl_n_hidden),
            k=cfg.gpgl_k,
            lr=float(cfg.gpgl_lr),
            beta=float(cfg.gpgl_beta),
            bias=float(cfg.gpgl_bias),
            init_std=float(cfg.gpgl_init_std),
            num_workers=cfg.num_workers,
        )
    elif graph_mode == "w.o.p":
        base_A = fit_gpgl_adjacency(
            X_train=X_gpgl_scaled,
            y_train=y_gpgl,
            cache_path=str(source_cache_path),
            force_retrain=bool(cfg.gpgl_force_retrain),
            window_size=int(cfg.n_step),
            n_epochs=int(cfg.gpgl_pretrain_epochs),
            batch_size=int(cfg.gpgl_batch_size),
            n_hidden=int(cfg.gpgl_n_hidden),
            k=cfg.gpgl_k,
            lr=float(cfg.gpgl_lr),
            beta=0.0,
            bias=float(cfg.gpgl_bias),
            init_std=float(cfg.gpgl_init_std),
            num_workers=cfg.num_workers,
        )
    elif graph_mode == "correlation-graph":
        base_A = None if bool(cfg.gpgl_force_retrain) else load_cached_adjacency(source_cache_path)
        if base_A is None:
            base_A = compute_pearson_adjacency(X_gpgl_raw)
            summarize_graph(base_A)
            save_graph_cache(source_cache_path, base_A)
            print(f"Saved correlation graph to: {source_cache_path}")
    elif graph_mode == "random-graph":
        base_A = None if bool(cfg.gpgl_force_retrain) else load_cached_adjacency(source_cache_path)
        if base_A is None:
            base_A = compute_random_adjacency(X_gpgl_raw.shape[-1], seed=cfg.seed)
            summarize_graph(base_A)
            save_graph_cache(source_cache_path, base_A)
            print(f"Saved random graph to: {source_cache_path}")
    else:
        raise ValueError(f"Unsupported graph_mode: {graph_mode}")

    final_A = prune_adjacency_top_k(base_A, graph_top_k)
    print("\n===== Active Graph Summary After top-k Selection =====")
    summarize_graph(final_A, top_k_edges=min(int(graph_top_k or 5), 10))

    gpgl_coords = adjacency_to_layout(final_A, size=5)
    save_graph_cache(runtime_cache_path, final_A, gpgl_coords)
    print(f"Saved runtime graph to: {runtime_cache_path}")
    return gpgl_coords


def load_dataloaders(cfg):
    data_path = resolve_task_path(cfg.data_path)
    gpgl_cache_path = resolve_task_path(cfg.gpgl_cache_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Configured path: {cfg.data_path}\n"
            f"dataset.py location: {Path(__file__).resolve()}"
        )

    raw_data = DATA(str(data_path), cfg.aircraft_id, data_type="flight", sample_period=0.1)
    frame = raw_data.get_labelled_data()
    frame = frame.loc[cfg.segment_start:cfg.segment_end].copy()
    frame = add_fault_label(frame)

    X_raw = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_raw = frame["fault"].to_numpy(dtype=np.int64)
    X_all, y_all = build_time_windows(X_raw, y_raw, n_step=int(cfg.n_step))

    split_seed = cfg.seed if cfg.data_split_seed is None else int(cfg.data_split_seed)

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
        train_size=cfg.sample_size,
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

    gpgl_coords = build_graph_layout_for_config(
        X_gpgl_raw=X_gpgl_raw,
        y_gpgl=y_gpgl,
        cfg=cfg,
        runtime_cache_path=str(gpgl_cache_path),
    )

    X_train, X_val, X_test = scale_windows(X_train_raw, X_val_raw, X_test_raw)
    y_train = y_train_raw.astype(np.int64)
    y_val = y_val_raw.astype(np.int64)
    y_test = y_test_raw.astype(np.int64)

    X_train_gpgl, X_val_gpgl, X_test_gpgl, _ = build_gpgl_sets(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        X_test=X_test,
        gpgl_coords=gpgl_coords,
        window_size=int(cfg.n_step),
        n_epochs=int(cfg.gpgl_pretrain_epochs),
        batch_size=int(cfg.gpgl_batch_size),
        n_hidden=int(cfg.gpgl_n_hidden),
        k=cfg.gpgl_k,
        lr=float(cfg.gpgl_lr),
        beta=float(cfg.gpgl_beta),
        bias=float(cfg.gpgl_bias),
        init_std=float(cfg.gpgl_init_std),
        num_workers=cfg.num_workers,
    )

    train_dataset = DualInputDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(X_train_gpgl),
        torch.LongTensor(y_train),
    )
    val_dataset = DualInputDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(X_val_gpgl),
        torch.LongTensor(y_val),
    )
    test_dataset = DualInputDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(X_test_gpgl),
        torch.LongTensor(y_test),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
