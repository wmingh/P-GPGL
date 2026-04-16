import argparse
import csv
import json
import statistics
import time
from dataclasses import asdict
from pathlib import Path

try:
    from .trainer import TrainingConfig, clone_config, print_training_config, run_training
except ImportError:
    from trainer import TrainingConfig, clone_config, print_training_config, run_training


DEFAULT_GRAPH_MODES = ["PTCNet-full", "w.o.P", "Correlation-graph", "Random-graph"]
DEFAULT_TOP_K = [5, 10, 15, 20]
DEFAULT_SEEDS = [42, 52, 62]

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

GRAPH_MODE_LABELS = {
    "ptcnet-full": "PTCNet-full",
    "w.o.p": "w.o.P",
    "correlation-graph": "Correlation-graph",
    "random-graph": "Random-graph",
}

GRAPH_MODE_DIRS = {
    "ptcnet-full": "ptcnet_full",
    "w.o.p": "wo_p",
    "correlation-graph": "correlation_graph",
    "random-graph": "random_graph",
}


def normalize_graph_mode(graph_mode):
    key = str(graph_mode).strip().lower()
    if key not in GRAPH_MODE_ALIASES:
        choices = ", ".join(DEFAULT_GRAPH_MODES)
        raise ValueError(f"Unsupported graph mode: {graph_mode}. Supported values: {choices}")
    return GRAPH_MODE_ALIASES[key]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run topology-budget experiments for PTCNet graph-source comparisons."
    )
    parser.add_argument("--graph-modes", nargs="+", default=DEFAULT_GRAPH_MODES, help="Subset of graph modes to run.")
    parser.add_argument("--top-k", nargs="+", type=int, default=DEFAULT_TOP_K, help="top-k edge budgets to evaluate.")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS, help="Training seeds to evaluate.")
    parser.add_argument("--sample-size", type=int, default=360, help="TrainingConfig.sample_size value.")
    parser.add_argument("--data-split-seed", type=int, default=42, help="Fixed data split seed shared across runs.")
    parser.add_argument("--results-root", default="results/topology_budget", help="Root directory for experiment outputs.")
    parser.add_argument("--device", default=None, help="Override TrainingConfig.device.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override TrainingConfig.num_workers.")
    parser.add_argument(
        "--gpgl-pretrain-epochs",
        type=int,
        default=None,
        help="Override TrainingConfig.gpgl_pretrain_epochs.",
    )
    parser.add_argument(
        "--temporal-pretrain-epochs",
        type=int,
        default=None,
        help="Override TrainingConfig.temporal_pretrain_epochs.",
    )
    parser.add_argument("--fusion-epochs", type=int, default=None, help="Override TrainingConfig.fusion_epochs.")
    parser.add_argument(
        "--force-retrain-graph",
        action="store_true",
        help="Ignore cached source graphs and relearn/rebuild them.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse saved per-run metrics if the target run directory already contains test_metrics.json.",
    )
    return parser


def ensure_parent(path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_json(path, payload):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(path, rows, fieldnames):
    ensure_parent(path)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_base_config(args, seeds):
    cfg = TrainingConfig()
    cfg.sample_size = int(args.sample_size)
    cfg.data_split_seed = int(args.data_split_seed)
    cfg.num_random_seeds = len(seeds)
    cfg.gpgl_k = 0
    cfg.gpgl_force_retrain = bool(args.force_retrain_graph)

    if args.device is not None:
        cfg.device = args.device
    if args.num_workers is not None:
        cfg.num_workers = int(args.num_workers)
    if args.gpgl_pretrain_epochs is not None:
        cfg.gpgl_pretrain_epochs = int(args.gpgl_pretrain_epochs)
    if args.temporal_pretrain_epochs is not None:
        cfg.temporal_pretrain_epochs = int(args.temporal_pretrain_epochs)
    if args.fusion_epochs is not None:
        cfg.fusion_epochs = int(args.fusion_epochs)

    return cfg


def build_experiment_root(args):
    return (
        Path(args.results_root)
        / f"sample_{int(args.sample_size)}"
        / f"split_{int(args.data_split_seed)}"
    )


def prepare_run_config(base_cfg, experiment_root, graph_mode, top_k, seed):
    cfg = clone_config(base_cfg)
    cfg.seed = int(seed)
    cfg.graph_mode = graph_mode
    cfg.graph_top_k = int(top_k)

    graph_dir = GRAPH_MODE_DIRS[graph_mode]
    run_dir = experiment_root / graph_dir / f"k_{int(top_k)}" / f"seed_{int(seed)}"
    source_dir = experiment_root / "_graph_cache" / graph_dir / f"seed_{int(seed)}"

    cfg.graph_source_cache_path = str(source_dir / "A_source.npz")
    cfg.gpgl_cache_path = str(run_dir / "A.npz")
    cfg.temporal_save_path = str(run_dir / "PINN.pth")
    cfg.save_path = str(run_dir / "MODEL.pth")
    return cfg, run_dir


def collect_test_result(metrics, graph_mode, top_k, seed):
    return {
        "graph_mode": graph_mode,
        "graph_label": GRAPH_MODE_LABELS[graph_mode],
        "top_k": int(top_k),
        "seed": int(seed),
        "test_acc": float(metrics["test_acc"]),
        "test_precision": float(metrics["test_precision"]),
        "test_recall": float(metrics["test_recall"]),
        "test_f1": float(metrics["test_f1"]),
        "time_minutes": float(metrics["time_minutes"]),
    }


def run_single_combo(base_cfg, experiment_root, graph_mode, top_k, seed, skip_existing=False):
    cfg, run_dir = prepare_run_config(base_cfg, experiment_root, graph_mode, top_k, seed)
    metrics_path = run_dir / "test_metrics.json"
    config_path = run_dir / "config.json"

    print(
        f"\n{'=' * 18} {GRAPH_MODE_LABELS[graph_mode]} | "
        f"top-k={int(top_k)} | seed={int(seed)} {'=' * 18}"
    )

    if skip_existing and metrics_path.exists():
        print(f"Skipping existing run and reusing metrics from: {metrics_path}")
        return load_json(metrics_path)

    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_training(cfg)
    result = collect_test_result(metrics, graph_mode, top_k, seed)

    save_json(config_path, asdict(cfg))
    save_json(metrics_path, result)
    return result


def load_saved_results(experiment_root):
    rows = []
    for metrics_path in sorted(Path(experiment_root).rglob("test_metrics.json")):
        rows.append(load_json(metrics_path))
    return rows


def deduplicate_rows(rows):
    deduped = {}
    for row in rows:
        key = (row["graph_mode"], int(row["top_k"]), int(row["seed"]))
        deduped[key] = row
    return [deduped[key] for key in sorted(deduped)]


def mean_std(values):
    values = list(values)
    mean_value = statistics.mean(values)
    std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean_value, std_value


def summarize_results(rows, graph_modes, top_k_values):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["graph_mode"], int(row["top_k"])), []).append(row)

    summary_rows = []
    for graph_mode in graph_modes:
        for top_k in top_k_values:
            key = (graph_mode, int(top_k))
            if key not in grouped:
                continue

            group = grouped[key]
            acc_mean, acc_std = mean_std(item["test_acc"] for item in group)
            precision_mean, precision_std = mean_std(item["test_precision"] for item in group)
            recall_mean, recall_std = mean_std(item["test_recall"] for item in group)
            f1_mean, f1_std = mean_std(item["test_f1"] for item in group)

            summary_rows.append(
                {
                    "graph_mode": graph_mode,
                    "graph_label": GRAPH_MODE_LABELS[graph_mode],
                    "top_k": int(top_k),
                    "acc_mean": acc_mean,
                    "acc_std": acc_std,
                    "precision_mean": precision_mean,
                    "precision_std": precision_std,
                    "recall_mean": recall_mean,
                    "recall_std": recall_std,
                    "f1_mean": f1_mean,
                    "f1_std": f1_std,
                }
            )
    return summary_rows


def format_summary_text(summary_rows, graph_modes):
    grouped = {}
    for row in summary_rows:
        grouped.setdefault(row["graph_mode"], []).append(row)

    lines = []
    lines.append("===== Topology Budget Summary (test set, mean +- std over seeds) =====")
    for graph_mode in graph_modes:
        if graph_mode not in grouped:
            continue
        lines.append(f"[{GRAPH_MODE_LABELS[graph_mode]}]")
        for row in sorted(grouped[graph_mode], key=lambda item: item["top_k"]):
            lines.append(
                "k={top_k} | ACC={acc_mean:.2f}%+-{acc_std:.2f}% | "
                "Precision={precision_mean:.2f}%+-{precision_std:.2f}% | "
                "Recall={recall_mean:.2f}%+-{recall_std:.2f}% | "
                "F1={f1_mean:.2f}%+-{f1_std:.2f}%".format(
                    top_k=row["top_k"],
                    acc_mean=row["acc_mean"] * 100.0,
                    acc_std=row["acc_std"] * 100.0,
                    precision_mean=row["precision_mean"] * 100.0,
                    precision_std=row["precision_std"] * 100.0,
                    recall_mean=row["recall_mean"] * 100.0,
                    recall_std=row["recall_std"] * 100.0,
                    f1_mean=row["f1_mean"] * 100.0,
                    f1_std=row["f1_std"] * 100.0,
                )
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    parser = build_parser()
    args = parser.parse_args()

    graph_modes = [normalize_graph_mode(item) for item in args.graph_modes]
    top_k_values = [int(item) for item in args.top_k]
    seeds = [int(item) for item in args.seeds]

    base_cfg = build_base_config(args, seeds)
    experiment_root = build_experiment_root(args)
    experiment_root.mkdir(parents=True, exist_ok=True)

    display_cfg = clone_config(base_cfg)
    display_cfg.seed = "varies_per_run"
    display_cfg.graph_mode = "varies_per_run"
    display_cfg.graph_top_k = "varies_per_run"
    display_cfg.graph_source_cache_path = str(experiment_root / "_graph_cache" / "<graph_mode>" / "<seed>" / "A_source.npz")
    display_cfg.gpgl_cache_path = str(experiment_root / "<graph_mode>" / "<k>" / "<seed>" / "A.npz")
    display_cfg.temporal_save_path = str(experiment_root / "<graph_mode>" / "<k>" / "<seed>" / "PINN.pth")
    display_cfg.save_path = str(experiment_root / "<graph_mode>" / "<k>" / "<seed>" / "MODEL.pth")

    print_training_config(display_cfg)
    print(f"Graph modes: {[GRAPH_MODE_LABELS[item] for item in graph_modes]}")
    print(f"top-k values: {top_k_values}")
    print(f"seeds: {seeds}")
    print(f"Results root: {experiment_root}")

    per_seed_rows = []
    start_time = time.time()
    total_runs = len(graph_modes) * len(top_k_values) * len(seeds)
    current_run = 0

    for graph_mode in graph_modes:
        for top_k in top_k_values:
            for seed in seeds:
                current_run += 1
                print(f"\nRun {current_run}/{total_runs}")
                per_seed_rows.append(
                    run_single_combo(
                        base_cfg=base_cfg,
                        experiment_root=experiment_root,
                        graph_mode=graph_mode,
                        top_k=top_k,
                        seed=seed,
                        skip_existing=bool(args.skip_existing),
                    )
                )

    all_rows = deduplicate_rows(load_saved_results(experiment_root))
    summary_graph_modes = [mode for mode in GRAPH_MODE_LABELS if any(row["graph_mode"] == mode for row in all_rows)]
    summary_top_k = sorted({int(row["top_k"]) for row in all_rows})
    summary_rows = summarize_results(all_rows, summary_graph_modes, summary_top_k)
    summary_text = format_summary_text(summary_rows, summary_graph_modes)
    print(f"\n{summary_text}")

    per_seed_csv_path = experiment_root / "per_seed_results.csv"
    summary_csv_path = experiment_root / "summary_mean_std.csv"
    summary_txt_path = experiment_root / "summary_mean_std.txt"

    write_csv(
        per_seed_csv_path,
        all_rows,
        [
            "graph_mode",
            "graph_label",
            "top_k",
            "seed",
            "test_acc",
            "test_precision",
            "test_recall",
            "test_f1",
            "time_minutes",
        ],
    )
    write_csv(
        summary_csv_path,
        summary_rows,
        [
            "graph_mode",
            "graph_label",
            "top_k",
            "acc_mean",
            "acc_std",
            "precision_mean",
            "precision_std",
            "recall_mean",
            "recall_std",
            "f1_mean",
            "f1_std",
        ],
    )
    ensure_parent(summary_txt_path)
    summary_txt_path.write_text(summary_text, encoding="utf-8")

    total_minutes = (time.time() - start_time) / 60.0
    print(f"Saved per-seed results to: {per_seed_csv_path}")
    print(f"Saved summary CSV to: {summary_csv_path}")
    print(f"Saved summary text to: {summary_txt_path}")
    print(f"All runs finished. Total wall-clock time: {total_minutes:.2f} mins")


if __name__ == "__main__":
    main()
