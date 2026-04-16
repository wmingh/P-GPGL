import random
import statistics
import time

try:
    from .trainer import TrainingConfig, clone_config, print_training_config, run_training
except ImportError:
    from trainer import TrainingConfig, clone_config, print_training_config, run_training


def generate_random_seeds(num_seeds, low=1, high=2**31 - 1):
    rng = random.SystemRandom()
    return rng.sample(range(low, high), num_seeds)


def run_single_seed(base_cfg, seed, run_index, total_runs):
    cfg = clone_config(base_cfg)
    cfg.seed = int(seed)

    print(f"\n{'=' * 20} Run {run_index}/{total_runs} | seed={cfg.seed} {'=' * 20}")
    metrics = run_training(cfg)
    metrics["seed"] = cfg.seed
    return metrics


def summarize_results(results):
    metric_names = [
        "best_val_acc",
        "best_val_loss",
        "test_acc",
        "test_precision",
        "test_recall",
        "test_f1",
        "time_minutes",
    ]

    print("\n===== Per-Seed Results =====")
    for item in results:
        print(
            f"seed={item['seed']} | "
            f"best_val_acc={item['best_val_acc'] * 100:.2f}% | "
            f"best_val_loss={item['best_val_loss']:.4f} | "
            f"test_acc={item['test_acc'] * 100:.2f}% | "
            f"test_precision={item['test_precision'] * 100:.2f}% | "
            f"test_recall={item['test_recall'] * 100:.2f}% | "
            f"test_f1={item['test_f1'] * 100:.2f}%"
        )

    print(f"\n===== {len(results)}-Seed Average =====")
    best_test_item = max(results, key=lambda item: item["test_acc"])
    print(f"best_acc={best_test_item['test_acc'] * 100:.2f}% (seed={best_test_item['seed']}, from test_acc)")

    for name in metric_names:
        values = [item[name] for item in results]
        mean_value = statistics.mean(values)
        std_value = statistics.pstdev(values) if len(values) > 1 else 0.0
        if any(token in name for token in ["acc", "precision", "recall", "f1"]):
            print(f"{name}: mean={mean_value * 100:.2f}%, std={std_value * 100:.2f}%")
        else:
            print(f"{name}: mean={mean_value:.4f}, std={std_value:.4f}")


def main():
    cfg = TrainingConfig()
    cfg.data_split_seed = 42 if cfg.data_split_seed is None else cfg.data_split_seed

    seeds = generate_random_seeds(cfg.num_random_seeds)
    print(f"Random seeds for this run: {seeds}")

    display_cfg = clone_config(cfg)
    display_cfg.seed = "generated_per_run"
    print_training_config(display_cfg)

    results = []
    start_time = time.time()
    for run_index, seed in enumerate(seeds, start=1):
        results.append(run_single_seed(cfg, seed, run_index, cfg.num_random_seeds))

    summarize_results(results)
    total_minutes = (time.time() - start_time) / 60.0
    print(f"\nAll runs finished. Total wall-clock time: {total_minutes:.2f} mins")


if __name__ == "__main__":
    main()
