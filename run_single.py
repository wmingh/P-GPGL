try:
    from .trainer import TrainingConfig, print_training_config, run_training
except ImportError:
    from trainer import TrainingConfig, print_training_config, run_training


def main():
    cfg = TrainingConfig()
    metrics = run_training(cfg)
    print_training_config(cfg)
    return metrics


if __name__ == "__main__":
    main()
