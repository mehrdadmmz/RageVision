"""Run all hyperparameter experiments sequentially."""

from experiments.learning_rate import run as run_lr
from experiments.dropout_rate import run as run_dropout
from experiments.batch_size import run as run_batch_size
from experiments.augmentation import run as run_augmentation
from experiments.freezing import run as run_freezing
from experiments.optimizer import run as run_optimizer


def main():
    print("=" * 60)
    print("  RageVision - Hyperparameter Experiments")
    print("=" * 60)

    run_lr()
    run_dropout()
    run_batch_size()
    run_augmentation()
    run_freezing()
    run_optimizer()

    print("\n" + "=" * 60)
    print("  All experiments completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
