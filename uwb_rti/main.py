import argparse
import os
import numpy as np
import torch

from .config import DEVICE, RANDOM_SEED, GRID_SIZE
from .forward_model import compute_weight_matrix
from .validate_model import run_all_validations
from .data_generator import generate_and_save_all_datasets
from .train import train_mlp, generate_cfp_training_data, train_cfp
from .evaluate import evaluate_on_test_set, evaluate_across_noise_levels
from .visualize import (
    plot_weight_matrix, plot_training_curves,
    plot_reconstruction_grid, plot_error_maps,
    plot_noise_comparison_table,
)
from .models.mlp_model import MLPModel
from .models.cfp_model import CFPModel


def load_models(checkpoint_dir="checkpoints"):
    """Load trained MLP and CFP models from checkpoints."""
    mlp_model = MLPModel().to(DEVICE)
    mlp_path = os.path.join(checkpoint_dir, "mlp_best.pt")
    mlp_model.load_state_dict(torch.load(mlp_path, map_location=DEVICE, weights_only=True))
    mlp_model.eval()

    cfp_model = CFPModel().to(DEVICE)
    cfp_path = os.path.join(checkpoint_dir, "cfp_best.pt")
    cfp_model.load_state_dict(torch.load(cfp_path, map_location=DEVICE, weights_only=True))
    cfp_model.eval()

    return mlp_model, cfp_model


def stage_validate():
    """Stage 1: Validate forward model."""
    print("\n" + "=" * 60)
    print("STAGE: Forward Model Validation")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    W = compute_weight_matrix()
    plot_weight_matrix(W, [0, 5, 10, 15])
    run_all_validations()


def stage_generate():
    """Stage 2: Generate all datasets."""
    print("\n" + "=" * 60)
    print("STAGE: Data Generation")
    print("=" * 60)
    generate_and_save_all_datasets()


def stage_train_mlp():
    """Stage 3: Train MLP model."""
    print("\n" + "=" * 60)
    print("STAGE: MLP Training")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    mlp_model, mlp_history = train_mlp()
    plot_training_curves(mlp_history, "MLP Training", "figures/mlp_training.png")

    generate_cfp_training_data(mlp_model)


def stage_train_cfp():
    """Stage 4: Train CFP model."""
    print("\n" + "=" * 60)
    print("STAGE: CFP Training")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    cfp_model, cfp_history = train_cfp()
    plot_training_curves(cfp_history, "CFP Training", "figures/cfp_training.png")


def stage_evaluate():
    """Stage 5: Evaluate and visualize results."""
    print("\n" + "=" * 60)
    print("STAGE: Evaluation")
    print("=" * 60)

    os.makedirs("figures", exist_ok=True)
    mlp_model, cfp_model = load_models()

    results, mlp_out, cfp_out, theta_ideal = evaluate_on_test_set(mlp_model, cfp_model)

    plot_reconstruction_grid(
        theta_ideal, mlp_out, cfp_out, num_samples=5,
        save_path="figures/reconstruction_grid.png",
    )
    plot_error_maps(
        theta_ideal, mlp_out, cfp_out, num_samples=5,
        save_path="figures/error_maps.png",
    )

    noise_results = evaluate_across_noise_levels(mlp_model, cfp_model)
    plot_noise_comparison_table(noise_results)


def main():
    parser = argparse.ArgumentParser(description="UWB RTI Deep Learning Pipeline")
    parser.add_argument(
        "--stage",
        choices=["all", "validate", "generate", "train_mlp", "train_cfp", "evaluate"],
        default="all",
        help="Pipeline stage to run",
    )
    args = parser.parse_args()

    print(f"Using device: {DEVICE}")
    print(f"Random seed: {RANDOM_SEED}")

    stages = {
        "validate": stage_validate,
        "generate": stage_generate,
        "train_mlp": stage_train_mlp,
        "train_cfp": stage_train_cfp,
        "evaluate": stage_evaluate,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            fn()
    else:
        stages[args.stage]()

    print("\nDone.")


if __name__ == "__main__":
    main()
