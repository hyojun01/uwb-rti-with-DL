"""
Fixed experiment runner for UWB RTI autoresearch.
Runs the full pipeline: MLP train → CFP data gen → CFP train → Evaluate.
DO NOT MODIFY THIS FILE.
"""
import os
import sys
import time
import json
import traceback

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from uwb_rti.config import DEVICE, RANDOM_SEED

TIME_BUDGET_SECONDS = 900  # 15 minutes max for entire experiment


def run_experiment():
    t_start = time.time()
    results = {
        "status": "running",
        "mlp_train_loss": None,
        "mlp_val_loss": None,
        "cfp_train_loss": None,
        "cfp_val_loss": None,
        "test_rmse": None,
        "test_ssim": None,
        "cfp_test_rmse": None,
        "cfp_test_ssim": None,
        "peak_vram_mb": None,
        "total_seconds": None,
        "error": None,
    }

    try:
        import torch
        print(f"Device: {DEVICE}")
        print(f"Random seed: {RANDOM_SEED}")

        # Step 0: Ensure data exists
        data_dir = "data"
        if not os.path.exists(os.path.join(data_dir, "mlp_data.npz")):
            print("Data not found. Generating datasets...")
            from uwb_rti.data_generator import generate_and_save_all_datasets
            generate_and_save_all_datasets(data_dir)

        # Step 1: Train MLP
        print("\n" + "=" * 50)
        print("STAGE 1: MLP Training")
        print("=" * 50)
        from uwb_rti.train import train_mlp, generate_cfp_training_data, train_cfp
        mlp_model, mlp_history = train_mlp(data_dir=data_dir)

        results["mlp_train_loss"] = mlp_history["train_loss"][-1]
        results["mlp_val_loss"] = min(mlp_history["val_loss"])

        elapsed = time.time() - t_start
        print(f"MLP training done in {elapsed:.1f}s")

        if elapsed > TIME_BUDGET_SECONDS:
            results["status"] = "timeout"
            results["error"] = "MLP training exceeded time budget"
            _print_results(results)
            return results

        # Step 2: Generate CFP training data
        print("\n" + "=" * 50)
        print("STAGE 2: CFP Data Generation")
        print("=" * 50)
        generate_cfp_training_data(mlp_model, data_dir=data_dir)

        # Step 3: Train CFP
        print("\n" + "=" * 50)
        print("STAGE 3: CFP Training")
        print("=" * 50)
        cfp_model, cfp_history = train_cfp(data_dir=data_dir)

        results["cfp_train_loss"] = cfp_history["train_loss"][-1]
        results["cfp_val_loss"] = min(cfp_history["val_loss"])

        elapsed = time.time() - t_start
        print(f"CFP training done in {elapsed:.1f}s")

        if elapsed > TIME_BUDGET_SECONDS:
            results["status"] = "timeout"
            results["error"] = "Total training exceeded time budget"
            _print_results(results)
            return results

        # Step 4: Evaluate
        print("\n" + "=" * 50)
        print("STAGE 4: Evaluation")
        print("=" * 50)
        from uwb_rti.evaluate import evaluate_on_test_set
        eval_results, mlp_out, cfp_out, theta_ideal = evaluate_on_test_set(
            mlp_model, cfp_model, data_dir=data_dir
        )

        results["test_rmse"] = eval_results["mlp_rmse"]
        results["test_ssim"] = eval_results["mlp_ssim"]
        if "cfp_rmse" in eval_results:
            results["cfp_test_rmse"] = eval_results["cfp_rmse"]
            results["cfp_test_ssim"] = eval_results["cfp_ssim"]

        # Record VRAM usage
        if torch.cuda.is_available():
            results["peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

        results["status"] = "success"

    except Exception as e:
        results["status"] = "crash"
        results["error"] = f"{type(e).__name__}: {str(e)}"
        traceback.print_exc()

    results["total_seconds"] = round(time.time() - t_start, 1)
    _print_results(results)
    return results


def _print_results(results):
    """Print results in a grep-friendly format."""
    print("\n---")
    for key, value in results.items():
        if value is not None:
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")


if __name__ == "__main__":
    run_experiment()
