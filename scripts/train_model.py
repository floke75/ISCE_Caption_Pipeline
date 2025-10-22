# C:\dev\Captions_Formatter\Formatter_machine\scripts\train_model.py

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from isce.model_builder import build_weights, derive_constraints, create_feature_row
from isce.config import Config, load_config
from isce.scorer import Scorer
from isce.types import TokenRow, Engineered

# =========================================
# REFACTORED DATA LOADING
# =========================================
def get_full_feature_table_and_rows(corpus_paths: list[str], cfg: Config) -> tuple[pd.DataFrame, list[TokenRow]]:
    """
    Loads and processes the entire training corpus into a feature DataFrame.

    This function iterates through all the pre-enriched training JSON files in
    the corpus. For each decision point (boundary between two tokens), it calls
    `create_feature_row` to generate a flat dictionary of discrete features.

    It produces two key outputs:
    1.  A pandas DataFrame where each row is a decision point and each column
        is a feature. This is the primary input for the `build_weights` function.
    2.  A parallel list of `TokenRow` objects, which is used during the
        iterative reweighting process to re-score the training data with an
        updated model.

    Args:
        corpus_paths: A list of file paths to the training JSON files.
        cfg: The main `Config` object.

    Returns:
        A tuple containing:
        - The featurized pandas DataFrame.
        - A parallel list of `TokenRow` objects.
    """
    all_breakpoints_data = []
    all_token_rows = []
    print("Building full feature table from pre-engineered training data...")
    
    for path in tqdm(corpus_paths, desc="Processing Corpus"):
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            tokens = data.get("tokens", [])
            if not tokens: continue
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"\nWarning: Skipping file {path} due to error: {e}")
            continue
        
        for i in range(len(tokens) - 1):
            token = tokens[i]
            nxt = tokens[i+1]

            if token.get("break_type") is None:
                continue

            if token.get("w", "").strip() in ("-", "–", "—"):
                continue

            row = TokenRow(token=token, nxt=nxt, feats=Engineered())
            feature_dict = create_feature_row(row, cfg)
            all_breakpoints_data.append(feature_dict)
            all_token_rows.append(row)
            
    return pd.DataFrame(all_breakpoints_data), all_token_rows

def run_training(
    corpus_dir: str | Path,
    constraints_path: str | Path,
    weights_path: str | Path,
    *,
    config_path: str | Path = "config.yaml",
    iterations: int = 3,
    error_boost_factor: float = 1.0,
) -> dict[str, str]:
    """Train the statistical model and persist the resulting artifacts.

    This helper encapsulates the full training workflow so it can be invoked
    both from the command line as well as programmatically (e.g. the control
    panel UI).

    Args:
        corpus_dir: Directory containing ``*.json`` training files produced by
            ``build_training_pair_standalone.py``.
        constraints_path: Destination path for the generated ``constraints.json``.
        weights_path: Destination path for the generated ``model_weights.json``.
        config_path: Optional path to the configuration YAML file.
        iterations: Number of reweighting passes to run.
        error_boost_factor: Amount added to the weight of each misclassified
            sample during boosting.

    Returns:
        A dictionary containing the persisted ``constraints_path`` and
        ``weights_path`` so callers can surface the outputs in their own
        metadata stores.

    Raises:
        FileNotFoundError: If the corpus directory does not contain any JSON
            files.
        ValueError: If the corpus could not be converted into a feature table.
        RuntimeError: If training fails to produce a set of weights.
    """

    corpus_dir = Path(corpus_dir)
    constraints_path = Path(constraints_path)
    weights_path = Path(weights_path)
    config_path = Path(config_path)

    corpus_paths = sorted(str(p) for p in corpus_dir.glob("*.json"))
    if not corpus_paths:
        raise FileNotFoundError(
            f"No .json files found in corpus directory: {corpus_dir}"
        )

    print(f"Found {len(corpus_paths)} training files.")

    cfg = load_config(config_path)

    print("\n--- Deriving Constraints ---")
    constraints = derive_constraints(corpus_paths, cfg)
    constraints_path.parent.mkdir(parents=True, exist_ok=True)
    with open(constraints_path, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)
    print(f"Successfully saved constraints to {constraints_path}")

    df, token_rows = get_full_feature_table_and_rows(corpus_paths, cfg)

    if df.empty:
        raise ValueError(
            "No valid training data could be loaded. The feature table is empty."
        )

    sample_weights = pd.Series(1.0, index=df.index)
    print("\nStarting with uniform sample weights.")

    final_weights: Optional[dict[str, Any]] = None
    for i in range(iterations):
        print(f"\n--- Starting Training Iteration {i + 1}/{iterations} ---")

        current_weights = build_weights(df, cfg=cfg, sample_weights=sample_weights)
        final_weights = current_weights

        if i == iterations - 1:
            break

        print("Evaluating model on training data to find hard examples...")
        scorer = Scorer(weights=current_weights, constraints={}, sliders={}, cfg=cfg)

        predictions = []
        for row in tqdm(token_rows, desc=f"Predicting (Iter {i+1})"):
            scores = scorer.score_transition(row)
            prediction = max(scores, key=scores.get)
            predictions.append(prediction)

        df["prediction"] = predictions
        errors = df["prediction"] != df["outcome"]

        accuracy = 1 - errors.mean()
        print(f"Iteration {i + 1} accuracy on training set: {accuracy:.2%}")

        if not errors.any():
            print("Model achieved 100% accuracy on the training set. Stopping early.")
            break

        print(f"Boosting weights of {errors.sum()} misclassified samples...")
        sample_weights[errors] += error_boost_factor

    if final_weights is None:
        raise RuntimeError("Training did not produce any weights.")

    print("\n--- Final Model Training Complete ---")
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(final_weights, f, indent=2)
    print(f"Successfully saved final model weights to {weights_path}")
    print("\nAdvanced model training complete.")

    return {
        "constraints_path": str(constraints_path),
        "weights_path": str(weights_path),
    }


def main():
    """Command-line entry point for training the statistical model."""

    parser = argparse.ArgumentParser(
        description="Build an advanced statistical model using class balancing and iterative reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--corpus", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--constraints", type=str, required=True, help="Output path for constraints.json.")
    parser.add_argument("--weights", type=str, required=True, help="Output path for model_weights.json.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of reweighting iterations to perform.")
    parser.add_argument(
        "--error-boost-factor",
        type=float,
        default=1.0,
        help="Amount to ADD to the weight of misclassified samples.",
    )
    args = parser.parse_args()

    try:
        run_training(
            corpus_dir=args.corpus,
            constraints_path=args.constraints,
            weights_path=args.weights,
            config_path=args.config,
            iterations=args.iterations,
            error_boost_factor=args.error_boost_factor,
        )
    except Exception as exc:  # pragma: no cover - surfaced to CLI only
        print(f"\n[ERROR] {exc}")
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
