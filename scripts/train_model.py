"""Command-line script for training the statistical segmentation model.

This script orchestrates the entire model training workflow. It takes a corpus
of labeled training data (enriched JSON files) and produces two key artifacts:
`constraints.json` and `model_weights.json`.

The training process involves several stages:
1.  **Constraint Derivation**: It first analyzes the corpus to learn the
    statistical properties of human-made subtitles, saving these as hard and
    soft constraints for the segmentation algorithm.
2.  **Feature Engineering**: It processes the entire corpus, transforming the
    rich token data into a discrete feature set suitable for a statistical model.
3.  **Iterative Reweighting (Hard Example Mining)**: The core of the training
    process. It runs in a loop where:
    a. A model is trained on the current data.
    b. The model is used to predict on the training set itself.
    c. The examples that the model gets wrong ("hard examples") are identified.
    d. The sample weight of these hard examples is increased.
    This forces the model in the next iteration to pay more attention to the
    examples it previously failed on, leading to a more robust final model.
4.  **Final Model Saving**: After the final iteration, the script saves the
    fully trained model weights.
"""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from typing import Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from isce.model_builder import build_weights, derive_constraints, create_feature_row
from isce.config import Config, load_config
from isce.scorer import Scorer
from isce.types import TokenRow, Engineered
from isce.token_utils import normalize_token_payload

# =========================================
# REFACTORED DATA LOADING
# =========================================
RAW_FILENAME_MARKERS = (".raw.", ".raw_")


def partition_corpus_paths(corpus_dir: Path) -> Tuple[list[Path], list[Path]]:
    """Split training files into human-edited and simulated-ASR variants.

    Parameters
    ----------
    corpus_dir:
        Directory expected to contain ``*.train.words.json`` files.  Both the
        hand-aligned transcripts and the optional ``*.raw`` synthetic variants
        live side-by-side in this location.

    Returns
    -------
    tuple[list[Path], list[Path]]
        Two lists of paths ``(human_paths, raw_paths)`` filtered to exclude
        unrelated JSON sidecars.  The first element contains the gold-standard
        human alignments while the second contains WhisperX style synthetic
        copies that can be optionally folded into training.

    Notes
    -----
    The heavy filtering is intentionally duplicated here so that corpus
    partitioning stays consistent with the old CLI behaviour.  The verbose
    filtering steps prevent ``notes.json`` or other metadata sidecars from
    sneaking into the training set and upsetting feature generation.
    """

    human_paths: list[Path] = []
    raw_paths: list[Path] = []

    for path in sorted(corpus_dir.glob("*.json")):
        name = path.name.lower()

        if ".words." not in name:
            continue

        if ".words.json" not in name:
            continue

        if not name.endswith("words.json"):
            continue

        # Skip JSON sidecar files that do not contain training token data.
        # Only `*.train.words.json` (and their simulated `*.train.raw.words.json`
        # counterparts) should participate in constraint derivation.  The test
        # corpus fixtures also include miscellaneous metadata files (for example
        # `notes.json`) which previously slipped into the human file list.  That
        # polluted the training set and caused the filtering logic in
        # `derive_constraints` to run against empty or malformed payloads.
        if ".train." not in name:
            continue

        if any(marker in name for marker in RAW_FILENAME_MARKERS):
            raw_paths.append(path)
        else:
            human_paths.append(path)

    return human_paths, raw_paths


def get_full_feature_table_and_rows(corpus_paths: list[str], cfg: Config) -> tuple[pd.DataFrame, list[TokenRow]]:
    """Materialise feature tables and ``TokenRow`` objects from the corpus.

    Parameters
    ----------
    corpus_paths:
        Ordered collection of enriched ``*.train.words.json`` file paths.
    cfg:
        Loaded :class:`~isce.config.Config` containing feature engineering
        settings.

    Returns
    -------
    tuple[pd.DataFrame, list[TokenRow]]
        * A pandas ``DataFrame`` with one row per decision point and columns for
          every engineered feature.
        * A list of :class:`~isce.types.TokenRow` instances mirroring the rows in
          the DataFrame.  These objects are used during iterative reweighting to
          rescore the corpus with fresh weights.

    Notes
    -----
    The function performs the exact same normalisation as the live scorer by
    invoking :func:`isce.token_utils.normalize_token_payload` on every token
    pair.  This keeps feature keys (for example, lemma bigrams) stable between
    training and inference even when the source JSON has mixed types.
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

        for idx, token in enumerate(tokens):
            token.setdefault("token_index", idx)

        for i in range(len(tokens) - 1):
            token = normalize_token_payload(tokens[i], idx=i) or {}
            nxt = normalize_token_payload(tokens[i + 1], idx=i + 1)

            if token.get("break_type") is None:
                continue

            if token.get("w", "").strip() in ("-", "–", "—"):
                continue

            row = TokenRow(token=token, nxt=nxt, feats=Engineered())
            feature_dict = create_feature_row(row, cfg)
            all_breakpoints_data.append(feature_dict)
            all_token_rows.append(row)
            
    return pd.DataFrame(all_breakpoints_data), all_token_rows

def main():
    """
    Main entry point for the command-line model training script.

    This script orchestrates the entire model training process, which includes:
    1.  Parsing command-line arguments for corpus path, output paths, and
        training parameters.
    2.  Loading the base configuration.
    3.  Running `derive_constraints` on the corpus to generate and save
        `constraints.json`.
    4.  Loading the entire training corpus into a feature DataFrame using
        `get_full_feature_table_and_rows`.
    5.  Executing an iterative reweighting loop:
        a.  Build a weights model using the current sample weights.
        b.  Score the entire training set with the new model.
        c.  Identify misclassified examples ("hard examples").
        d.  Increase the sample weight of the hard examples.
    6.  Saving the final, trained `model_weights.json` after the last iteration.
    """
    parser = argparse.ArgumentParser(
        description="Build an advanced statistical model using class balancing and iterative reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument(
        "--include-simulated-raw",
        action="store_true",
        help=(
            "Include *.train.raw.words.json corpora when building the feature table. "
            "Disabled by default to avoid duplicating synthetic ASR copies."
        ),
    )
    args = parser.parse_args()

    corpus_dir = Path(args.corpus)
    human_paths, raw_paths = partition_corpus_paths(corpus_dir)
    if not human_paths and not raw_paths:
        raise FileNotFoundError(f"No .json files found in corpus directory: {args.corpus}")

    print(f"Found {len(human_paths)} human-edited training file(s).")

    if args.include_simulated_raw:
        training_paths = human_paths + raw_paths
        if raw_paths:
            print(f"Including {len(raw_paths)} simulated raw training file(s).")
    else:
        training_paths = human_paths
        if raw_paths:
            print(
                "Skipping %d simulated raw training file(s) (use --include-simulated-raw to include)."
                % len(raw_paths)
            )
            print(
                "These raw duplicates flatten punctuation and timing cues, so we exclude them to keep"
                " statistics anchored to human formatting decisions."
            )

    if not training_paths:
        raise FileNotFoundError(
            "Only simulated raw training files were found. Rerun with --include-simulated-raw to train on them."
        )

    cfg = load_config(args.config)

    print("\n--- Deriving Constraints ---")
    constraint_paths = human_paths if human_paths else training_paths
    constraints = derive_constraints([str(p) for p in constraint_paths], cfg)
    with open(args.constraints, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)
    print(f"Successfully saved constraints to {args.constraints}")

    df, token_rows = get_full_feature_table_and_rows([str(p) for p in training_paths], cfg)
    
    if df.empty:
        print("\n[ERROR] No valid training data could be loaded. The feature table is empty. Aborting.")
        sys.exit(1)
        
    sample_weights = pd.Series(1.0, index=df.index)
    print("\nStarting with uniform sample weights.")

    final_weights = None
    for i in range(args.iterations):
        print(f"\n--- Starting Training Iteration {i + 1}/{args.iterations} ---")
        
        current_weights = build_weights(df, cfg=cfg, sample_weights=sample_weights)
        final_weights = current_weights
        
        if i == args.iterations - 1:
            break

        print("Evaluating model on training data to find hard examples...")
        scorer = Scorer(weights=current_weights, constraints={}, sliders={}, cfg=cfg)
        
        predictions = []
        for row in tqdm(token_rows, desc=f"Predicting (Iter {i+1})"):
            scores = scorer.score_transition(row)
            prediction = max(scores, key=scores.get)
            predictions.append(prediction)
        
        df['prediction'] = predictions
        errors = df['prediction'] != df['outcome']
        
        accuracy = 1 - errors.mean()
        print(f"Iteration {i + 1} accuracy on training set: {accuracy:.2%}")
        
        if not errors.any():
            print("Model achieved 100% accuracy on the training set. Stopping early.")
            break

        print(f"Boosting weights of {errors.sum()} misclassified samples...")
        sample_weights[errors] += args.error_boost_factor

    print("\n--- Final Model Training Complete ---")
    with open(args.weights, "w", encoding="utf-8") as f:
        json.dump(final_weights, f, indent=2)
    print(f"Successfully saved final model weights to {args.weights}")
    print("\nAdvanced model training complete.")

if __name__ == "__main__":
    main()