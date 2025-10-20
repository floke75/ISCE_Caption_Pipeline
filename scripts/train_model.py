# C:\dev\Captions_Formatter\Formatter_machine\scripts\train_model.py

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from isce.model_builder import build_weights, derive_constraints, create_feature_row
from isce.config import Config, load_config
from isce.scorer import Scorer

# =========================================
# LOCAL DATA STRUCTURES
# =========================================
@dataclass(frozen=True)
class Engineered:
    """Minimal container for features requiring logical checks in the scorer."""
    pass # In the new design, features are read directly from the token dict.

@dataclass
class TokenRow:
    """A container holding dictionaries for the current and next token."""
    token: dict
    nxt: Optional[dict]
    feats: Engineered

# =========================================
# REFACTORED DATA LOADING
# =========================================
def get_full_feature_table_and_rows(corpus_paths: list[str], cfg: Config) -> tuple[pd.DataFrame, list[TokenRow]]:
    """
    Loads pre-enriched data, creates the feature DataFrame, and a
    parallel list of TokenRow objects for accurate scoring.
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

def main():
    parser = argparse.ArgumentParser(
        description="Build an advanced statistical model using class balancing and iterative reweighting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--corpus", type=str, required=True, help="Path to the training data directory.")
    parser.add_argument("--constraints", type=str, required=True, help="Output path for constraints.json.")
    parser.add_argument("--weights", type=str, required=True, help="Output path for model_weights.json.")
    parser.add_argument("--config", default="config.yaml", help="Path to the configuration YAML file.")
    parser.add_argument("--iterations", type=int, default=3, help="Number of reweighting iterations to perform.")
    parser.add_argument("--error-boost-factor", type=float, default=1.0, help="Amount to ADD to the weight of misclassified samples.")
    args = parser.parse_args()

    corpus_paths = [str(p) for p in Path(args.corpus).glob("*.json")]
    if not corpus_paths:
        raise FileNotFoundError(f"No .json files found in corpus directory: {args.corpus}")

    print(f"Found {len(corpus_paths)} training files.")
    
    cfg = load_config(args.config)
    
    print("\n--- Deriving Constraints ---")
    constraints = derive_constraints(corpus_paths, cfg)
    with open(args.constraints, "w", encoding="utf-8") as f:
        json.dump(constraints, f, indent=2)
    print(f"Successfully saved constraints to {args.constraints}")

    df, token_rows = get_full_feature_table_and_rows(corpus_paths, cfg)
    
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