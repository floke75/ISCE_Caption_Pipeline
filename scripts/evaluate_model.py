# C:\dev\Captions_Formatter\Formatter_machine\scripts\evaluate_model.py
"""Command-line script for evaluating the performance of a trained model.

This script serves as a key tool for model validation and diagnostics. It
compares the output of the segmentation model against a ground-truth, human-
labeled reference file. The evaluation is multifaceted, providing both
comparison metrics and intrinsic quality scores.

The primary functions are:
-   **Comparison Metrics**: Calculates precision, recall, and F1-score for each
    break type (`SB`, `LB`, `O`) by comparing the model's decisions to the
    reference labels. It can also generate a detailed CSV report of every
    point of disagreement, which is invaluable for error analysis.
-   **Intrinsic Readability Metrics**: Assesses the quality of the model's
    output on its own, without a reference. It checks for violations of
    learned constraints (e.g., block duration, characters-per-second) to
    produce a single "readability score."
"""
import argparse
import json
import sys
import csv
from pathlib import Path

# Add project root to path to allow for package imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Note: This script depends on isce/evaluate.py, which was not provided.
# This code assumes the functions within it are compatible with the new Token object.
try:
    from isce.evaluate import compare_breaks, readability_from_tokens
except ImportError:
    print("\n[ERROR] Could not import evaluation functions from 'isce/evaluate.py'.")
    print("Please ensure this file exists and is correct.")
    # Define dummy functions so the script doesn't crash immediately
    def compare_breaks(*args, **kwargs):
        print("[WARN] 'compare_breaks' function is not available.")
        return {'scores': {}, 'disagreements': []}
    def readability_from_tokens(*args, **kwargs):
        print("[WARN] 'readability_from_tokens' function is not available.")
        return {'readability_score': 0, 'violations': []}
    
from isce.io_utils import load_tokens

def main():
    """
    Main entry point for the command-line model evaluation script.

    This script provides a comprehensive evaluation of a trained model's
    performance by comparing its output against a ground-truth reference file.
    It performs two main types of analysis:

    1.  **Comparison Metrics**: It directly compares the break decisions (`SB`,
        `LB`, `O`) between the generated file and the reference file,
        calculating metrics like precision, recall, and F1-score for each
        break type. It can optionally save a detailed CSV of every
        disagreement.

    2.  **Intrinsic Readability Metrics**: It analyzes the generated file on its
        own to assess its quality based on the data-driven constraints. This
        includes checking for violations of ideal CPS, block duration, and line
        length, providing a "readability score."

    The script requires both the generated and reference files to be in the
    enriched token JSON format.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate caption segmentation performance against a reference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--generated", required=True, help="Path to the generated (model output) labeled tokens JSON file.")
    parser.add_argument("--reference", required=True, help="Path to the ground-truth (human-edited) labeled tokens JSON file.")
    parser.add_argument("--constraints", required=True, help="Path to the constraints.json file used by the model.")
    parser.add_argument("--disagreements-out", help="Optional: Path to write a detailed disagreements CSV file.")
    args = parser.parse_args()

    try:
        print("Loading files...")
        gen_tokens = load_tokens(args.generated)
        ref_tokens = load_tokens(args.reference)
        with open(args.constraints, "r", encoding="utf-8") as f:
            constraints = json.load(f)

        print("\n--- Comparison Metrics (vs. Reference) ---")
        comparison_report = compare_breaks(gen_tokens, ref_tokens)
        if comparison_report.get('scores'):
            print(json.dumps(comparison_report['scores'], indent=2))
        
        if args.disagreements_out and comparison_report.get('disagreements'):
            Path(args.disagreements_out).parent.mkdir(parents=True, exist_ok=True)
            print(f"\nWriting {len(comparison_report['disagreements'])} disagreements to {args.disagreements_out}...")
            with open(args.disagreements_out, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["index", "token", "generated", "reference"])
                writer.writeheader()
                writer.writerows(comparison_report['disagreements'])

        print("\n--- Intrinsic Readability Metrics (of Generated File) ---")
        readability_report = readability_from_tokens(gen_tokens, constraints)
        print(f"Readability Score: {readability_report.get('readability_score', 0):.2f} / 100")
        if readability_report.get('violations'):
            print("Violations found:")
            for v in readability_report['violations']:
                print(f"  - {v}")

    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()