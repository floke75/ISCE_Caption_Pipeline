# C:\dev\Captions_Formatter\Formatter_machine\main.py

import argparse
import json
import sys
from pathlib import Path

# Add project root to path for robust execution
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from isce.config import load_config
from isce.io_utils import load_tokens, save_tokens
from isce.scorer import Scorer
from isce.beam_search import segment
from isce.srt_writer import tokens_to_srt

def main():
    """
    Main command-line interface for the ISCE captioning engine.

    This script orchestrates the final segmentation and SRT generation process.
    It performs the following steps:
    1.  Loads the main configuration file (`config.yaml`), the statistical model
        weights, and the corpus constraints.
    2.  Loads the enriched tokens from the input JSON file.
    3.  Initializes the `Scorer` with the loaded models and configuration.
    4.  Runs the beam search segmentation algorithm (`segment`) to determine the
        optimal break points (`SB`, `LB`, `O`).
    5.  Formats the segmented tokens into the standard SRT file format.
    6.  Writes the final output to the specified SRT file.
    """
    parser = argparse.ArgumentParser(
        description="Generate subtitles from enriched token files using the ISCE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input", 
        required=True, 
        help="Path to the input (unlabeled) enriched tokens JSON file."
    )
    parser.add_argument(
        "--output", 
        required=True, 
        help="Path to write the output SRT file."
    )
    parser.add_argument(
        "--config", 
        default="config.yaml", 
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--save-labeled-json",
        action="store_true",
        help="In addition to the SRT, save the output as a labeled JSON file."
    )
    parser.add_argument(
        "--refine-blocks",
        dest="refine_blocks",
        action="store_true",
        help="Enable the localized refinement pass to re-score low quality cues.",
    )
    parser.add_argument(
        "--no-refine-blocks",
        dest="refine_blocks",
        action="store_false",
        help="Disable the localized refinement pass regardless of config settings.",
    )
    parser.set_defaults(refine_blocks=None)
    args = parser.parse_args()

    try:
        # 1. Load configurations
        print(f"Loading configuration from {args.config}...")
        cfg = load_config(args.config)

        if args.refine_blocks is not None:
            cfg.enable_refinement_pass = args.refine_blocks

        # 2. Load input data
        print(f"Loading tokens from {args.input}...")
        tokens = load_tokens(args.input)

        # 3. Load the statistical model and constraints
        config_dir = Path(args.config).parent
        constraints_path = config_dir / cfg.paths["constraints"]
        weights_path = config_dir / cfg.paths["model_weights"]
        
        print(f"Loading constraints from {constraints_path}...")
        with open(constraints_path, "r", encoding="utf-8") as f:
            constraints = json.load(f)
            
        print(f"Loading model weights from {weights_path}...")
        with open(weights_path, "r", encoding="utf-8") as f:
            weights = json.load(f)

        # 4. Initialize the Scorer
        scorer = Scorer(weights=weights, constraints=constraints, sliders=cfg.sliders, cfg=cfg)

        # 5. Run the segmentation algorithm
        print("Segmenting tokens...")
        segmented_tokens = segment(tokens, scorer, cfg)

        # 6. Format output as SRT
        print("Formatting output to SRT...")
        srt_content = tokens_to_srt(segmented_tokens)

        # 7. Write to output file(s)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt_content)
        
        print(f"\nSuccessfully wrote SRT output to {args.output}")

        if args.save_labeled_json:
            json_output_path = output_path.with_suffix('.json')
            save_tokens(str(json_output_path), segmented_tokens)
            print(f"Successfully wrote labeled JSON to {json_output_path}")

    except (FileNotFoundError, ValueError, TypeError, KeyError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()