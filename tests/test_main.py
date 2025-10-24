import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import patch, MagicMock
from main import main

import json
import pytest
from unittest.mock import patch, MagicMock
from main import main

@patch("main.tokens_to_srt")
@patch("main.segment")
@patch("main.Scorer")
@patch("main.load_config")
@patch("isce.io_utils.load_tokens")
def test_main(mock_load_tokens, mock_load_config, mock_scorer, mock_segment, mock_tokens_to_srt, tmp_path):
    """Test that the main function writes the correct content to the output file."""
    # Arrange
    input_file = tmp_path / "input.json"
    input_file.write_text(json.dumps({"tokens": []}))
    output_file = tmp_path / "output.srt"
    config_file = tmp_path / "config.yaml"

    # Create dummy constraints and weights files
    constraints_file = tmp_path / "constraints.json"
    constraints_file.write_text(json.dumps({}))
    weights_file = tmp_path / "model_weights.json"
    weights_file.write_text(json.dumps({}))

    mock_load_config.return_value = MagicMock(
        paths={"constraints": str(constraints_file), "model_weights": str(weights_file)},
        sliders={}
    )
    mock_tokens_to_srt.return_value = "srt content"

    sys.argv = [
        "main.py",
        "--input",
        str(input_file),
        "--output",
        str(output_file),
        "--config",
        str(config_file),
    ]

    # Act
    main()

    # Assert
    assert output_file.read_text() == "srt content"
