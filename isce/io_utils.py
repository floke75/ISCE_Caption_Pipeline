# C:\dev\Captions_Formatter\Formatter_machine\isce\io_utils.py
"""Provides utility functions for loading and saving token data.

This module contains helper functions dedicated to the serialization and
deserialization of `Token` objects. It defines a standardized JSON structure
where the token list is stored under a "tokens" key. The `load_tokens` function
is designed to be robust against extra, unknown fields in the input JSON,
while the `save_tokens` function ensures a consistent and human-readable
output format.
"""
import json
from typing import List
from .types import Token

def load_tokens(path: str) -> List[Token]:
    """
    Loads a list of Token objects from a JSON file.

    This function is responsible for parsing a JSON file that contains a list
    of token dictionaries under the "tokens" key. It is designed to be robust
    by filtering the loaded dictionaries to only include keys that are valid
    fields in the `Token` dataclass. This prevents errors if the input JSON
    contains extra, unexpected fields.

    Args:
        path: The path to the input JSON file.

    Returns:
        A list of `Token` dataclass instances.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
        ValueError: If the file is not valid JSON.
        TypeError: If the JSON structure is incorrect (e.g., "tokens" key is
                   missing or not a list, or an item in the list is not a
                   dictionary).
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file not found at: {path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from {path}: {e}")

    items = data.get("tokens")
    if not isinstance(items, list):
        raise TypeError(f"Expected a 'tokens' key with a list of objects in {path}")

    out = []
    token_fields = Token.get_field_names()
    for i, t_dict in enumerate(items):
        if not isinstance(t_dict, dict):
            raise TypeError(f"Token item at index {i} in {path} is not a dictionary.")
        
        # Filter the dictionary to only include keys that are fields in the Token dataclass
        filtered_dict = {k: v for k, v in t_dict.items() if k in token_fields}
        
        try:
            out.append(Token(**filtered_dict))
        except TypeError as e:
            raise TypeError(f"Mismatch between JSON object and Token dataclass at index {i} in {path}: {e}")
            
    return out

def save_tokens(path: str, tokens: List[Token]) -> None:
    """
    Saves a list of Token objects to a JSON file.

    This function serializes a list of `Token` dataclass instances into a
    JSON file with a specific structure: the root of the JSON is a dictionary
    with a single key, "tokens", which contains the list of token dictionaries.
    The output is formatted with indentation for human readability.

    Args:
        path: The destination path for the output JSON file.
        tokens: The list of `Token` objects to save.
    """
    token_dicts = [t.__dict__ for t in tokens]
    data = {"tokens": token_dicts}
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)