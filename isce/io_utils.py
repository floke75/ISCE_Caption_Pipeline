# C:\dev\Captions_Formatter\Formatter_machine\isce\io_utils.py

import json
from typing import List
from .types import Token

def load_tokens(path: str) -> List[Token]:
    """
    Loads a list of tokens from a JSON file.
    It safely handles cases where the JSON might have extra fields not present
    in the Token dataclass, making it robust to variations in the input data.
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
    """Saves a list of Token objects to a JSON file in a structured format."""
    token_dicts = [t.__dict__ for t in tokens]
    data = {"tokens": token_dicts}
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)