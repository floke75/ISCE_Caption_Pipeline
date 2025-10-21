#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_training_pair_standalone.py — Self-contained data alignment and enrichment.
This is the refactored, consolidated version. Its mission is to take a
time-stamped ASR reference and a primary text file (TXT or SRT) and produce
a single, fully enriched JSON file for training or inference.
"""

import os
import json
import re
import statistics
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import unicodedata
import copy
from rapidfuzz import fuzz
import pysrt
import yaml

# =========================
# Dependency guards
# =========================
try:
    import spacy
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Doc = None

# =========================
# DEFAULT SETTINGS (Self-Contained)
# =========================
DEFAULT_SETTINGS: Dict[str, Any] = {
    "project_root": r"C:\dev\Captions_Formatter\Formatter_machine",
    "pipeline_root": r"T:\AI-Subtitles\Pipeline",
    "build_pair": {
        "in_txt_dir":          "{pipeline_root}/4_MANUAL_TXT_PLACEMENT",
        "in_srt_dir":          "{pipeline_root}/3_MANUAL_SRT_PLACEMENT",
        "out_training_dir":    "{pipeline_root}/_intermediate/_training",
        "out_inference_dir":   "{pipeline_root}/_intermediate/_inference_input",
        "language": "sv",
        "time_tolerance_s": 0.15,
        "round_seconds": 3,
        "enable_num_unit_glue": True,
        "num_regex": r"\d+[.,]?\d*",
        "unit_vocab": ["%", "kr", "kronor", "$", "€", "st", "kg", "cm", "mm", "m", "g"],
        "spacy_enable": True,
        "spacy_model": "sv_core_news_lg",
        "spacy_add_dependencies": True,
        "txt_match_close": 0.82,
        "txt_match_weak":  0.65,
        "emit_asr_style_training_copy": True,
    }
}

# =========================
# Configuration Helper Functions (Self-Contained)
# =========================
def _recursive_update(base: Dict, update: Dict) -> Dict:
    """
    Recursively updates a dictionary.

    Args:
        base: The dictionary to be updated.
        update: The dictionary containing new values.

    Returns:
        The updated 'base' dictionary.
    """
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base

def _resolve_paths(config: Dict, context: Dict) -> Dict:
    """
    Resolves placeholder variables in configuration paths.

    Args:
        config: The configuration dictionary with unresolved path strings.
        context: A dictionary mapping placeholder keys to their values.

    Returns:
        The configuration dictionary with path placeholders resolved.
    """
    for k, v in config.items():
        if isinstance(v, str) and "{" in v and "}" in v:
            try:
                config[k] = v.format(**context)
            except KeyError:
                pass
        elif isinstance(v, dict):
            config[k] = _resolve_paths(v, context)
    return config

# =========================
# Utilities
# =========================
def ensure_dirs(p: Path):
    """Ensures that the directory for a given path exists."""
    p.mkdir(parents=True, exist_ok=True)

def base_of(path: Path) -> str:
    """Gets the base name of a file path, excluding the extension."""
    return path.stem

def _save_json(obj: dict, p: Path):
    """Saves a dictionary to a JSON file."""
    ensure_dirs(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def morph_to_str(m) -> Optional[str]:
    """
    Safely converts a spaCy MorphAnalysis object to a string.

    Args:
        m: The MorphAnalysis object.

    Returns:
        A string representation of the morphology, or None.
    """
    if not m: return None
    try: return m.to_string()
    except Exception: return str(m)

# =========================
# Migrated TXT/SRT Alignment Logic (PERFORMANCE OPTIMIZED)
# =========================
_PUNCT_EDGES_RE = re.compile(r"^\W+|\W+$", flags=re.UNICODE)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_STYLE_TAG_RE = re.compile(r"\{[^}]*\}")

def _norm_token(s: str) -> str:
    """Normalizes a token for robust comparison."""
    s2 = unicodedata.normalize("NFKC", s).casefold()
    s2 = _PUNCT_EDGES_RE.sub("", s2)
    return re.sub(r"\s+", "", s2)

def strip_rendered_markup(text: str) -> str:
    """Removes markup tags that do not appear in rendered captions."""
    without_styles = _STYLE_TAG_RE.sub("", text)
    without_html = _HTML_TAG_RE.sub("", without_styles)
    return without_html

def _match_score(a: str, b: str, settings: Dict) -> int:
    """Calculates a similarity score between two tokens."""
    na, nb = _norm_token(a), _norm_token(b)
    if not na and not nb: return 1
    if na == nb: return 4
    r = fuzz.ratio(na, nb) / 100.0
    if r >= settings.get("txt_match_close", 0.82): return 2
    if r >= settings.get("txt_match_weak", 0.65): return 0
    return -3

def _global_align(a_tokens: List[str], b_tokens: List[str], settings: Dict, gap_penalty: int = -3) -> List[Tuple[Optional[int], Optional[int]]]:
    """Performs a global sequence alignment (Needleman-Wunsch)."""
    n, m = len(a_tokens), len(b_tokens)
    S = [[0]*(m+1) for _ in range(n+1)]
    B = [[(0,0)]*(m+1) for _ in range(n+1)]
    for i in range(1, n+1): S[i][0] = i*gap_penalty; B[i][0] = (i-1, 0)
    for j in range(1, m+1): S[0][j] = j*gap_penalty; B[0][j] = (0, j-1)
    
    print(f"[DEBUG] Starting global alignment of {n}x{m} matrix...", flush=True)
    for i in range(1, n+1):
        if i > 0 and i % 50 == 0:
            print(f"  -> Global alignment progress: {i}/{n} rows processed...", flush=True)
        for j in range(1, m+1):
            s_diag = S[i-1][j-1] + _match_score(a_tokens[i-1], b_tokens[j-1], settings)
            s_up   = S[i-1][j] + gap_penalty
            s_left = S[i][j-1] + gap_penalty
            best, bp = max((s_diag, (i-1, j-1)), (s_up, (i-1, j)), (s_left, (i, j-1)))
            S[i][j], B[i][j] = best, bp
            
    print(f"[DEBUG] Global alignment matrix built. Tracing back path...", flush=True)
    path: List[Tuple[Optional[int], Optional[int]]] = []
    i, j = n, m
    count = 0
    while i > 0 or j > 0:
        count += 1
        if count > 1 and count % 100 == 0:
            print(f"  -> Traceback progress: {count} steps...", flush=True)
        if count > (n + m) * 2:
            print(f"[ERROR] Traceback is taking too long, breaking.", flush=True)
            break
        pi, pj = B[i][j]
        if pi == i-1 and pj == j-1: path.append((i-1, j-1))
        elif pi == i-1 and pj == j: path.append((i-1, None))
        else: path.append((None, j-1))
        i, j = pi, pj
    path.reverse()
    print(f"[DEBUG] Traceback finished after {count} steps.", flush=True)
    return path

def _safe_interval_split(t0: float, t1: float, k: int) -> List[Tuple[float,float]]:
    if k <= 0: return []
    t1 = max(t0, t1)
    step = (t1 - t0)/k if k > 0 else 0
    return [(t0 + i*step, t0 + (i+1)*step) for i in range(k)]

def align_text_to_asr(
    edited_tokens: List[str],
    asr_words: List[Dict[str, Any]],
    settings: Dict,
    return_alignment: bool = False,
) -> Any:
    """
    Aligns a list of edited text tokens to a reference ASR transcript.

    This function uses a global sequence alignment algorithm to find the
    optimal mapping between the edited tokens and the ASR words. It then
    transfers the timestamps from the ASR words to the edited tokens.
    For tokens that are inserted (i.e., have no corresponding ASR word),
    it interpolates timestamps based on the surrounding aligned words.

    Args:
        edited_tokens: A list of strings from the primary input (TXT or SRT).
        asr_words: A list of word dictionaries from the ASR reference,
                   each containing 'start', 'end', and 'speaker' keys.
        settings: Configuration dictionary for alignment parameters.

    Returns:
        If `return_alignment` is False (default), returns a new list of word
        dictionaries where each dictionary corresponds to an edited token and
        has been assigned timestamps and a speaker.

        If `return_alignment` is True, returns a tuple of three elements:
        (`aligned_tokens`, `source_indices`, `asr_indices`). `aligned_tokens`
        contains the enriched tokens described above, `source_indices` maps
        each aligned token back to its originating index in `edited_tokens`,
        and `asr_indices` captures the originating index of the ASR word that
        donated the timestamps (or `None` when timestamps were interpolated).
    """
    asr_token_texts = [str(w.get("w") or "") for w in asr_words]
    if not edited_tokens:
        if return_alignment:
            return [], [], []
        return []
    if not asr_token_texts:
        slices = _safe_interval_split(0.0, 0.01 * len(edited_tokens), len(edited_tokens))
        aligned_tokens = [
            {"w": t, "start": s, "end": e, "speaker": None}
            for t, (s, e) in zip(edited_tokens, slices)
        ]
        if return_alignment:
            none_alignment = [None] * len(edited_tokens)
            return aligned_tokens, list(range(len(edited_tokens))), none_alignment
        return aligned_tokens

    path = _global_align(edited_tokens, asr_token_texts, settings)

    print(f"[DEBUG] Reconstructing aligned words from path ({len(path)} steps)...", flush=True)
    out: List[Dict[str, Any]] = []
    source_alignment: List[Optional[int]] = []
    asr_alignment: List[Optional[int]] = []
    last_time = 0.0
    idx = 0
    while idx < len(path):
        if idx > 0 and idx % 100 == 0:
            print(f"  -> Path reconstruction progress: {idx}/{len(path)} steps...", flush=True)
        i, j = path[idx]
        if i is not None and j is not None:
            w_info = asr_words[j]
            st = max(last_time, float(w_info.get("start", 0.0)))
            en = max(st, float(w_info.get("end", 0.0)))
            out.append({"w": edited_tokens[i], "start": st, "end": en, "speaker": w_info.get("speaker")})
            source_alignment.append(i)
            asr_alignment.append(j)
            last_time = en
            idx += 1
        elif j is None:
            run_i: List[int] = []
            k = idx
            while k < len(path) and path[k][0] is not None and path[k][1] is None:
                run_i.append(path[k][0]); k += 1
            j_left = path[idx-1][1] if idx > 0 else None
            j_right = path[k][1] if k < len(path) else None
            t_left = asr_words[j_left]["end"] if j_left is not None else last_time
            t_right = asr_words[j_right]["start"] if j_right is not None else (t_left + 0.01 * len(run_i))
            spk_left = asr_words[j_left].get("speaker") if j_left is not None else None
            spk_right = asr_words[j_right].get("speaker") if j_right is not None else None
            slices = _safe_interval_split(max(last_time, t_left), t_right, len(run_i))
            for s_idx, i_tok in enumerate(run_i):
                st, en = slices[s_idx]
                out.append({"w": edited_tokens[i_tok], "start": st, "end": en, "speaker": spk_left or spk_right})
                source_alignment.append(i_tok)
                asr_alignment.append(j_left if j_left is not None else j_right)
                last_time = en
            idx = k
        else:
            idx += 1
    for u in range(1, len(out)):
        if out[u]["start"] < out[u-1]["end"]: out[u]["start"] = out[u-1]["end"]
        if out[u]["end"] < out[u]["start"]: out[u]["end"] = out[u]["start"]
    print(f"[DEBUG] Finished reconstructing {len(out)} words.", flush=True)
    if return_alignment:
        return out, source_alignment, asr_alignment
    return out

def read_srt_cues(path: Path) -> List[Dict[str, Any]]:
    """Reads an SRT file and converts its cues into a list of dictionaries."""
    srt = pysrt.open(str(path), encoding="utf-8")
    return [{"id": i, "start": c.start.ordinal/1000.0, "end": c.end.ordinal/1000.0, "text": (c.text or "").replace("\r", "")} for i, c in enumerate(srt)]

def tokenize_srt_cues(cues: List[Dict[str, Any]]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Tokenizes cue text while preserving structural metadata."""
    tokens: List[str] = []
    metadata: List[Dict[str, Any]] = []

    for cue in cues:
        cue_id = int(cue["id"])
        text = (cue.get("text") or "").replace("\r", "")
        lines = text.split("\n")
        line_count = len(lines)
        cue_start = len(metadata)
        for line_idx, raw_line in enumerate(lines):
            clean_line = strip_rendered_markup(raw_line)
            parts = [p for p in clean_line.split() if p]
            if not parts:
                if len(metadata) > cue_start and line_idx < line_count - 1:
                    metadata[-1]["line_break_after"] = True
                continue

            for token_idx, part in enumerate(parts):
                tokens.append(part)
                metadata.append({
                    "cue_id": cue_id,
                    "line_index": line_idx,
                    "token_index": token_idx,
                    "line_break_after": False,
                    "is_last_in_cue": False,
                })

            if len(metadata) > cue_start:
                metadata[-1]["line_break_after"] = line_idx < line_count - 1

        if len(metadata) > cue_start:
            metadata[-1]["line_break_after"] = False
            metadata[-1]["is_last_in_cue"] = True

    return tokens, metadata

# =========================
# Data Loading
# =========================
def load_asr_words(path: Path) -> List[Dict[str, Any]]:
    """
    Loads and standardizes word data from an ASR JSON file.

    It extracts the list of words, ensures required keys ('start', 'end')
    are present, converts values to the correct types, and sorts the words
    chronologically.

    Args:
        path: The path to the ASR JSON file.

    Returns:
        A sorted list of standardized word dictionaries.
    """
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw = obj.get("words") or []
    words: List[Dict[str, Any]] = []
    for it in raw:
        if "start" not in it or "end" not in it: continue
        words.append({
            "w": str(it.get("w", it.get("text", ""))),
            "start": float(it["start"]),
            "end": float(it["end"]),
            "speaker": it.get("speaker"),
        })
    words.sort(key=lambda d: (d["start"], d["end"]))
    return words

# =========================
# Speaker Correction Logic
# =========================
def correct_speaker_labels(tokens: List[Dict[str, Any]]):
    """
    Corrects speaker labels using a "sole winner" algorithm at the sentence level.

    This function groups tokens into sentences. Within each sentence, it
    counts the occurrences of each speaker label. The speaker with the most
    occurrences is declared the "winner," and all tokens in that sentence
    are reassigned to that winning speaker. This helps correct minor
-    diarization errors.

    Args:
        tokens: A list of word dictionaries, each potentially having a 'speaker' key.
    """
    if not tokens: return
    print("[INFO] Running 'Sole Winner' speaker correction...")
    sentences = []
    current_sentence = []
    for token in tokens:
        current_sentence.append(token)
        if token.get("w", "").strip().endswith((".", "?", "!")):
            sentences.append(current_sentence)
            current_sentence = []
    if current_sentence:
        sentences.append(current_sentence)

    for sentence in sentences:
        if not sentence: continue
        speaker_counts = {}
        for token in sentence:
            speaker = token.get("speaker")
            if speaker:
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        if not speaker_counts: continue
        winner_speaker = max(speaker_counts, key=speaker_counts.get)
        for token in sentence:
            token["speaker"] = winner_speaker

# =========================
# Consolidated Feature Engineering
# =========================
def engineer_features(tokens: List[Dict[str, Any]], settings: Dict[str, Any]):
    """
    Adds a comprehensive set of linguistic and heuristic features to the tokens.

    This function operates in multiple passes:
    1.  **Prosody Features**: Calculates the pause duration after each token.
    2.  **SpaCy Linguistic Features**: (If enabled) Adds part-of-speech tags,
        lemmas, morphological information, and dependency parsing tags.
    3.  **Heuristic & Guardrail Features**: Adds features like speaker changes,
        sentence boundaries, and flags for specific patterns (e.g., a number
        followed by a unit).

    Args:
        tokens: The list of word dictionaries to be enriched.
        settings: A configuration dictionary for feature engineering.
    """
    if not tokens: return
    if settings.get("spacy_enable") and not spacy:
        raise RuntimeError(
            "spaCy is enabled in the configuration, but the spaCy package is not installed. "
            "Install the requested model or disable spaCy features to keep training and inference aligned."
        )
    # Pass 1: Prosody Features
    pause_after_values: List[int] = []
    for i in range(len(tokens)):
        prev_end = tokens[i-1]["end"] if i > 0 else tokens[i]["start"]
        pause_before = max(0.0, tokens[i]["start"] - prev_end)
        pause_after = max(0.0, tokens[i+1]["start"] - tokens[i]["end"]) if i < len(tokens) - 1 else 0.0
        pause_before_ms = int(round(pause_before * 1000))
        pause_after_ms = int(round(pause_after * 1000))
        tokens[i]["pause_before_ms"] = pause_before_ms
        tokens[i]["pause_after_ms"] = pause_after_ms
        pause_after_values.append(pause_after_ms)

    if pause_after_values:
        pause_mean = statistics.mean(pause_after_values)
        pause_std = statistics.pstdev(pause_after_values)
        denom = pause_std if pause_std > 1e-6 else max(abs(pause_mean), 1.0)
        if denom <= 0:
            denom = 1.0
        for token in tokens:
            token["pause_z"] = (token.get("pause_after_ms", 0) - pause_mean) / denom

    # Pass 2: SpaCy Linguistic Features
    if settings.get("spacy_enable"):
        print("[spaCy] Loading large NLP model. This may take several minutes...")
        try:
            nlp = spacy.load(settings["spacy_model"])
            print("[spaCy] NLP model loaded successfully.")
            words_text = [t.get("w", "") for t in tokens]
            doc = Doc(nlp.vocab, words=words_text)
            to_disable = ["ner"]
            if not settings.get("spacy_add_dependencies"):
                if "parser" in nlp.pipe_names: to_disable.append("parser")
            
            print("[spaCy] Processing text...")
            with nlp.select_pipes(disable=to_disable):
                doc = nlp(doc)

            for i, tok in enumerate(doc):
                tokens[i].update({
                    "pos": tok.pos_, "lemma": tok.lemma_, "tag": tok.tag_,
                    "morph": morph_to_str(tok.morph),
                })
                if settings.get("spacy_add_dependencies") and tok.has_dep():
                    tokens[i].update({"dep": tok.dep_, "head_idx": tok.head.i})
        except OSError as exc:
            raise RuntimeError(
                f"spaCy model '{settings['spacy_model']}' is not available. "
                "Install the model or disable spaCy features to ensure parity between training and inference."
            ) from exc
    
    # Pass 3: Heuristic, Sentence Boundary, and Guardrail Features
    num_re = re.compile(settings.get("num_regex", r"\d+[.,]?\d*"))
    unit_vocab = set(settings.get("unit_vocab", []))
    for i in range(len(tokens)):
        token = tokens[i]
        prv = tokens[i - 1] if i > 0 else None
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        
        is_raw_speaker_change = bool(nxt and token.get("speaker") != nxt.get("speaker"))
        if is_raw_speaker_change:
            pause_is_significant = token.get("pause_after_ms", 0) > 150 
            ends_sentence = token.get("w", "").strip().endswith((".", "?", "!"))
            token["speaker_change"] = True if pause_is_significant or ends_sentence else False
        else:
            token["speaker_change"] = False
        
        token["starts_with_dialogue_dash"] = bool(nxt and nxt.get("w", "").strip().startswith(('-', '–', '—')))
        is_numeric = num_re.fullmatch(str(token.get("w", "")))
        is_unit = bool(nxt and nxt.get("w", "").lower() in unit_vocab)
        token["num_unit_glue"] = bool(is_numeric and is_unit)

        current_word_is_capitalized = token.get("w", "") and token.get("w")[0].isupper()
        prev_ends_with_punctuation = prv.get("w", "").strip().endswith((".", "?", "!")) if prv else False
        token["is_sentence_initial"] = (i == 0 or (prev_ends_with_punctuation and current_word_is_capitalized))

        ends_with_punctuation = token.get("w", "").strip().endswith((".", "?", "!"))
        next_word_is_sentence_initial = False
        if nxt:
            next_is_cap = nxt.get("w", "") and nxt.get("w")[0].isupper()
            next_word_is_sentence_initial = ends_with_punctuation and next_is_cap
        token["is_sentence_final"] = ends_with_punctuation and (nxt is None or next_word_is_sentence_initial)

    # Pass 4: Dangling sentence detection
    dangling_threshold = settings.get("dangling_pause_threshold_ms", 300)
    for i, token in enumerate(tokens):
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if not nxt:
            token["is_dangling_eos"] = False
            continue
        same_speaker = token.get("speaker") == nxt.get("speaker")
        next_is_sentence_initial = nxt.get("is_sentence_initial", False)
        token["is_dangling_eos"] = bool(
            token.get("is_sentence_final")
            and same_speaker
            and not next_is_sentence_initial
            and token.get("pause_after_ms", 0) <= dangling_threshold
        )

    # Pass 5: Relative sentence position
    def assign_relative_positions(sentence_tokens: List[Dict[str, Any]]):
        if not sentence_tokens:
            return
        if len(sentence_tokens) == 1:
            sentence_tokens[0]["relative_position"] = 1.0
            return
        denom = len(sentence_tokens) - 1
        for idx, tok in enumerate(sentence_tokens):
            tok["relative_position"] = idx / denom if denom else 0.0

    current_sentence: List[Dict[str, Any]] = []
    for token in tokens:
        current_sentence.append(token)
        if token.get("is_sentence_final"):
            assign_relative_positions(current_sentence)
            current_sentence = []
    if current_sentence:
        assign_relative_positions(current_sentence)

# =========================
# Labeling Logic (For Training Mode)
# =========================
def generate_labels_from_cues(tokens: List[Dict[str, Any]], cues: List[Dict[str, Any]], settings: Dict):
    """
    Generates ground-truth break labels for tokens based on SRT cues.

    Tokens that already carry cue metadata (from `tokenize_srt_cues`) keep
    their structural annotations, ensuring that human-authored line breaks are
    respected exactly. Any remaining tokens (typically insertions introduced by
    alignment) fall back to a timestamp-based association that mirrors the
    previous heuristic behaviour.

    Args:
        tokens: The list of timed word dictionaries.
        cues: The list of cue dictionaries read from the ground-truth SRT file.
        settings: A configuration dictionary.
    """
    cue_lookup = {c["id"]: c for c in cues}

    for token in tokens:
        token["break_type"] = token.get("break_type", "O") or "O"

    cue_groups: Dict[int, List[Dict[str, Any]]] = {}
    for token in tokens:
        cue_id = token.get("cue_id")
        if cue_id is None or cue_id == -1:
            continue
        cue_groups.setdefault(int(cue_id), []).append(token)

    for cue_id, cue_tokens in cue_groups.items():
        if not cue_tokens:
            continue
        for token in cue_tokens:
            if token.get("line_break_after"):
                token["break_type"] = "LB"
                token.setdefault("is_llm_structural_break", True)
            elif token.get("break_type") not in {"LB", "SB"}:
                token["break_type"] = "O"
        cue_tokens[-1]["break_type"] = "SB"
        cue_tokens[-1]["is_llm_structural_break"] = False

    unresolved = [token for token in tokens if token.get("cue_id", -1) == -1]
    if unresolved and cues:
        tol = settings.get("time_tolerance_s", 0.15)
        for token in unresolved:
            mid_time = (token["start"] + token["end"]) / 2
            best_cue_id = -1
            min_dist = float('inf')
            for cue in cues:
                if cue["start"] - tol <= mid_time <= cue["end"] + tol:
                    best_cue_id = cue["id"]
                    break
                dist = min(abs(mid_time - cue["start"]), abs(mid_time - cue["end"]))
                if dist < min_dist:
                    min_dist = dist
                    best_cue_id = cue["id"]
            token["cue_id"] = best_cue_id

        fallback_groups: Dict[int, List[Dict[str, Any]]] = {}
        for token in unresolved:
            cue_id = token.get("cue_id", -1)
            fallback_groups.setdefault(int(cue_id), []).append(token)

        for cue_id, cue_tokens in fallback_groups.items():
            if not cue_tokens:
                continue
            for t in cue_tokens:
                if t.get("break_type") not in {"LB", "SB"}:
                    t["break_type"] = "O"

            cue = cue_lookup.get(cue_id)
            if not cue:
                cue_tokens[-1]["break_type"] = "SB"
                continue

            lines = [strip_rendered_markup(line) for line in (cue.get("text") or "").split('\n') if strip_rendered_markup(line).strip()]
            if len(lines) >= 2:
                line1_text = re.sub(r'\s+', ' ', lines[0]).strip()
                line1_char_len = len(line1_text)
                current_char_count = 0
                for i, token in enumerate(cue_tokens):
                    current_char_count += len(token.get("w", "")) + (1 if i > 0 else 0)
                    if current_char_count >= line1_char_len and cue_tokens[i]["break_type"] != "SB":
                        cue_tokens[i]["break_type"] = "LB"
                        break
            cue_tokens[-1]["break_type"] = "SB"

# =========================
# Main Pipeline
# =========================
def process_file(
    primary_path: Path,
    asr_reference_path: Path,
    paths: Dict[str, Path],
    settings: Dict[str, Any],
    *,
    asr_only_mode: bool = False,
    output_basename: Optional[str] = None,
):
    """
    Main processing pipeline for a single file.

    This function orchestrates the entire workflow:
    1.  Loads the ASR reference and the primary input (TXT or SRT).
    2.  Aligns the primary text to the ASR timestamps.
    3.  Corrects speaker labels.
    4.  (Training mode only) Generates ground-truth break labels from SRT cues.
    5.  Engineers all linguistic and heuristic features.
    6.  Saves the final enriched data to a JSON file.

    Args:
        primary_path: Path to the primary input file (.txt or .srt).
        asr_reference_path: Path to the ASR JSON file.
        paths: Dictionary of output paths.
        settings: Configuration dictionary.
    """
    if output_basename:
        base = output_basename
    elif asr_only_mode:
        base = base_of(asr_reference_path)
        if base.endswith(".asr.visual.words.diar"):
            base = base.split(".asr.visual.words.diar", 1)[0]
    else:
        base = base_of(primary_path)
    print(f"\n===== Processing: {base} =====")

    is_training_mode = (not asr_only_mode) and primary_path.suffix.lower() == ".srt"

    print(f"Loading ASR reference from: {asr_reference_path.name}")
    asr_words = load_asr_words(asr_reference_path)

    processed_tokens_with_hints: List[Dict[str, Any]] = []
    alignment_sources: List[Optional[int]] = []
    asr_alignment: List[Optional[int]] = []
    cue_token_metadata: List[Dict[str, Any]] = []
    if asr_only_mode:
        print("Operating in ASR-only mode. Skipping primary text alignment.")
        tokens = [
            {
                "w": str(word.get("w", "")),
                "start": float(word.get("start", 0.0)),
                "end": float(word.get("end", 0.0)),
                "speaker": word.get("speaker"),
            }
            for word in asr_words
        ]
        cues = []
        alignment_sources = [None] * len(tokens)
        asr_alignment = list(range(len(tokens)))
    elif is_training_mode:
        print(f"Loading SRT for training from: {primary_path.name}")
        cues = read_srt_cues(primary_path)
        primary_tokens, cue_token_metadata = tokenize_srt_cues(cues)
        print("Aligning primary text to ASR reference...")
        tokens, alignment_sources, asr_alignment = align_text_to_asr(primary_tokens, asr_words, settings, return_alignment=True)
    else:
        print(f"Loading TXT for inference from: {primary_path.name}")
        raw_text = primary_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()
        for line in lines:
            if not line.strip():
                if processed_tokens_with_hints:
                    processed_tokens_with_hints[-1]["llm_break_hint"] = True
                continue
            words_in_group = line.split()
            if not words_in_group:
                continue
            for word in words_in_group:
                processed_tokens_with_hints.append({"w": word, "llm_break_hint": False})
            processed_tokens_with_hints[-1]["llm_break_hint"] = True
        if processed_tokens_with_hints:
            processed_tokens_with_hints[-1]["llm_break_hint"] = False
        primary_tokens = [d["w"] for d in processed_tokens_with_hints]
        cues = []
        print("Aligning primary text to ASR reference...")
        tokens, alignment_sources, asr_alignment = align_text_to_asr(primary_tokens, asr_words, settings, return_alignment=True)
    if not tokens:
        print("[WARN] Alignment produced no tokens. Skipping.")
        return

    if tokens:
        for idx, token in enumerate(tokens):
            asr_idx = asr_alignment[idx] if idx < len(asr_alignment) else None
            if asr_idx is not None and 0 <= asr_idx < len(asr_words):
                token["asr_source_word"] = str(asr_words[asr_idx].get("w", ""))
            else:
                token["asr_source_word"] = None

    if is_training_mode and cue_token_metadata:
        for idx, token in enumerate(tokens):
            source_idx = alignment_sources[idx] if idx < len(alignment_sources) else None
            if source_idx is not None and 0 <= source_idx < len(cue_token_metadata):
                meta = cue_token_metadata[source_idx]
                token["cue_id"] = meta.get("cue_id")
                token["cue_line_index"] = meta.get("line_index")
                token["line_break_after"] = meta.get("line_break_after", False)
                token["is_last_in_cue"] = meta.get("is_last_in_cue", False)
                token["is_llm_structural_break"] = meta.get("line_break_after", False)
            else:
                token.setdefault("cue_id", -1)
                token["line_break_after"] = False
                token["is_last_in_cue"] = False
                token.setdefault("is_llm_structural_break", False)

    if not is_training_mode and not asr_only_mode:
        if processed_tokens_with_hints:
            for idx, token in enumerate(tokens):
                source_idx = alignment_sources[idx] if idx < len(alignment_sources) else None
                if source_idx is not None and 0 <= source_idx < len(processed_tokens_with_hints):
                    token["is_llm_structural_break"] = processed_tokens_with_hints[source_idx].get("llm_break_hint", False)
                else:
                    token["is_llm_structural_break"] = False
        else:
            print(f"[INFO] No LLM hints found in TXT. Defaulting structural hint flags to False.")
            for token in tokens:
                token["is_llm_structural_break"] = False
    elif asr_only_mode:
        for token in tokens:
            token["is_llm_structural_break"] = False

    correct_speaker_labels(tokens)

    if is_training_mode:
        print("Generating ground-truth labels from SRT cues...")
        generate_labels_from_cues(tokens, cues, settings)

    print("Performing all feature engineering...")
    engineer_features(tokens, settings)
    
    final_tokens = []
    for t in tokens:
        final_tokens.append({
            "w": t.get("w", ""), "start": round(t.get("start", 0.0), settings.get("round_seconds", 3)),
            "end": round(t.get("end", 0.0), settings.get("round_seconds", 3)), "speaker": t.get("speaker"),
            "cue_id": t.get("cue_id"), "is_sentence_initial": t.get("is_sentence_initial", False),
            "is_sentence_final": t.get("is_sentence_final", False), "pause_after_ms": t.get("pause_after_ms", 0),
            "pause_before_ms": t.get("pause_before_ms", 0), "pause_z": t.get("pause_z", 0.0),
            "pos": t.get("pos"), "lemma": t.get("lemma"), "tag": t.get("tag"), "morph": t.get("morph"),
            "dep": t.get("dep"), "head_idx": t.get("head_idx"), "break_type": t.get("break_type"),
            "starts_with_dialogue_dash": t.get("starts_with_dialogue_dash", False),
            "speaker_change": t.get("speaker_change", False), "num_unit_glue": t.get("num_unit_glue", False),
            "is_llm_structural_break": t.get("is_llm_structural_break", False),
            "is_dangling_eos": t.get("is_dangling_eos", False),
            "relative_position": t.get("relative_position", 0.0),
            "asr_source_word": t.get("asr_source_word"),
            "cue_line_index": t.get("cue_line_index"),
            "line_break_after": t.get("line_break_after", False),
            "is_last_in_cue": t.get("is_last_in_cue", False),
        })

    output_obj = {"tokens": final_tokens}
    asr_style_obj: Optional[Dict[str, Any]] = None
    if is_training_mode and settings.get("emit_asr_style_training_copy"):
        asr_tokens = copy.deepcopy(final_tokens)
        for tok in asr_tokens:
            if tok.get("asr_source_word"):
                tok["w"] = str(tok["asr_source_word"])
        asr_style_obj = {"tokens": asr_tokens}

    if is_training_mode:
        out_path = paths["out_training_dir"] / f"{base}.train.words.json"
        _save_json(output_obj, out_path)
        print(f"[OK] Wrote TRAINING data to: {out_path.name}")
        if asr_style_obj:
            asr_path = paths["out_training_dir"] / f"{base}.asrstyle.train.words.json"
            _save_json(asr_style_obj, asr_path)
            print(f"[OK] Wrote ASR-STYLE TRAINING data to: {asr_path.name}")
    else:
        out_path = paths["out_inference_dir"] / f"{base}.enriched.json"
        _save_json(output_obj, out_path)
        print(f"[OK] Wrote INFERENCE data to: {out_path.name}")

def main():
    """
    Main entry point for the command-line interface.

    Parses command-line arguments, loads configuration from YAML files,
    and calls the main `process_file` function to execute the data
    enrichment pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Align, enrich, and label word-level data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--primary-input", required=True, type=Path, help="Path to the primary TXT or SRT file.")
    parser.add_argument("--asr-reference", required=True, type=Path, help="Path to the time-stamped ASR JSON file.")
    parser.add_argument("--out-training-dir", type=Path, help="Override the output directory for training files.")
    parser.add_argument("--out-inference-dir", type=Path, help="Override the output directory for inference files.")
    parser.add_argument("--config-file", type=Path, help="Path to the pipeline_config.yaml file.")
    parser.add_argument("--asr-only-mode", action="store_true", help="Use ASR tokens directly without TXT/SRT alignment.")
    parser.add_argument("--output-basename", type=str, help="Override the output base filename.")
    args = parser.parse_args()

    config = DEFAULT_SETTINGS.copy()
    if args.config_file and args.config_file.exists():
        with open(args.config_file, "r", encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config: config = _recursive_update(config, yaml_config)
    
    path_context = {k: v for k, v in config.items() if isinstance(v, str)}
    config = _resolve_paths(config, path_context)
    SETTINGS = config.get("build_pair", {})

    paths = {
        "out_training_dir": args.out_training_dir or Path(SETTINGS["out_training_dir"]),
        "out_inference_dir": args.out_inference_dir or Path(SETTINGS["out_inference_dir"]),
    }
    ensure_dirs(paths["out_training_dir"]); ensure_dirs(paths["out_inference_dir"])

    if not args.primary_input.exists():
        raise FileNotFoundError(f"Primary input file not found: {args.primary_input}")
    if not args.asr_reference.exists():
        raise FileNotFoundError(f"ASR reference file not found: {args.asr_reference}")

    try:
        asr_only_mode = args.asr_only_mode or args.primary_input.resolve() == args.asr_reference.resolve()
        base_override = args.output_basename.strip() if args.output_basename else None
        process_file(
            args.primary_input,
            args.asr_reference,
            paths,
            SETTINGS,
            asr_only_mode=asr_only_mode,
            output_basename=base_override,
        )
    except Exception:
        print(f"[FAIL] Unhandled error processing {args.primary_input.name}:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()