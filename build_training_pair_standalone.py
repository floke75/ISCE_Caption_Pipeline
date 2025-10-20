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
    }
}

# =========================
# Configuration Helper Functions (Self-Contained)
# =========================
def _recursive_update(base: Dict, update: Dict) -> Dict:
    for k, v in update.items():
        if isinstance(v, dict) and k in base and isinstance(base[k], dict):
            base[k] = _recursive_update(base[k], v)
        else:
            base[k] = v
    return base

def _resolve_paths(config: Dict, context: Dict) -> Dict:
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
    p.mkdir(parents=True, exist_ok=True)

def base_of(path: Path) -> str:
    return path.stem

def _save_json(obj: dict, p: Path):
    ensure_dirs(p.parent)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def morph_to_str(m) -> Optional[str]:
    if not m: return None
    try: return m.to_string()
    except Exception: return str(m)

# =========================
# Migrated TXT/SRT Alignment Logic (PERFORMANCE OPTIMIZED)
# =========================
_PUNCT_EDGES_RE = re.compile(r"^\W+|\W+$", flags=re.UNICODE)

def _norm_token(s: str) -> str:
    s2 = unicodedata.normalize("NFKC", s).casefold()
    s2 = _PUNCT_EDGES_RE.sub("", s2)
    return re.sub(r"\s+", "", s2)

def _match_score(a: str, b: str, settings: Dict) -> int:
    na, nb = _norm_token(a), _norm_token(b)
    if not na and not nb: return 1
    if na == nb: return 4
    r = fuzz.ratio(na, nb) / 100.0
    if r >= settings.get("txt_match_close", 0.82): return 2
    if r >= settings.get("txt_match_weak", 0.65): return 0
    return -3

def _global_align(a_tokens: List[str], b_tokens: List[str], settings: Dict, gap_penalty: int = -3) -> List[Tuple[Optional[int], Optional[int]]]:
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

def align_text_to_asr(edited_tokens: List[str], asr_words: List[Dict[str, Any]], settings: Dict) -> List[Dict[str, Any]]:
    asr_token_texts = [str(w.get("w") or "") for w in asr_words]
    if not edited_tokens: return []
    if not asr_token_texts:
        slices = _safe_interval_split(0.0, 0.01 * len(edited_tokens), len(edited_tokens))
        return [{"w": t, "start": s, "end": e, "speaker": None} for t, (s, e) in zip(edited_tokens, slices)]

    path = _global_align(edited_tokens, asr_token_texts, settings)
    
    print(f"[DEBUG] Reconstructing aligned words from path ({len(path)} steps)...", flush=True)
    out: List[Dict[str, Any]] = []
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
                last_time = en
            idx = k
        else:
            idx += 1
    for u in range(1, len(out)):
        if out[u]["start"] < out[u-1]["end"]: out[u]["start"] = out[u-1]["end"]
        if out[u]["end"] < out[u]["start"]: out[u]["end"] = out[u]["start"]
    print(f"[DEBUG] Finished reconstructing {len(out)} words.", flush=True)
    return out

def read_srt_cues(path: Path) -> List[Dict[str, Any]]:
    srt = pysrt.open(str(path), encoding="utf-8")
    return [{"id": i, "start": c.start.ordinal/1000.0, "end": c.end.ordinal/1000.0, "text": (c.text or "").replace("\r", "")} for i, c in enumerate(srt)]

def tokenize_srt_cues(cues: List[Dict[str, Any]]) -> Tuple[List[str], List[int]]:
    tokens, cue_ids = [], []
    for c in cues:
        text = c["text"]
        parts = [p for p in text.split() if p]
        for t in parts:
            tokens.append(t)
            cue_ids.append(int(c["id"]))
    return tokens, cue_ids

# =========================
# Data Loading
# =========================
def load_asr_words(path: Path) -> List[Dict[str, Any]]:
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
    if not tokens: return
    # Pass 1: Prosody Features
    for i in range(len(tokens)):
        prev_end = tokens[i-1]["end"] if i > 0 else tokens[i]["start"]
        pause_after = max(0.0, tokens[i+1]["start"] - tokens[i]["end"]) if i < len(tokens) - 1 else 0.0
        tokens[i]["pause_after_ms"] = int(round(pause_after * 1000))

    # Pass 2: SpaCy Linguistic Features
    if settings.get("spacy_enable") and spacy:
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
        except OSError:
            print(f"[ERROR] SpaCy model '{settings['spacy_model']}' not found.")
    
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
        is_unit = str(nxt.get("w", "").lower() in unit_vocab if nxt else False
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

# =========================
# Labeling Logic (For Training Mode)
# =========================
def generate_labels_from_cues(tokens: List[Dict[str, Any]], cues: List[Dict[str, Any]], settings: Dict):
    tol = settings.get("time_tolerance_s", 0.15)
    for token in tokens:
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

    cue_to_tokens = {}
    for token in tokens:
        cue_id = token.get("cue_id", -1)
        if cue_id not in cue_to_tokens: cue_to_tokens[cue_id] = []
        cue_to_tokens[cue_id].append(token)

    for cue_id, cue_tokens in cue_to_tokens.items():
        if not cue_tokens: continue
        for t in cue_tokens: t["break_type"] = "O"
        cue_tokens[-1]["break_type"] = "SB"
        
        cue = next((c for c in cues if c["id"] == cue_id), None)
        if not cue: continue
        
        lines = [line.strip() for line in cue["text"].split('\n') if line.strip()]
        if len(lines) < 2: continue

        line1_text = re.sub(r'\s+', ' ', lines[0]).strip()
        line1_char_len = len(line1_text)
        
        current_char_count = 0
        for i, token in enumerate(cue_tokens):
            current_char_count += len(token["w"]) + (1 if i > 0 else 0)
            if current_char_count >= line1_char_len:
                if token["break_type"] != "SB":
                    token["break_type"] = "LB"
                break

# =========================
# Main Pipeline
# =========================
def process_file(primary_path: Path, asr_reference_path: Path, paths: Dict[str, Path], settings: Dict[str, Any]):
    base = base_of(primary_path)
    print(f"\n===== Processing: {base} =====")

    is_training_mode = primary_path.suffix.lower() == ".srt"
    
    print(f"Loading ASR reference from: {asr_reference_path.name}")
    asr_words = load_asr_words(asr_reference_path)

    processed_tokens_with_hints = []
    if is_training_mode:
        print(f"Loading SRT for training from: {primary_path.name}")
        cues = read_srt_cues(primary_path)
        primary_tokens, _ = tokenize_srt_cues(cues)
    else:
        print(f"Loading TXT for inference from: {primary_path.name}")
        raw_text = primary_path.read_text(encoding="utf-8")
        hint_groups = raw_text.strip().split('\n')
        for group in hint_groups:
            words_in_group = group.split()
            if not words_in_group: continue
            for i, word in enumerate(words_in_group):
                is_hinted_break = (i == len(words_in_group) - 1)
                processed_tokens_with_hints.append({"w": word, "llm_break_hint": is_hinted_break})
        if processed_tokens_with_hints:
            processed_tokens_with_hints[-1]["llm_break_hint"] = False
        primary_tokens = [d["w"] for d in processed_tokens_with_hints]
        cues = []

    print("Aligning primary text to ASR reference...")
    tokens = align_text_to_asr(primary_tokens, asr_words, settings)
    if not tokens:
        print("[WARN] Alignment produced no tokens. Skipping.")
        return

    if not is_training_mode:
        if len(tokens) == len(processed_tokens_with_hints):
            for i, token in enumerate(tokens):
                token["is_llm_structural_break"] = processed_tokens_with_hints[i].get("llm_break_hint", False)
        else:
            print(f"[WARN] Token count mismatch after alignment. Cannot apply LLM break hints.")

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
            "is_llm_structural_break": t.get("is_llm_structural_break", False)
        })

    output_obj = {"tokens": final_tokens}
    
    if is_training_mode:
        out_path = paths["out_training_dir"] / f"{base}.train.words.json"
        _save_json(output_obj, out_path)
        print(f"[OK] Wrote TRAINING data to: {out_path.name}")
    else:
        out_path = paths["out_inference_dir"] / f"{base}.enriched.json"
        _save_json(output_obj, out_path)
        print(f"[OK] Wrote INFERENCE data to: {out_path.name}")

def main():
    parser = argparse.ArgumentParser(
        description="Align, enrich, and label word-level data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--primary-input", required=True, type=Path, help="Path to the primary TXT or SRT file.")
    parser.add_argument("--asr-reference", required=True, type=Path, help="Path to the time-stamped ASR JSON file.")
    parser.add_argument("--out-training-dir", type=Path, help="Override the output directory for training files.")
    parser.add_argument("--out-inference-dir", type=Path, help="Override the output directory for inference files.")
    parser.add_argument("--config-file", type=Path, help="Path to the pipeline_config.yaml file.")
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
        process_file(args.primary_input, args.asr_reference, paths, SETTINGS)
    except Exception:
        print(f"[FAIL] Unhandled error processing {args.primary_input.name}:\n{traceback.format_exc()}")

if __name__ == "__main__":
    main()