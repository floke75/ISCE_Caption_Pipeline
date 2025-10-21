#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
build_training_pair_standalone.py — self-contained training pair builder

- Reads a human SRT (timing anchor) and a visual-words JSON/aligned words JSON.
- Writes:
    <base>.train.words.json
    <base>.train.labels.json
- Non-destructive: token text/timings are never changed unless you explicitly enable
  a transform in SETTINGS (defaults are SAFE: all transforms OFF).

What it adds:
  - Prosody inline per token: pause_before_ms, pause_after_ms, pause_z
  - Dialogue-dash cue flag (no token removal)
  - Number–unit glue flag (feature only; does not change tokens)
  - Break labels from SRT newlines: O / LB / LB_HARD / SB
  - Per-token break_type (LB_HARD collapsed to LB)
  - spaCy inline: pos/lemma/tag/morph (if spaCy available)
  - Optional spaCy dependencies: dep, head_i (if enabled and parser is available)
  - Optional vectors sidecar (.npy, float16) + pointer in meta

SRT dash normalization (in-memory ONLY, never touches your file on disk):
  1) Strip leading dash ONLY on the upper/first line IF the next character is lowercase.
  2) Strip trailing dash on the lower/second line OR on a single-line cue.
"""

from __future__ import annotations
import os, json, re, statistics
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# -------------------------------
# SETTINGS (edit here, no CLI/YAML)
# -------------------------------
SETTINGS: Dict[str, Any] = {
    # Required I/O
    "paths": {
        # Full paths or relative to this script
        "srt": r"T:\AI-Subtitles\Training\Training_data\News_Panel\BL_010_240408_Ordkedjan.srt",
        "visual_words": r"T:\AI-Subtitles\Training\Training_data\News_Panel\_align\BL_010_240408_Ordkedjan.visual.words.json",
        "out_dir": r"T:\AI-Subtitles\Training\Training_data\News_Panel\_training",
        # Optional fixed base name for outputs (defaults to SRT stem)
        "base": None,
        # Optional override for vectors sidecar output folder
        "vectors_out": None,  # default: <out_dir>/../_sidecars/spacy_vectors
    },

    # General
    "language": "sv",
    "time_tolerance": 0.15,    # seconds tolerance around cue borders
    "round_seconds": 3,        # decimals for rounding start/end in output
    "prefer_diarized_words": True,
    "strip_end_credit_cue": False,  # your .set stage already strips; keep False by default

    # Dialogue dashes (flags only; do NOT remove tokens)
    "dash_chars": ["-", "–", "—"],
    "strip_dialogue_dash_token": False,  # keep OFF unless you really want to drop a leading dash token

    # SRT dash cleanup rules (in-memory only; recommended ON)
    "srt_dash_cleanup": {
        "enable": True,
        "strip_leading_on_upper_if_lowercase": True,
        "strip_trailing_on_lower_or_single": True,
    },

    # Label schema and LB hardening
    "emit_line_break_labels": True,  # True: O/LB/LB_HARD/SB, False: O/SB only
    "lb_hard": {
        "enable": True,
        "pause_z_thresh": 0.8,       # LB upgraded to LB_HARD if next-token pause_z >= this
        "punctuation_is_hard": True, # punctuation on line end marks LB as hard
        "speaker_change_is_hard": True,
        "punct_set": list(".,!?…:;"),
    },

    # Transforms (ALL OFF by default; enable with caution)
    "enable_split_false_dialogue_hyphens": False,
    "enable_merge_true_hyphenation": False,
    "coalesce_trailing_punct": False,
    "punct_to_coalesce": [",", ".", "!", "?", ":", ";", "…"],
    "strip_token_edges": False,            # trims whitespace-only tokens; can change token counts
    "normalize_nbsp_to_space": False,      # normalizes \u00A0/… to " " (can change char counts)

    # Feature flags (do not change tokens)
    "enable_num_unit_glue": True,
    "num_regex": r"\d+[.,]?\d*",
    "unit_vocab": ["%", "kr", "kronor", "$", "€"],

    # spaCy
    "spacy": {
        "enable": True,                     # turn off to skip spaCy entirely
        "model": "sv_core_news_lg",         # Swedish core model (has parser + vectors)
        "add_dependencies": True,           # dep, head_i
        "write_vectors_npy": True,          # save vectors sidecar
    },
}

# -------------
# Dependencies
# -------------
try:
    import pysrt
except Exception:
    raise SystemExit("[ERROR] pysrt is required. Install with: pip install pysrt")

# spaCy optional (we degrade gracefully)
try:
    import spacy  # type: ignore
    from spacy.tokens import Doc  # type: ignore
except Exception:
    spacy = None
    Doc = None

# numpy optional (only needed for vectors)
try:
    import numpy as np  # type: ignore
except Exception:
    np = None

# -------------------------------
# Small utilities / normalization
# -------------------------------
def _as_set(x) -> set:
    if isinstance(x, set): return x
    if isinstance(x, (list, tuple)): return set(x)
    return set([x])

def morph_to_str(m) -> Optional[str]:
    """Robust string for spaCy MorphAnalysis across versions/models."""
    if not m:
        return None
    try:
        return m.to_string()  # some builds support this
    except Exception:
        try:
            return str(m)     # fallback works broadly in spaCy v3
        except Exception:
            return None

def _label_to_break_type(lbl: str) -> str:
    """Collapse LB_HARD → LB for the inline convenience field."""
    if lbl in ("LB", "LB_HARD"):
        return "LB"
    if lbl == "SB":
        return "SB"
    return "O"

def _looks_like_end_credit(text: str) -> bool:
    """EFN end-credit heuristic: requires both 'EFN'+'Ekonomikanalen' and 'Ansvarig utgivare'."""
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    tl = " ".join(lines).casefold()
    return ("ansvarig utgivare" in tl) and ("efn" in tl) and ("ekonomikanalen" in tl)

def _strip_edges(s: str, normalize_nbsp: bool) -> str:
    """Trim ASCII+NBSP-like spaces at the ends only; keep interior spaces."""
    if not isinstance(s, str):
        s = str(s or "")
    if normalize_nbsp:
        s = s.replace("\u00A0", " ").replace("\u202F", " ").replace("\u2009", " ")
    return re.sub(r'^[\s\u00A0\u202F]+|[\s\u00A0\u202F]+$', '', s)

# -------------------------------
# SRT I/O with dash cleanup rules
# -------------------------------
_DASH_CHARS = "-\u2013\u2014"  # hyphen, en-dash, em-dash

def _strip_leading_dash_if_lowercase(line: str) -> str:
    """
    If the line begins with dash(es)+optional space and the first real char is lowercase alpha,
    remove that dash cluster (and following single space, if any). Else return unchanged.
    """
    if not line:
        return line
    m = re.match(rf'^\s*([{_DASH_CHARS}])\s*([^\s])', line)
    if not m:
        return line
    first = m.group(2)
    if first.isalpha() and first.islower():
        return re.sub(rf'^\s*([{_DASH_CHARS}])\s*', '', line, count=1)
    return line

def _strip_trailing_dash(line: str) -> str:
    """If the line ends with a dash (hyphen/en/em), strip it and trailing spaces."""
    if not line:
        return line
    s = line.rstrip()
    if s and s[-1] in _DASH_CHARS:
        return s[:-1].rstrip()
    return line

def srt_read(path: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    srt = pysrt.open(path, encoding="utf-8")
    cues: List[Dict[str, Any]] = []

    clean_cfg = cfg.get("srt_dash_cleanup") or {}
    do_clean = bool(clean_cfg.get("enable", False))
    do_lead  = bool(clean_cfg.get("strip_leading_on_upper_if_lowercase", False))
    do_tail  = bool(clean_cfg.get("strip_trailing_on_lower_or_single", False))

    for i, it in enumerate(srt):
        raw_text = (it.text or "").replace("\r", "")
        lines = raw_text.splitlines()

        if do_clean and lines:
            # 1) upper line leading dash → only if next char is lowercase
            if do_lead and len(lines) >= 1:
                lines[0] = _strip_leading_dash_if_lowercase(lines[0])

            # 2) trailing dash on lower (second) line OR on single-line cues
            if do_tail:
                if len(lines) == 1:
                    lines[0] = _strip_trailing_dash(lines[0])
                elif len(lines) >= 2:
                    lines[1] = _strip_trailing_dash(lines[1])

        cues.append({
            "id": i,
            "start": it.start.ordinal / 1000.0,
            "end": it.end.ordinal / 1000.0,
            "text": "\n".join(lines)
        })

    if cfg.get("strip_end_credit_cue", True) and cues:
        last_txt = cues[-1]["text"]
        if _looks_like_end_credit(last_txt):
            _ = cues.pop()
            print(f"[CLEAN] Stripped end-credit cue in {os.path.basename(path)}")
    return cues

# -------------------------------
# Visual-words loader & helpers
# -------------------------------
def resolve_visual_words_path(path_str: str, prefer_diarized: bool = True) -> Path:
    """
    If caller points to <base>.visual.words.json, automatically switch to a diarized sibling if present:
      1) <base>.visual.words.diar.json
      2) <base>.diar.json
    Else return the given path as-is.
    """
    p = Path(path_str)
    if not prefer_diarized:
        return p
    name = p.name.lower()
    if name.endswith(".visual.words.diar.json") or name.endswith(".diar.json"):
        return p
    if name.endswith(".visual.words.json"):
        core = p.name[:-len(".visual.words.json")]
        cand1 = p.parent / f"{core}.visual.words.diar.json"
        if cand1.exists():
            return cand1
        cand2 = p.parent / f"{core}.diar.json"
        if cand2.exists():
            return cand2
    return p

def load_visual_words(path: str) -> List[Dict[str, Any]]:
    """
    Accepts:
      - <base>.visual.words.json
      - <base>.visual.words.diar.json
      - <base>.diar.json
    Expects {"words":[{word|text,start,end,speaker?,cue_id?,cue_first?,source?},...]}.
    Returns canonical fields {w,start,end,speaker?,cue_id?,cue_first?,source?} sorted by time.
    """
    obj = json.load(open(path, "r", encoding="utf-8"))
    raw = obj.get("words") or []
    words: List[Dict[str, Any]] = []
    for it in raw:
        if "start" not in it or "end" not in it:
            continue
        w = it.get("word", it.get("text", "")) or ""
        rec: Dict[str, Any] = {
            "w": str(w),
            "start": float(it["start"]),
            "end": float(it["end"]),
        }
        if "speaker" in it: rec["speaker"] = it.get("speaker")
        if "cue_id" in it and it["cue_id"] is not None:
            try: rec["cue_id"] = int(it["cue_id"])
            except Exception: rec["cue_id"] = it["cue_id"]
        if "cue_first" in it: rec["cue_first"] = bool(it.get("cue_first"))
        if "source" in it: rec["source"] = it.get("source")
        words.append(rec)
    words.sort(key=lambda d: (d["start"], d["end"]))
    return words

def token_mid(t: Dict[str, Any]) -> float:
    return 0.5 * (float(t["start"]) + float(t["end"]))

# -------------------------------
# Safe, conservative transforms
# -------------------------------
def strip_cue_initial_dash_from_tokens(tokens: List[Dict[str, Any]], dash_chars: set) -> List[Dict[str, Any]]:
    if not tokens:
        return tokens
    t0 = (tokens[0].get("w") or "").strip()
    if (t0 in dash_chars) or re.fullmatch(r"[\-–—]", t0):
        return tokens[1:]
    return tokens

def split_false_dialogue_hyphens(tokens: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Split tokens like 'väntat-Börslunch' into ['väntat','Börslunch'] based on case pattern."""
    out: List[Dict[str, Any]] = []
    for t in tokens:
        w = str(t.get("w", ""))
        if w.count("-") == 1:
            left, right = w.split("-", 1)
            left_s, right_s = left.strip(), right.strip()
            if left_s and right_s and left_s[-1:].islower() and right_s[:1].isupper():
                st = float(t.get("start", 0.0)); en = float(t.get("end", st))
                dur = max(0.02, en - st)
                ll = max(1, len(left_s)); rl = max(1, len(right_s))
                boundary = st + dur * (ll / (ll + rl))
                t_left = dict(t);  t_left["w"] = left_s;  t_left["end"] = boundary
                t_right = dict(t); t_right["w"] = right_s; t_right["start"] = boundary
                out.extend([t_left, t_right]); continue
        out.append(t)
    return out

def merge_true_hyphenation(tokens: List[Dict[str, Any]], dash_chars: set) -> List[Dict[str, Any]]:
    """Conservatively merge true in-word hyphenations (word - word), never dialogue dashes."""
    def _is_wordy(s: str) -> bool: return bool(re.search(r"\w", s or ""))
    def _starts_upper(s: str) -> bool:
        s = (s or "").strip(); ch = s[:1]
        return bool(ch and ch.isalpha() and ch == ch.upper())
    out: List[Dict[str, Any]] = []
    i = 0
    while i < len(tokens):
        if i + 2 < len(tokens):
            left, dash, right = tokens[i], tokens[i+1], tokens[i+2]
            lw, dw, rw = str(left.get("w", "")), str(dash.get("w", "")), str(right.get("w", ""))
            if (dw in dash_chars) and _is_wordy(lw) and _is_wordy(rw):
                if str(dash.get("source", "")).strip().lower() == "dialogue_dash":
                    pass
                elif _starts_upper(rw):
                    pass
                else:
                    merged = {
                        "w": f"{lw}-{rw}",
                        "start": float(min(float(left.get("start", 1e9)), float(dash.get("start", 1e9)), float(right.get("start", 1e9)))),
                        "end": float(max(float(left.get("end", -1.0)), float(dash.get("end", -1.0)), float(right.get("end", -1.0)))),
                        "speaker": left.get("speaker") or right.get("speaker"),
                    }
                    out.append(merged); i += 3; continue
        out.append(tokens[i]); i += 1
    return out

def coalesce_standalone_punct(tokens: List[Dict[str, Any]], punct_set: set) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for t in tokens:
        if out and (t.get("w") in punct_set):
            out[-1]["end"] = max(float(out[-1]["end"]), float(t["end"]))
        else:
            out.append(t)
    return out

def detect_num_unit_glue(tokens: List[Dict[str, Any]], num_regex: str, unit_vocab: set) -> List[Dict[str, Any]]:
    num_re = re.compile(num_regex)
    for i in range(len(tokens)-1):
        a, b = tokens[i], tokens[i+1]
        if num_re.fullmatch(str(a.get("w", ""))) and str(b.get("w", "")) in unit_vocab:
            a["num_unit_glue"], b["num_unit_glue"] = True, True
    return tokens

def strip_token_edges(tokens: List[Dict[str, Any]], normalize_nbsp: bool) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    dropped = 0
    for t in tokens:
        w = _strip_edges(t.get("w", ""), normalize_nbsp)
        if not w:
            dropped += 1; continue
        if w != t.get("w"):
            t = dict(t); t["w"] = w
        out.append(t)
    if dropped:
        print(f"[CLEAN] Dropped {dropped} whitespace-only tokens")
    return out

# -------------------------------
# Cue mapping / labels / speakers
# -------------------------------
def assign_tokens_to_cues(tokens: List[Dict[str, Any]], cues: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[int]:
    tol = float(cfg.get("time_tolerance", 0.15))
    valid_ids = {c["id"] for c in cues}
    # Prefer embedded cue_id if present and valid; else time-based mapping
    if tokens and all(("cue_id" in t and t["cue_id"] is not None) for t in tokens):
        ids: List[int] = []
        for t in tokens:
            try:
                cid = int(t.get("cue_id"))
            except Exception:
                cid = None
            if cid in valid_ids:
                ids.append(cid); continue
            # fallback to time mapping for this token
            mid = token_mid(t)
            best, best_dist = None, 1e9
            for c in cues:
                st, en = c["start"] - tol, c["end"] + tol
                if st <= mid <= en:
                    best = c["id"]; break
                d = (st - mid) if mid < st else ((mid - en) if mid > en else 0.0)
                if d < best_dist:
                    best_dist, best = d, c["id"]
            ids.append(best if best is not None else (min(valid_ids) if valid_ids else 0))
        return ids

    # General case: by time
    ids = []
    for t in tokens:
        mid = token_mid(t)
        best, best_dist = None, 1e9
        for c in cues:
            st, en = c["start"] - tol, c["end"] + tol
            if st <= mid <= en:
                best = c["id"]; break
            d = (st - mid) if mid < st else ((mid - en) if mid > en else 0.0)
            if d < best_dist:
                best_dist, best = d, c["id"]
        ids.append(best if best is not None else 0)
    return ids

def _cue_lines(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines()]

def _lb_boundary_index_for_cue(tokens_slice: list[dict], line1_len_chars: int) -> Optional[int]:
    """
    Return the index (in tokens_slice) of the last token on line 1,
    using exact character counting of token text with single spaces.
    """
    if not tokens_slice or line1_len_chars <= 0:
        return None
    acc = 0
    for j, t in enumerate(tokens_slice):
        w = str(t.get("w") or "")
        add = len(w)
        if j > 0:
            add += 1  # single space between words
        acc += add
        if acc >= line1_len_chars:
            return j
    return None

def _is_hard_break(tokens: list[dict], global_i_line_end: int, global_i_next: int, cfg: Dict[str, Any]) -> bool:
    """LB→LB_HARD using punctuation, next token's pause_z, and speaker change."""
    opts = cfg.get("lb_hard", {})
    if not opts.get("enable", True):
        return False

    punct_set = _as_set(opts.get("punct_set", set(".,!?…:")))
    hard = False

    # punctuation at the line end
    w_end = str(tokens[global_i_line_end].get("w") or "")
    if opts.get("punctuation_is_hard", True) and (len(w_end) and w_end[-1] in punct_set):
        hard = True

    # high pause before next token (pause_z computed earlier)
    if opts.get("pause_z_thresh", None) is not None and 0 <= global_i_next < len(tokens):
        try:
            if float(tokens[global_i_next].get("pause_z", 0.0)) >= float(opts["pause_z_thresh"]):
                hard = True
        except Exception:
            pass

    # speaker change across the line break
    if opts.get("speaker_change_is_hard", True) and 0 <= global_i_next < len(tokens):
        s1 = tokens[global_i_line_end].get("speaker")
        s2 = tokens[global_i_next].get("speaker")
        if (s1 is not None or s2 is not None) and (s1 != s2):
            hard = True

    return hard

def add_prosody_inline(tokens: List[Dict[str, Any]]) -> None:
    """Compute pause_before_ms, pause_after_ms, and file-level z (from pause_before_ms)."""
    if not tokens:
        return
    tokens.sort(key=lambda t: (float(t["start"]), float(t["end"])))
    n = len(tokens)
    pauses_before: List[float] = []
    for i in range(n):
        if i == 0:
            pb = 0.0
        else:
            pb = max(0.0, float(tokens[i]["start"]) - float(tokens[i-1]["end"]))
        pa = 0.0 if i == n-1 else max(0.0, float(tokens[i+1]["start"]) - float(tokens[i]["end"]))
        tokens[i]["pause_before_ms"] = int(round(pb * 1000))
        tokens[i]["pause_after_ms"]  = int(round(pa * 1000))
        pauses_before.append(pb)
    # z from pause_before_ms
    try:
        mu = statistics.mean(pauses_before)
        sd = statistics.pstdev(pauses_before)
    except statistics.StatisticsError:
        mu, sd = 0.0, 0.0
    for i, x in enumerate(pauses_before):
        z = 0.0 if sd == 0.0 else (x - mu) / sd
        tokens[i]["pause_z"] = round(float(z), 4)

def label_O_LB_SB(tokens: list[dict], cues: list[dict], token_cue_ids: list[int], cfg: Dict[str, Any]) -> list[str]:
    """
    Newline-true LB; SB on the last token of each cue.
    Emits LB_HARD when configured and predicates fire.
    """
    labels = ["O"] * len(tokens)

    # SB at the final token of each cue
    cue_id_set = {c["id"] for c in cues}
    for cid in cue_id_set:
        idx = [i for i, k in enumerate(token_cue_ids) if k == cid]
        if idx:
            labels[idx[-1]] = "SB"

    if not cfg.get("emit_line_break_labels", True):
        return labels

    # LB (and possibly LB_HARD)
    for c in cues:
        lines = _cue_lines(c["text"])
        if len(lines) < 2:
            continue  # single-line cue → no LB

        # gather token indices for this cue
        idx = [i for i, k in enumerate(token_cue_ids) if k == c["id"]]
        if len(idx) < 2:
            continue

        # line1 length (SRT text is already cleaned in srt_read per your rules)
        line1_norm = re.sub(r"\s+", " ", lines[0]).strip()
        lb_local = _lb_boundary_index_for_cue([tokens[i] for i in idx], len(line1_norm))
        if lb_local is None or lb_local >= len(idx) - 1:
            continue  # can't place LB safely

        i_line_end = idx[lb_local]          # global index of last token on line 1
        i_next     = idx[lb_local + 1]      # global index of first token on line 2

        # Don't overwrite SB (LB must be strictly before SB)
        if labels[i_line_end] == "SB":
            if lb_local > 0:
                i_line_end = idx[lb_local - 1]
                i_next     = idx[lb_local]
                if labels[i_line_end] == "SB":
                    continue
            else:
                continue

        hard = _is_hard_break(tokens, i_line_end, i_next, cfg)
        labels[i_line_end] = "LB_HARD" if hard else "LB"

    return labels

def quantify_sentence_flags(tokens: List[Dict[str, Any]], token_cue_ids: List[int]) -> None:
    """Mark first/last token within each cue; best-effort 'sentence' proxies."""
    by_cue: Dict[int, List[int]] = {}
    for i, cid in enumerate(token_cue_ids):
        by_cue.setdefault(cid, []).append(i)
    for inds in by_cue.values():
        if not inds:
            continue
        tokens[inds[0]]["is_sentence_initial"] = True
        tokens[inds[-1]]["is_sentence_final"] = True

# -------------------------------
# Builder core
# -------------------------------
def build_training_pair(
    srt_path: str,
    visual_words_path: str,
    out_dir: str,
    base: Optional[str],
    cfg: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Returns: (words_path, labels_path)
    """
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)
    base = base or os.path.splitext(os.path.basename(srt_path))[0]

    # Prefer diarized words file if present
    resolved_vw = resolve_visual_words_path(
        visual_words_path, prefer_diarized=bool(cfg.get("prefer_diarized_words", True))
    )
    if not Path(resolved_vw).exists():
        raise SystemExit(f"[ERROR] visual-words file not found: {resolved_vw}")

    # Load sources
    cues  = srt_read(srt_path, cfg)
    words = load_visual_words(str(resolved_vw))

    # --- Step 0: initial mapping on ORIGINAL tokens (stable reference)
    cue_ids0 = assign_tokens_to_cues(words, cues, cfg)

    # Per-cue index lists on ORIGINAL tokens (for dash detect & majority speaker)
    by_cue_idx0: Dict[int, List[int]] = {}
    for i, cid in enumerate(cue_ids0):
        by_cue_idx0.setdefault(cid, []).append(i)

    # Dialogue dash flag per cue from ORIGINAL tokens; fallback to SRT text heuristic
    dash_chars = _as_set(cfg.get("dash_chars", ["-"]))
    cue_dash: Dict[int, bool] = {}
    for cid, inds in by_cue_idx0.items():
        if not inds:
            cue_dash[cid] = False
            continue
        # head token of the cue
        head_i = next((i for i in inds if words[i].get("cue_first")), None)
        if head_i is None:
            head_i = min(inds, key=lambda k: (float(words[k].get("start", 1e9)), k))
        w0  = str(words[head_i].get("w", "")).strip()
        src = str(words[head_i].get("source", "")).strip().lower()
        is_dash = (w0 in dash_chars) or (re.fullmatch(r"[\-–—]\s*", w0) is not None) or (src == "dialogue_dash")
        cue_dash[cid] = bool(is_dash)

    # --- Step 1: working copy we transform conservatively for TRAINING
    train_tokens = [dict(t) for t in words]

    # Optionally remove a cue-initial dash token
    if cfg.get("strip_dialogue_dash_token", False):
        new_tokens: List[Dict[str, Any]] = []
        new_cue_ids: List[int] = []
        for cid in sorted(by_cue_idx0.keys()):
            inds = by_cue_idx0[cid]
            if not inds:
                continue
            slice_tokens = [train_tokens[i] for i in inds]
            if cue_dash.get(cid, False):
                slice_tokens = strip_cue_initial_dash_from_tokens(slice_tokens, dash_chars)
            new_tokens.extend(slice_tokens)
            new_cue_ids.extend([cid] * len(slice_tokens))
        train_tokens = new_tokens
        cue_ids = new_cue_ids
    else:
        cue_ids = list(cue_ids0)  # unchanged

    # --- Step 2: structural transforms (OFF by default; enable with care)
    if cfg.get("enable_split_false_dialogue_hyphens", False):
        train_tokens = split_false_dialogue_hyphens(train_tokens)
    if cfg.get("enable_merge_true_hyphenation", False):
        train_tokens = merge_true_hyphenation(train_tokens, dash_chars)
    if cfg.get("coalesce_trailing_punct", False):
        punct_set = _as_set(cfg.get("punct_to_coalesce", []))
        train_tokens = coalesce_standalone_punct(train_tokens, punct_set)
    if cfg.get("strip_token_edges", False):
        train_tokens = strip_token_edges(train_tokens, normalize_nbsp=bool(cfg.get("normalize_nbsp_to_space", False)))

    # Feature-only: number–unit glue
    if cfg.get("enable_num_unit_glue", True):
        unit_vocab = _as_set(cfg.get("unit_vocab", {"%"}))
        train_tokens = detect_num_unit_glue(train_tokens, cfg.get("num_regex", r"\d+"), unit_vocab)

    # IMPORTANT: Recompute cue_ids AFTER any structural changes (no-ops by default)
    cue_ids = assign_tokens_to_cues(train_tokens, cues, cfg)

    # --- Step 3: features & speakers on FINAL tokens
    by_cue_idx: Dict[int, List[int]] = {}
    for i, cid in enumerate(cue_ids):
        by_cue_idx.setdefault(cid, []).append(i)

    # Mark dialogue-dash feature per cue
    for cid, inds in by_cue_idx.items():
        is_dash_cue = bool(cue_dash.get(cid, False))
        for i in inds:
            train_tokens[i]["dialogue_dash_cue"] = is_dash_cue

    # Majority speaker per cue stamped onto final tokens
    def _majority_speaker_for_cid(cid: int) -> Optional[str]:
        inds = by_cue_idx0.get(cid, [])
        counts: Dict[str, int] = {}
        for k in inds:
            spk = words[k].get("speaker")
            if spk:
                counts[spk] = counts.get(spk, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    cue_speaker: Dict[int, Optional[str]] = {cid: _majority_speaker_for_cid(cid) for cid in by_cue_idx0.keys()}
    for i, cid in enumerate(cue_ids):
        spk = cue_speaker.get(cid)
        if spk:
            train_tokens[i]["speaker"] = spk

    # Sentence flags within each cue (proxy)
    quantify_sentence_flags(train_tokens, cue_ids)

    # Prosody inline (pauses + z)
    add_prosody_inline(train_tokens)

    # --- (Optional) spaCy inline + vectors
    spacy_block = {}
    spcfg = SETTINGS.get("spacy") or {}
    if spacy is not None and bool(spcfg.get("enable", True)):
        try:
            spacy_model = spcfg.get("model", "sv_core_news_lg")
            want_deps   = bool(spcfg.get("add_dependencies", False))
            write_vec   = bool(spcfg.get("write_vectors_npy", True))

            nlp = spacy.load(spacy_model)
            words_text = [(t.get("w") or "") for t in train_tokens]
            doc = Doc(nlp.vocab, words=words_text)

            # Disable only heavy/unneeded pipes; keep parser if dependencies requested
            to_disable = [p for p in ("ner", "senter", "textcat") if p in nlp.pipe_names]
            if (not want_deps) and ("parser" in nlp.pipe_names):
                to_disable.append("parser")

            with nlp.select_pipes(disable=to_disable):
                doc = nlp(doc)  # runs tok2vec + tagger/morphologizer (+ parser if kept)

            added_fields = ["pos", "lemma", "tag", "morph"]
            for i, tok in enumerate(doc):
                tgt = train_tokens[i]
                tgt["pos"]   = tok.pos_ or None
                tgt["tag"]   = tok.tag_ or None
                tgt["lemma"] = tok.lemma_ or None
                tgt["morph"] = morph_to_str(tok.morph)

                if want_deps and ("parser" in nlp.pipe_names):
                    try:
                        tgt["dep"]    = tok.dep_ or None
                        tgt["head_i"] = int(tok.head.i)
                        if "dep" not in added_fields:    added_fields.append("dep")
                        if "head_i" not in added_fields: added_fields.append("head_i")
                    except Exception:
                        pass

            vec_info = {}
            # Save vectors if available
            has_vectors = bool(getattr(nlp.vocab, "vectors", None)) and getattr(nlp.vocab.vectors, "shape", [0,0])[1] > 0
            if write_vec and has_vectors and np is not None:
                vec_dim = int(nlp.vocab.vectors.shape[1])
                mat = np.vstack([
                    tok.vector if getattr(tok, "has_vector", False) else np.zeros(vec_dim, dtype="float32")
                    for tok in doc
                ]).astype("float16", copy=False)

                vectors_root = (Path(out_dir).parent / "_sidecars" / "spacy_vectors") if (SETTINGS["paths"].get("vectors_out") is None) else Path(SETTINGS["paths"]["vectors_out"])
                vectors_root.mkdir(parents=True, exist_ok=True)
                base_plain = Path(base).name
                out_npy = vectors_root / f"{base_plain}.spacy.vec.f16.npy"
                np.save(out_npy, mat)

                vec_info = {"file": str(out_npy.resolve()), "dtype": "float16", "dim": vec_dim, "n": int(mat.shape[0])}

            spacy_block = {
                "model": spacy_model,
                "version": nlp.meta.get("version"),
                "inline_fields": added_fields,
                "vectors": vec_info,
            }
        except Exception as e:
            print(f"[WARN] spaCy inline disabled due to: {e}")
            spacy_block = {"model": spcfg.get("model", "sv_core_news_lg"), "inline_fields": [], "vectors": {}}
    elif spacy is None and (SETTINGS.get("spacy") or {}).get("enable", True):
        print("[WARN] spaCy not available; skipping NLP inline.")

    # Labels
    labels = label_O_LB_SB(train_tokens, cues, cue_ids, cfg)

    # --- Emit training pair
    rnd = int(cfg.get("round_seconds", 3))
    out_words = {
        "tokens": [
            {
                "w": t["w"],
                "start": round(float(t["start"]), rnd),
                "end":   round(float(t["end"]),   rnd),
                "cue_id": int(cue_ids[i]),
                "speaker": t.get("speaker"),

                # Inline features
                "dialogue_dash_cue": bool(t.get("dialogue_dash_cue", False)),
                "is_sentence_initial": bool(t.get("is_sentence_initial", False)),
                "is_sentence_final":   bool(t.get("is_sentence_final", False)),
                "num_unit_glue": bool(t.get("num_unit_glue", False)),

                # Prosody
                "pause_before_ms": int(t.get("pause_before_ms", 0)),
                "pause_after_ms":  int(t.get("pause_after_ms", 0)),
                "pause_z":         float(t.get("pause_z", 0.0)),

                # spaCy light fields (None if missing)
                "pos":   t.get("pos"),
                "lemma": t.get("lemma"),
                "tag":   t.get("tag"),
                "morph": t.get("morph"),

                # Optional deps
                "dep":    t.get("dep"),
                "head_i": t.get("head_i"),

                # NEW: collapsed per-token break marker (LB_HARD → LB)
                "break_type": _label_to_break_type(labels[i]),
            }
            for i, t in enumerate(train_tokens)
        ],
        "meta": {
            "language": cfg.get("language", "sv"),
            "source_visual_words": os.path.abspath(str(resolved_vw)),
            "source_srt": os.path.abspath(srt_path),
            "source_srt_for_align": os.path.abspath(srt_path),
            "counts": {"orig_tokens": len(words), "final_tokens": len(train_tokens)},
            "prosody": {"source": "gaps_from_alignment", "version": 1, "units": "ms", "z_from": "pause_before_ms"},
            "srt_cleanup": {
                "applied": bool((cfg.get("srt_dash_cleanup") or {}).get("enable", False)),
                "strip_leading_on_upper_if_lowercase": bool((cfg.get("srt_dash_cleanup") or {}).get("strip_leading_on_upper_if_lowercase", False)),
                "strip_trailing_on_lower_or_single": bool((cfg.get("srt_dash_cleanup") or {}).get("strip_trailing_on_lower_or_single", False)),
            }
        }
    }
    if spacy_block:
        out_words["meta"]["spacy"] = spacy_block

    # Always emit 4-label schema by default; allow opting out to O/SB via cfg
    emit_4 = bool(cfg.get("emit_line_break_labels", True))
    out_labels = {
        "labels": labels,
        "schema": ("O/LB/LB_HARD/SB" if emit_4 else "O/SB")
    }

    words_path  = os.path.join(out_dir, f"{base}.train.words.json")
    labels_path = os.path.join(out_dir, f"{base}.train.labels.json")
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(out_words, f, ensure_ascii=False, indent=2)
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(out_labels, f, ensure_ascii=False, indent=2)

    print(f"[BUILD] Wrote:\n  - {words_path}\n  - {labels_path}")
    print(f"[BUILD] orig={len(words)} → final={len(train_tokens)} tokens; cues={len(cues)}")
    return words_path, labels_path

# -------------------------------
# Entrypoint (no CLI)
# -------------------------------
def main():
    p = SETTINGS["paths"]
    cfg = dict(SETTINGS)  # shallow copy is fine (we don't mutate nested dicts)
    srt_path = p["srt"]; visual_words = p["visual_words"]; out_dir = p["out_dir"]; base = p.get("base") or None

    # Normalize a few container types from SETTINGS
    cfg["dash_chars"] = list(cfg.get("dash_chars", ["-"]))
    cfg["punct_to_coalesce"] = list(cfg.get("punct_to_coalesce", []))
    cfg["unit_vocab"] = list(cfg.get("unit_vocab", ["%"]))

    build_training_pair(srt_path, visual_words, out_dir, base, cfg)

if __name__ == "__main__":
    main()
