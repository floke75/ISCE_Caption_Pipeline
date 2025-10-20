# C:\dev\Captions_Formatter\Formatter_machine\isce\srt_writer.py

from __future__ import annotations
from typing import List
from .types import Token

def format_time(s: float) -> str:
    """Converts a time in seconds to the SRT format HH:MM:SS,ms."""
    s = max(0, s)
    total_ms = int(round(s * 1000))
    h, remainder_ms = divmod(total_ms, 3_600_000)
    m, remainder_ms = divmod(remainder_ms, 60_000)
    sec, ms = divmod(remainder_ms, 1000)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"

def tokens_to_srt(tokens: List[Token]) -> str:
    """
    Converts a list of tokens with break_type decisions into a valid SRT formatted string.
    """
    if not tokens:
        return ""

    output_blocks = []
    block_idx = 1
    current_token_idx = 0

    while current_token_idx < len(tokens):
        block_start_idx = current_token_idx
        
        block_end_idx = -1
        for i in range(block_start_idx, len(tokens)):
            if tokens[i].break_type == "SB":
                block_end_idx = i
                break
        
        if block_end_idx == -1:
            block_end_idx = len(tokens) - 1

        if block_end_idx < block_start_idx:
            break

        block_tokens = tokens[block_start_idx : block_end_idx + 1]
        if not block_tokens:
            break

        start_time_str = format_time(block_tokens[0].start)
        end_time_str = format_time(block_tokens[-1].end)
        timestamp_line = f"{start_time_str} --> {end_time_str}"

        line_break_idx = -1
        for i, token in enumerate(block_tokens):
            if token.break_type == "LB":
                line_break_idx = i
                break
        
        def build_text(token_slice: List[Token]) -> str:
            return " ".join(t.w for t in token_slice)

        if line_break_idx != -1:
            line1 = build_text(block_tokens[:line_break_idx + 1])
            line2 = build_text(block_tokens[line_break_idx + 1:])
            text_content = f"{line1}\n{line2}"
        else:
            text_content = build_text(block_tokens)

        srt_block_parts = [
            str(block_idx),
            timestamp_line,
            text_content
        ]
        output_blocks.append("\n".join(srt_block_parts))

        block_idx += 1
        current_token_idx = block_end_idx + 1

    return "\n\n".join(output_blocks)```