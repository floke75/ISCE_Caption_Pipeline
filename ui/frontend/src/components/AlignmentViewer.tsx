import React, { useMemo, useState, useRef } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import client from '../api/client';
import '../styles/alignment-viewer.css';

// Types for the JSON data
interface Token {
  w: string;
  start: number;
  end: number;
  cue_id?: number;
  is_llm_structural_break?: boolean;
  is_sentence_final?: boolean;
}

interface TokenListJson {
  tokens: Token[];
}

interface AsrWord {
  w: string;
  start: number;
  end: number;
  speaker?: string;
  score?: number;
}

interface AsrVisualWordsJson {
  words: AsrWord[];
}

interface CueGroup {
  id: number;
  text: string;
  start: number;
  end: number;
  tokens: Token[];
  isInference?: boolean;
}

const PIXELS_PER_SECOND = 60;

function useFileContent<T>(path: string | null) {
  return useQuery({
    queryKey: ['fileContent', path],
    queryFn: async () => {
      if (!path) return null;
      const response = await client.get(`/files/content`, {
        params: { path, limit: 5000000 }, // Increase limit for large JSONs
      });
      return JSON.parse(response.data.content) as T;
    },
    enabled: !!path,
  });
}

function groupTokensByCue(tokens: Token[]): CueGroup[] {
  const groups: Map<number, CueGroup> = new Map();

  tokens.forEach((token) => {
    if (token.cue_id === undefined || token.cue_id === -1) return;

    if (!groups.has(token.cue_id)) {
      groups.set(token.cue_id, {
        id: token.cue_id,
        text: '',
        start: token.start,
        end: token.end,
        tokens: [],
      });
    }

    const group = groups.get(token.cue_id)!;
    group.tokens.push(token);
    // Update bounds
    group.start = Math.min(group.start, token.start);
    group.end = Math.max(group.end, token.end);
  });

  // Second pass to reconstruct text
  for (const group of groups.values()) {
    group.text = group.tokens.map((t) => t.w).join(' ');
  }

  return Array.from(groups.values()).sort((a, b) => a.id - b.id);
}

function groupTokensForInference(tokens: Token[], mode: 'lines' | 'sentences'): CueGroup[] {
  const groups: CueGroup[] = [];
  let currentGroup: Token[] = [];
  let groupId = 1;

  tokens.forEach((token, idx) => {
    currentGroup.push(token);

    let shouldBreak = false;
    if (mode === 'lines') {
      shouldBreak = !!token.is_llm_structural_break;
    } else {
      shouldBreak = !!token.is_sentence_final;
    }

    // Force break on last token if not already broken
    if (idx === tokens.length - 1) shouldBreak = true;

    if (shouldBreak) {
        if (currentGroup.length > 0) {
            const start = currentGroup[0].start;
            const end = currentGroup[currentGroup.length - 1].end;
            // Note: We don't reconstruct text here because we render tokens individually
            // to show structural hints. But we populate 'text' for fallback.
            const text = currentGroup.map(t => t.w).join(' ');

            groups.push({
                id: groupId++,
                text,
                start,
                end,
                tokens: [...currentGroup],
                isInference: true
            });
            currentGroup = [];
        }
    }
  });
  return groups;
}

export function AlignmentViewer() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const trainPath = searchParams.get('train');
  const inferencePath = searchParams.get('inference');
  const asrPath = searchParams.get('asr');

  const mode = inferencePath ? 'inference' : 'training';
  const leftPath = inferencePath || trainPath;

  const { data: leftData, isLoading: leftLoading, error: leftError } = useFileContent<TokenListJson>(leftPath);
  const { data: asrData, isLoading: asrLoading, error: asrError } = useFileContent<AsrVisualWordsJson>(asrPath);

  const [zoom, setZoom] = useState(PIXELS_PER_SECOND);
  const [groupingMode, setGroupingMode] = useState<'lines' | 'sentences'>('lines');
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const cueGroups = useMemo(() => {
    if (!leftData?.tokens) return [];
    if (mode === 'inference') {
        return groupTokensForInference(leftData.tokens, groupingMode);
    }
    return groupTokensByCue(leftData.tokens);
  }, [leftData, mode, groupingMode]);

  const maxTime = useMemo(() => {
    const maxCue = cueGroups.length ? cueGroups[cueGroups.length - 1].end : 0;
    const maxAsr = asrData?.words.length ? asrData.words[asrData.words.length - 1].end : 0;
    return Math.max(maxCue, maxAsr, 10); // Minimum 10s
  }, [cueGroups, asrData]);

  const containerHeight = Math.ceil(maxTime * zoom) + 200; // Extra padding

  // Generate time markers
  const markers = useMemo(() => {
    const count = Math.ceil(maxTime);
    return Array.from({ length: count + 1 }, (_, i) => i);
  }, [maxTime]);

  if (!leftPath || !asrPath) {
    return (
      <div className="alignment-error">
        <h3>Alignment artifacts missing</h3>
        <p>Provide <code>train</code> (or <code>inference</code>) and <code>asr</code> query params.</p>
        <div className="alignment-actions">
          <button type="button" className="action-button" onClick={() => navigate(-1)}>
            ← Back
          </button>
          <Link to="/" className="action-button primary" style={{ textDecoration: 'none' }}>
            Go to dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (leftLoading || asrLoading) {
    return (
      <div className="alignment-loading">
        <div className="spinner"></div>
        <p>Loading alignment data...</p>
      </div>
    );
  }

  if (leftError || asrError) {
    return (
      <div className="alignment-error">
        <h3>Error loading artifacts</h3>
        <p>Please ensure both the token JSON and ASR JSON are available.</p>
        <pre>{JSON.stringify(leftError || asrError, null, 2)}</pre>
      </div>
    );
  }

  if (!cueGroups.length && !asrData?.words?.length) {
    return (
      <div className="alignment-error">
        <h3>Artifacts loaded but empty</h3>
        <p>The provided files do not contain data to visualise.</p>
        <div className="alignment-actions">
          <button type="button" className="action-button" onClick={() => navigate(-1)}>
            ← Back
          </button>
          <Link to="/" className="action-button primary" style={{ textDecoration: 'none' }}>
            Go to dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="alignment-viewer">
      <header className="alignment-header">
        <div>
          <h1>{mode === 'inference' ? 'Inference Alignment (Stage 2)' : 'Training Alignment'}</h1>
          <p className="subtext">
            Comparing <strong>{mode === 'inference' ? 'Input Text' : 'Edited Subtitles'}</strong> (Left) vs. <strong>Raw ASR</strong> (Right)
          </p>
        </div>
        <div className="alignment-controls">
           {mode === 'inference' && (
             <div className="toggle-group">
               <button
                 className={`toggle-btn ${groupingMode === 'lines' ? 'active' : ''}`}
                 onClick={() => setGroupingMode('lines')}
               >
                 Input Lines
               </button>
               <button
                 className={`toggle-btn ${groupingMode === 'sentences' ? 'active' : ''}`}
                 onClick={() => setGroupingMode('sentences')}
               >
                 Sentences
               </button>
             </div>
           )}

           <label className="zoom-control">
              <span>Zoom</span>
              <input
                type="range"
                min="20"
                max="200"
                step="10"
                value={zoom}
                onChange={(e) => setZoom(Number(e.target.value))}
              />
              <span className="zoom-value">{zoom}</span>
           </label>
           <button type="button" className="action-button small" onClick={() => navigate(-1)}>
             Exit
           </button>
        </div>
      </header>

      <div className="alignment-body" ref={scrollContainerRef}>
        <div className="teleprompter-track" style={{ height: containerHeight }}>
          {/* Time Axis */}
          <div className="time-axis">
            <div className="time-line"></div>
            {markers.map((sec) => (
              <div
                key={sec}
                className="time-marker"
                style={{ top: sec * zoom }}
              >
                {sec}s
              </div>
            ))}
          </div>

          {/* Left Column: Edited Cues / Input Tokens */}
          <div className="column left-column">
            {cueGroups.map((cue) => (
              <div
                key={cue.id}
                className={`cue-block ${cue.isInference ? 'inference' : ''}`}
                style={{
                  top: cue.start * zoom,
                  height: Math.max(24, (cue.end - cue.start) * zoom),
                }}
              >
                <span className="cue-id">{mode === 'inference' ? (groupingMode === 'lines' ? 'LINE' : 'SENT') : 'CUE'} #{cue.id}</span>
                <p className="cue-text">
                    {mode === 'inference' ? (
                        cue.tokens.map((t, i) => (
                            <React.Fragment key={i}>
                                {t.w}
                                {t.is_llm_structural_break && <span className="hint-icon" title="Input structural break">↵</span>}
                                {' '}
                            </React.Fragment>
                        ))
                    ) : (
                        cue.text
                    )}
                </p>
                <div className="cue-meta">
                  {cue.start.toFixed(2)} - {cue.end.toFixed(2)}
                </div>
              </div>
            ))}
          </div>

          {/* Spacer */}
          <div className="column-spacer"></div>

          {/* Right Column: ASR Words */}
          <div className="column right-column">
            {asrData?.words.map((word, idx) => (
              <div
                key={idx}
                className="asr-word"
                style={{
                  top: word.start * zoom,
                  height: Math.max(20, (word.end - word.start) * zoom),
                }}
                title={`${word.start.toFixed(2)}s - ${word.end.toFixed(2)}s`}
              >
                {word.w}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
