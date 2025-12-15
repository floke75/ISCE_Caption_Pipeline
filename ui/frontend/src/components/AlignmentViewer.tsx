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
}

interface TrainWordsJson {
  tokens: Token[];
}

interface AsrWord {
  w: string;
  start: number;
  end: number;
  speaker?: string;
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

export function AlignmentViewer() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const trainPath = searchParams.get('train');
  const asrPath = searchParams.get('asr');

  const { data: trainData, isLoading: trainLoading, error: trainError } = useFileContent<TrainWordsJson>(trainPath);
  const { data: asrData, isLoading: asrLoading, error: asrError } = useFileContent<AsrVisualWordsJson>(asrPath);

  const [zoom] = useState(PIXELS_PER_SECOND);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  const cueGroups = useMemo(() => {
    if (!trainData?.tokens) return [];
    return groupTokensByCue(trainData.tokens);
  }, [trainData]);

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

  if (!trainPath || !asrPath) {
    return (
      <div className="alignment-error">
        <h3>Alignment artifacts missing</h3>
        <p>Provide both <code>train</code> and <code>asr</code> query params to render the comparison.</p>
        <ul>
          <li>Use the <strong>Visualise Alignment</strong> button in the Job Board after a training-pair job finishes.</li>
          <li>Or open with explicit URLs: <code>/jobs/alignment?train=...&asr=...</code></li>
        </ul>
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

  if (trainLoading || asrLoading) {
    return (
      <div className="alignment-loading">
        <div className="spinner"></div>
        <p>Loading alignment data...</p>
      </div>
    );
  }

  if (trainError || asrError) {
    return (
      <div className="alignment-error">
        <h3>Error loading artifacts</h3>
        <p>Please ensure both the training JSON and ASR JSON are available.</p>
        <pre>{JSON.stringify(trainError || asrError, null, 2)}</pre>
      </div>
    );
  }

  if (!cueGroups.length || !(asrData?.words?.length)) {
    return (
      <div className="alignment-error">
        <h3>Artifacts loaded but empty</h3>
        <p>
          The provided files do not contain any cues or ASR words to visualise. Confirm you selected the <code>.train.words.json</code>
          {' '}and matching <code>.asr.visual.words.diar.json</code> outputs from the job workspace.
        </p>
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
          <h1>Training Alignment</h1>
          <p className="subtext">
            Comparing <strong>Edited Subtitles</strong> (Left) vs. <strong>Raw ASR</strong> (Right)
          </p>
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

          {/* Left Column: Edited Cues */}
          <div className="column left-column">
            {cueGroups.map((cue) => (
              <div
                key={cue.id}
                className="cue-block"
                style={{
                  top: cue.start * zoom,
                  height: Math.max(24, (cue.end - cue.start) * zoom),
                }}
              >
                <span className="cue-id">CUE #{cue.id}</span>
                <p className="cue-text">{cue.text}</p>
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
