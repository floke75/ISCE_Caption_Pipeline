import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE_URL } from '../lib/api';

export type LogStreamStatus = 'idle' | 'streaming' | 'complete' | 'error';

interface LogEventPayload {
  type: 'log' | 'status';
  content?: string;
  offset?: number;
  status?: string;
}

const buildStreamUrl = (jobId: string, offset: number) => {
  const url = new URL(`/api/jobs/${jobId}/logs/stream`, API_BASE_URL);
  url.searchParams.set('offset', String(Math.max(0, offset)));
  return url.toString();
};

export const useJobLogs = (jobId?: string) => {
  const [log, setLog] = useState('');
  const [status, setStatus] = useState<LogStreamStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);

  const offsetRef = useRef(0);
  const sourceRef = useRef<EventSource | null>(null);

  const closeStream = useCallback((nextStatus: LogStreamStatus = 'idle') => {
    if (sourceRef.current) {
      sourceRef.current.close();
      sourceRef.current = null;
    }
    setIsStreaming(false);
    setStatus(nextStatus);
    if (nextStatus !== 'error') {
      setError(null);
    }
  }, []);

  const openStream = useCallback(
    (options: { resetLog?: boolean } = {}) => {
      if (!jobId) {
        return;
      }

      const { resetLog = false } = options;
      closeStream('idle');

      const startOffset = resetLog ? 0 : offsetRef.current;
      if (resetLog) {
        offsetRef.current = 0;
        setLog('');
      }

      setError(null);
      setStatus('streaming');
      setIsStreaming(true);

      const streamUrl = buildStreamUrl(jobId, startOffset);
      let firstChunk = resetLog;
      const source = new EventSource(streamUrl);
      sourceRef.current = source;

      source.onmessage = (event: MessageEvent<string>) => {
        try {
          const payload: LogEventPayload = JSON.parse(event.data);
          if (payload.type === 'log') {
            const content = payload.content ?? '';
            if (content) {
              setLog((prev) => {
                const next = firstChunk ? content : prev + content;
                firstChunk = false;
                return next;
              });
            }
            if (typeof payload.offset === 'number') {
              offsetRef.current = payload.offset;
            }
            return;
          }

          if (payload.type === 'status') {
            if (typeof payload.offset === 'number') {
              offsetRef.current = payload.offset;
            }
            closeStream('complete');
          }
        } catch (parseError) {
          console.warn('Failed to parse log stream payload', parseError);
        }
      };

      source.onerror = () => {
        if (sourceRef.current === source) {
          closeStream('error');
          setError('Log stream disconnected.');
        }
      };
    },
    [jobId, closeStream]
  );

  useEffect(() => {
    if (!jobId) {
      closeStream('idle');
      setLog('');
      offsetRef.current = 0;
      return () => undefined;
    }

    openStream({ resetLog: true });
    return () => closeStream('idle');
  }, [jobId, openStream, closeStream]);

  const reconnect = useCallback(() => {
    if (!jobId) {
      return;
    }
    openStream({ resetLog: false });
  }, [jobId, openStream]);

  return {
    log,
    status,
    error,
    isStreaming,
    reconnect,
  };
};
