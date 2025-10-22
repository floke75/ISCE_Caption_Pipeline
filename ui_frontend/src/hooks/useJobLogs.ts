import { useCallback, useEffect, useRef, useState } from 'react';
import { API_BASE_URL } from '../lib/api';

export type LogStreamStatus = 'idle' | 'streaming' | 'reconnecting' | 'complete' | 'error';

interface LogEventPayload {
  type: 'log' | 'status';
  content?: string;
  offset?: number;
  status?: string;
}

const INITIAL_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

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
  const [retryDelay, setRetryDelay] = useState<number | null>(null);

  const offsetRef = useRef(0);
  const sourceRef = useRef<EventSource | null>(null);
  const retryTimeoutRef = useRef<number | null>(null);
  const backoffRef = useRef(INITIAL_BACKOFF_MS);

  const clearRetryTimeout = useCallback(() => {
    if (retryTimeoutRef.current !== null) {
      window.clearTimeout(retryTimeoutRef.current);
      retryTimeoutRef.current = null;
    }
    setRetryDelay(null);
  }, []);

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
    if (nextStatus !== 'reconnecting') {
      clearRetryTimeout();
    }
  }, [clearRetryTimeout]);

  const openStream = useCallback(
    (options: { resetLog?: boolean; resetBackoff?: boolean } = {}) => {
      if (!jobId) {
        return;
      }

      const { resetLog = false, resetBackoff = true } = options;
      closeStream('idle');

      const startOffset = resetLog ? 0 : offsetRef.current;
      if (resetLog) {
        offsetRef.current = 0;
        setLog('');
      }

      if (resetBackoff) {
        backoffRef.current = INITIAL_BACKOFF_MS;
        clearRetryTimeout();
      }

      setError(null);
      setStatus('streaming');
      setIsStreaming(true);
      setRetryDelay(null);

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
          if (jobId) {
            const delay = backoffRef.current;
            setRetryDelay(delay);
            setStatus('reconnecting');
            retryTimeoutRef.current = window.setTimeout(() => {
              retryTimeoutRef.current = null;
              setRetryDelay(null);
              openStream({ resetLog: false, resetBackoff: false });
            }, delay);
            backoffRef.current = Math.min(backoffRef.current * 2, MAX_BACKOFF_MS);
          }
        }
      };
    },
    [jobId, closeStream, clearRetryTimeout]
  );

  useEffect(() => {
    if (!jobId) {
      closeStream('idle');
      setLog('');
      offsetRef.current = 0;
      clearRetryTimeout();
      return () => undefined;
    }

    openStream({ resetLog: true, resetBackoff: true });
    return () => {
      closeStream('idle');
      clearRetryTimeout();
    };
  }, [jobId, openStream, closeStream, clearRetryTimeout]);

  const reconnect = useCallback(() => {
    if (!jobId) {
      return;
    }
    openStream({ resetLog: false, resetBackoff: true });
  }, [jobId, openStream]);

  const refresh = useCallback(() => {
    if (!jobId) {
      return;
    }
    openStream({ resetLog: true, resetBackoff: true });
  }, [jobId, openStream]);

  return {
    log,
    status,
    error,
    isStreaming,
    reconnect,
    refresh,
    retryDelay,
  };
};
