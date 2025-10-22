import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

type StreamStatus = 'idle' | 'connecting' | 'open' | 'error' | 'closed';

interface RetryOptions {
  initial?: number;
  multiplier?: number;
  max?: number;
}

interface UseEventStreamOptions {
  enabled?: boolean;
  withCredentials?: boolean;
  eventTypes?: string[];
  retry?: RetryOptions;
  onOpen?: (event: Event) => void;
  onMessage?: (event: MessageEvent<string>) => void;
  onEvent?: (type: string, event: MessageEvent<string>) => void;
  onError?: (event: Event) => void;
}

interface EventStreamControls {
  status: StreamStatus;
  supported: boolean;
  error: Event | null;
  connect: () => void;
  disconnect: () => void;
}

const DEFAULT_RETRY: Required<RetryOptions> = {
  initial: 1000,
  multiplier: 2,
  max: 15000,
};

export function useEventStream(url: string | null, options: UseEventStreamOptions = {}): EventStreamControls {
  const supported = typeof window !== 'undefined' && typeof window.EventSource !== 'undefined';
  const [status, setStatus] = useState<StreamStatus>(supported ? 'idle' : 'closed');
  const [errorEvent, setErrorEvent] = useState<Event | null>(null);

  const eventSourceRef = useRef<EventSource | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const listenersRef = useRef<Array<{ type: string; listener: EventListener }>>([]);
  const shouldConnectRef = useRef(false);
  const optionsRef = useRef(options);
  const urlRef = useRef(url);

  const enabled = options.enabled ?? true;
  optionsRef.current = options;
  urlRef.current = url;

  const cleanup = useCallback(
    (nextStatus?: StreamStatus) => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      const source = eventSourceRef.current;
      if (source) {
        listenersRef.current.forEach(({ type, listener }) => {
          source.removeEventListener(type, listener);
        });
        listenersRef.current = [];
        source.onopen = null;
        source.onmessage = null;
        source.onerror = null;
        source.close();
        eventSourceRef.current = null;
      }
      if (typeof nextStatus === 'string') {
        setStatus(nextStatus);
      }
    },
    [],
  );

  const connect = useCallback(() => {
    shouldConnectRef.current = supported && Boolean(urlRef.current) && (optionsRef.current.enabled ?? true);
    if (!supported || !urlRef.current || !shouldConnectRef.current) {
      return;
    }
    cleanup();
    setStatus('connecting');
    setErrorEvent(null);

    const { withCredentials, eventTypes, onOpen, onMessage, onEvent, onError, retry } = optionsRef.current;
    const source = new EventSource(urlRef.current, { withCredentials });
    eventSourceRef.current = source;
    listenersRef.current = [];

    source.onopen = (event) => {
      reconnectAttemptsRef.current = 0;
      setStatus('open');
      onOpen?.(event);
    };

    source.onmessage = (event) => {
      onEvent?.('message', event);
      onMessage?.(event);
    };

    const customTypes = eventTypes ?? [];
    customTypes.forEach((type) => {
      const handler = (evt: Event) => {
        onEvent?.(type, evt as MessageEvent<string>);
      };
      source.addEventListener(type, handler);
      listenersRef.current.push({ type, listener: handler });
    });

    source.onerror = (event) => {
      setErrorEvent(event);
      onError?.(event);
      cleanup();
      if (!shouldConnectRef.current) {
        setStatus('closed');
        return;
      }
      setStatus('error');
      const retryConfig = { ...DEFAULT_RETRY, ...(retry ?? {}) };
      reconnectAttemptsRef.current += 1;
      const delay = Math.min(
        retryConfig.initial * Math.pow(retryConfig.multiplier, reconnectAttemptsRef.current - 1),
        retryConfig.max,
      );
      reconnectTimerRef.current = setTimeout(() => {
        if (shouldConnectRef.current) {
          setStatus('connecting');
          connect();
        }
      }, delay);
    };
  }, [cleanup, supported]);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    cleanup('closed');
  }, [cleanup]);

  const shouldConnect = useMemo(() => supported && Boolean(url) && enabled, [supported, url, enabled]);

  useEffect(() => {
    shouldConnectRef.current = shouldConnect;
    reconnectAttemptsRef.current = 0;
    if (!shouldConnect) {
      cleanup('idle');
      return () => {
        shouldConnectRef.current = false;
        cleanup('idle');
      };
    }
    connect();
    return () => {
      shouldConnectRef.current = false;
      cleanup('idle');
    };
  }, [shouldConnect, connect, cleanup]);

  return {
    status,
    supported,
    error: errorEvent,
    connect,
    disconnect,
  };
}

export type { StreamStatus };
