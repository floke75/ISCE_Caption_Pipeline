import { useEffect, useMemo, useRef, useState } from 'react';

import {
  PathKind,
  PathValidationRequest,
  PathValidationResult,
  validatePath,
} from '../../lib/api';

export interface PathFieldValidation {
  valid: boolean;
  resolvedPath: string | null;
  message: string | null;
  checking: boolean;
}

interface ValidatedPathFieldProps {
  label: string;
  value: string;
  onChange(value: string): void;
  kind: PathKind;
  required?: boolean;
  mustExist?: boolean;
  allowCreate?: boolean;
  placeholder?: string;
  description?: string;
  helpText?: string;
  disabled?: boolean;
  onValidation(result: PathFieldValidation): void;
}

type FieldStatus = 'idle' | 'validating' | 'valid' | 'invalid';

const REQUIRED_MESSAGE = 'This field is required.';

export function ValidatedPathField({
  label,
  value,
  onChange,
  kind,
  required = false,
  mustExist = true,
  allowCreate = false,
  placeholder,
  description,
  helpText,
  disabled = false,
  onValidation,
}: ValidatedPathFieldProps) {
  const [status, setStatus] = useState<FieldStatus>('idle');
  const [helper, setHelper] = useState<string | null>(helpText ?? null);
  const [resolved, setResolved] = useState<string | null>(null);
  const [allowedRoots, setAllowedRoots] = useState<string[]>([]);
  const validationRef = useRef<PathFieldValidation>({
    valid: !required,
    resolvedPath: null,
    message: required ? REQUIRED_MESSAGE : null,
    checking: false,
  });
  const requestId = useRef(0);

  const trimmedValue = useMemo(() => value.trim(), [value]);

  useEffect(() => {
    onValidation(validationRef.current);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (disabled) {
      setStatus('idle');
      setHelper(helpText ?? null);
      setResolved(null);
      setAllowedRoots([]);
      validationRef.current = {
        valid: !required,
        resolvedPath: null,
        message: required ? REQUIRED_MESSAGE : null,
        checking: false,
      };
      onValidation(validationRef.current);
      return;
    }

    if (!trimmedValue) {
      const message = required ? REQUIRED_MESSAGE : null;
      setStatus(required ? 'invalid' : 'idle');
      setHelper(message ?? helpText ?? null);
      setResolved(null);
      setAllowedRoots([]);
      validationRef.current = {
        valid: !required,
        resolvedPath: null,
        message,
        checking: false,
      };
      onValidation(validationRef.current);
      return;
    }

    const currentId = requestId.current + 1;
    requestId.current = currentId;
    setStatus('validating');
    setHelper('Validating path…');
    setResolved(null);
    setAllowedRoots([]);
    validationRef.current = {
      valid: false,
      resolvedPath: null,
      message: null,
      checking: true,
    };
    onValidation(validationRef.current);

    const payload: PathValidationRequest = {
      path: trimmedValue,
      kind,
      must_exist: mustExist,
      allow_create: allowCreate,
      purpose: description,
    };

    const timer = setTimeout(() => {
      validatePath(payload)
        .then((result: PathValidationResult) => {
          if (requestId.current !== currentId) {
            return;
          }
          setAllowedRoots(result.allowed_roots ?? []);
          if (result.valid) {
            const resolvedPath = result.resolved_path ?? trimmedValue;
            const message = result.exists
              ? 'Path validated successfully.'
              : 'Path will be created if needed.';
            setStatus('valid');
            setHelper(message);
            setResolved(resolvedPath);
            validationRef.current = {
              valid: true,
              resolvedPath,
              message: null,
              checking: false,
            };
          } else {
            const message = result.message || 'Path could not be validated.';
            setStatus('invalid');
            setHelper(message);
            setResolved(null);
            validationRef.current = {
              valid: false,
              resolvedPath: null,
              message,
              checking: false,
            };
          }
          onValidation(validationRef.current);
        })
        .catch((error: unknown) => {
          if (requestId.current !== currentId) {
            return;
          }
          const message = error instanceof Error ? error.message : 'Validation failed.';
          setStatus('invalid');
          setHelper(message);
          setResolved(null);
          setAllowedRoots([]);
          validationRef.current = {
            valid: false,
            resolvedPath: null,
            message,
            checking: false,
          };
          onValidation(validationRef.current);
        });
    }, 350);

    return () => {
      clearTimeout(timer);
    };
  }, [
    allowCreate,
    description,
    disabled,
    helpText,
    kind,
    mustExist,
    onValidation,
    required,
    trimmedValue,
  ]);

  const statusLabel = useMemo(() => {
    switch (status) {
      case 'validating':
        return 'Validating…';
      case 'valid':
        return 'Valid';
      case 'invalid':
        return 'Invalid';
      default:
        return '';
    }
  }, [status]);

  return (
    <div className={`validated-path-field ${status}`}>
      <label>
        {label}
        <div className="validated-path-input">
          <input
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
            disabled={disabled}
          />
          <span className="path-status" aria-live="polite">
            {statusLabel}
          </span>
        </div>
      </label>
      {helper && (
        <p className={`field-hint ${status === 'invalid' ? 'error' : ''}`}>
          {helper}
          {status === 'invalid' && allowedRoots.length > 0 && (
            <span className="allowed-roots">
              Allowed roots: {allowedRoots.join(', ')}
            </span>
          )}
        </p>
      )}
      {resolved && status === 'valid' && (
        <p className="field-hint resolved">Resolved path: {resolved}</p>
      )}
    </div>
  );
}

export default ValidatedPathField;
