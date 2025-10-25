import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import client from '../api/client';
import { FileListing, FileRoot, FileValidation } from '../types';
import '../styles/forms.css';

type FileKind = 'file' | 'directory';

type Props = {
  label: string;
  value: string;
  onChange: (next: string) => void;
  required?: boolean;
  helperText?: string;
  placeholder?: string;
  type?: FileKind;
  onValidityChange?: (valid: boolean) => void;
};

const VALIDATE_DEBOUNCE_MS = 350;

type ValidationState = 'idle' | 'checking' | 'valid' | 'invalid';

async function fetchRoots(): Promise<FileRoot[]> {
  const { data } = await client.get<FileRoot[]>('/files/roots');
  return data;
}

async function fetchListing(path?: string, root?: string): Promise<FileListing> {
  const { data } = await client.get<FileListing>('/files/list', { params: { path, root } });
  return data;
}

async function validatePath(path: string): Promise<FileValidation> {
  const { data } = await client.get<FileValidation>('/files/validate', { params: { path } });
  return data;
}

/**
 * A composite form control for selecting file or directory paths.
 *
 * This component provides a text input for manual path entry, along with a
 * "Browse" button that opens a modal file browser. It performs debounced,
 * real-time validation of the entered path against the backend API to ensure
 * the path is accessible and of the correct type (file or directory).
 *
 * @param {Props} props The props for the component.
 * @returns {JSX.Element} The rendered file path picker.
 */
export function FilePathPicker({
  label,
  value,
  onChange,
  required = false,
  helperText,
  placeholder,
  type = 'file',
  onValidityChange,
}: Props) {
  const [validationState, setValidationState] = useState<ValidationState>('idle');
  const [validationMessage, setValidationMessage] = useState<string>('');
  const [isBrowserOpen, setBrowserOpen] = useState(false);
  const [selectedRootId, setSelectedRootId] = useState<string>('');
  const [browserPath, setBrowserPath] = useState<string>('');
  const [listing, setListing] = useState<FileListing | null>(null);
  const [listingError, setListingError] = useState<string | null>(null);
  const [listingBusy, setListingBusy] = useState(false);
  const lastValidationValue = useRef<string>('');
  const lastListingPath = useRef<string>('');
  const parentPath = listing?.parent ?? null;

  const { data: roots = [], isLoading: rootsLoading } = useQuery<FileRoot[]>({
    queryKey: ['file-roots'],
    queryFn: fetchRoots,
    staleTime: 1000 * 60,
  });

  useEffect(() => {
    if (!roots.length || selectedRootId) return;
    const initial = roots[0];
    setSelectedRootId(initial.id);
    setBrowserPath(initial.path);
  }, [roots, selectedRootId]);

  const selectedRoot = useMemo(
    () => roots.find((root) => root.id === selectedRootId) ?? roots[0],
    [roots, selectedRootId],
  );

  const refreshListing = useCallback(
    async (targetPath: string, targetRoot?: string) => {
      if (!targetPath) return;
      setListingBusy(true);
      setListingError(null);
      try {
        const data = await fetchListing(targetPath, targetRoot);
        setListing(data);
        lastListingPath.current = data.path;
      } catch (error: any) {
        setListingError(error?.response?.data?.detail ?? 'Unable to load directory');
      } finally {
        setListingBusy(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!isBrowserOpen) return;
    const target = browserPath || selectedRoot?.path;
    if (target && target !== lastListingPath.current) {
      refreshListing(target, undefined);
    } else if (target && !listing) {
      refreshListing(target, undefined);
    }
  }, [isBrowserOpen, browserPath, selectedRoot, refreshListing, listing]);

  const handleRootChange = (event: ChangeEvent<HTMLSelectElement>) => {
    const nextRoot = event.target.value;
    setSelectedRootId(nextRoot);
    const rootDef = roots.find((root) => root.id === nextRoot);
    if (rootDef) {
      setBrowserPath(rootDef.path);
      setListing(null);
      lastListingPath.current = '';
    }
  };

  const handleInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    onChange(event.target.value);
  };

  const handleSelectPath = (path: string) => {
    onChange(path);
    setBrowserOpen(false);
  };

  const handleOpenDirectory = (path: string) => {
    setBrowserPath(path);
    lastListingPath.current = '';
  };

  const handleUseCurrentDirectory = () => {
    if (listing) {
      handleSelectPath(listing.path);
    }
  };

  useEffect(() => {
    const trimmed = value.trim();
    if (!trimmed) {
      if (required) {
        setValidationState('invalid');
        setValidationMessage('Path is required');
        onValidityChange?.(false);
      } else {
        setValidationState('idle');
        setValidationMessage('');
        onValidityChange?.(true);
      }
      return;
    }

    lastValidationValue.current = trimmed;
    setValidationState('checking');
    setValidationMessage('Validating path…');
    const timer = window.setTimeout(async () => {
      try {
        const result = await validatePath(trimmed);
        if (lastValidationValue.current !== trimmed) {
          return;
        }
        let valid = result.allowed;
        let message = result.detail ?? '';
        if (!result.allowed) {
          message = result.detail ?? 'Path is outside the allowed directories';
        } else if (type === 'directory') {
          // Directories can be created automatically by the backend, so only fail for conflicting entries.
          if (result.exists && !result.isDir) {
            valid = false;
            message = 'Expected a directory path';
          } else if (!result.exists) {
            message = 'Directory will be created automatically when the job runs';
          } else if (!message) {
            message = 'Directory verified';
          }
        } else if (!result.exists) {
          // Files must exist.
          valid = false;
          message = result.detail ?? 'Path does not exist';
        } else if (!result.isFile) {
          // And they must be files.
          valid = false;
          message = 'Expected a file path';
        } else if (!message) {
          message = 'Path verified';
        }
        setValidationState(valid ? 'valid' : 'invalid');
        setValidationMessage(message);
        onValidityChange?.(valid);
      } catch (error: any) {
        if (lastValidationValue.current !== trimmed) {
          return;
        }
        setValidationState('invalid');
        setValidationMessage(error?.response?.data?.detail ?? 'Unable to validate path');
        onValidityChange?.(false);
      }
    }, VALIDATE_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timer);
    };
  }, [value, required, type, onValidityChange]);

  const toggleBrowser = () => {
    if (!isBrowserOpen && selectedRoot && !browserPath) {
      setBrowserPath(selectedRoot.path);
    }
    setBrowserOpen((open) => !open);
  };

  return (
    <label className="field file-picker">
      <span>
        {label}
        {required ? ' *' : ''}
      </span>
      <div className="file-picker-controls">
        {roots.length > 1 && (
          <select value={selectedRootId} onChange={handleRootChange} disabled={rootsLoading}>
            {roots.map((root) => (
              <option key={root.id} value={root.id}>
                {root.label}
              </option>
            ))}
          </select>
        )}
        <input type="text" value={value} onChange={handleInputChange} placeholder={placeholder} />
        <button type="button" className="ghost" onClick={toggleBrowser}>
          {isBrowserOpen ? 'Close' : 'Browse'}
        </button>
      </div>
      {helperText && <span className="field-help">{helperText}</span>}
      <span className={`file-picker-status ${validationState}`}>
        {validationMessage || (validationState === 'checking' ? 'Validating…' : required ? ' ' : 'Optional path')}
      </span>
      {isBrowserOpen && (
        <div className="file-picker-browser">
          <div className="file-browser-header">
            <div className="file-browser-path">{listing?.path ?? browserPath}</div>
            <div className="file-browser-actions">
              {type === 'directory' && (
                <button type="button" className="ghost" onClick={handleUseCurrentDirectory} disabled={!listing}>
                  Use directory
                </button>
              )}
            </div>
          </div>
          {listingError && <div className="file-browser-error">{listingError}</div>}
          {listingBusy && <div className="file-browser-empty">Loading…</div>}
          {!listingBusy && listing && (
            <div className="file-browser-list">
              {parentPath && (
                <button type="button" className="file-browser-item" onClick={() => handleOpenDirectory(parentPath)}>
                  ← Up one level
                </button>
              )}
              {listing.entries.map((entry) => (
                <div key={entry.path} className="file-browser-row">
                  <button
                    type="button"
                    className="file-browser-item"
                    onClick={() => (entry.isDir ? handleOpenDirectory(entry.path) : handleSelectPath(entry.path))}
                  >
                    <span className={entry.isDir ? 'entry-icon dir' : 'entry-icon file'} aria-hidden />
                    {entry.name}
                  </button>
                  {entry.isDir && type === 'directory' && (
                    <button type="button" className="ghost" onClick={() => handleSelectPath(entry.path)}>
                      Select
                    </button>
                  )}
                  {entry.isFile && type === 'file' && (
                    <button type="button" className="ghost" onClick={() => handleSelectPath(entry.path)}>
                      Select
                    </button>
                  )}
                </div>
              ))}
              {!listing.entries.length && <div className="file-browser-empty">Directory is empty</div>}
            </div>
          )}
        </div>
      )}
    </label>
  );
}
