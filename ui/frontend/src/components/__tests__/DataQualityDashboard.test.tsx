import { render, screen, waitFor } from '@testing-library/react';
import { DataQualityDashboard } from '../DataQualityDashboard';
import { useArtifact } from '../../hooks/useArtifacts';
import { vi, describe, it, expect } from 'vitest';

// Mock the hook
vi.mock('../../hooks/useArtifacts');

describe('DataQualityDashboard', () => {
  it('displays loading state', () => {
    vi.mocked(useArtifact).mockReturnValue({
      data: undefined,
      isLoading: true,
      error: null,
      isError: false,
      isSuccess: false,
      status: 'pending',
      fetchStatus: 'fetching'
    } as any);

    render(<DataQualityDashboard artifactPath="/path/to/artifact.json" />);
    expect(screen.getByText('Loading artifact data...')).toBeInTheDocument();
  });

  it('displays error state', () => {
    vi.mocked(useArtifact).mockReturnValue({
      data: undefined,
      isLoading: false,
      error: new Error('Failed to fetch'),
      isError: true,
      isSuccess: false,
      status: 'error',
      fetchStatus: 'idle'
    } as any);

    render(<DataQualityDashboard artifactPath="/path/to/artifact.json" />);
    expect(screen.getByText('Failed to load artifact: Failed to fetch')).toBeInTheDocument();
  });

  it('displays metrics when data is loaded', async () => {
    const mockData = [
      { w: "hello", start: 0, end: 0.5, pause_after_ms: 100 },
      { w: "world", start: 0.5, end: 1.0, pause_after_ms: 600, speaker_change: true }
    ];

    vi.mocked(useArtifact).mockReturnValue({
      data: mockData,
      isLoading: false,
      error: null,
      isError: false,
      isSuccess: true,
      status: 'success',
      fetchStatus: 'idle'
    } as any);

    render(<DataQualityDashboard artifactPath="/path/to/artifact.json" />);

    expect(screen.getByText('Data Quality Metrics')).toBeInTheDocument();
    expect(screen.getByText('Total Tokens')).toBeInTheDocument();
    expect(screen.getByText('2')).toBeInTheDocument(); // Total tokens
    expect(screen.getByText('Speaker Changes')).toBeInTheDocument();
    // Use getAllByText for '1' since it appears multiple times (speaker changes, long pauses, etc.)
    expect(screen.getAllByText('1').length).toBeGreaterThan(0);
    expect(screen.getByText('Pauses > 500ms')).toBeInTheDocument();
  });
});
