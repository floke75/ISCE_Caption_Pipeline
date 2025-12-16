import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import '../styles/system-status.css';

interface HealthResponse {
  status: string;
  system: {
    disk: {
      free_bytes?: number;
      total_bytes?: number;
      percent_used?: number;
      error?: string;
    };
    memory: {
      available_bytes?: number;
      total_bytes?: number;
      percent_used?: number;
      error?: string;
    };
    gpu: {
      available: boolean;
      name: string | null;
      device_count: number;
    };
  };
  queue: {
    pending: number;
    active: number;
    slots_total: number;
  };
}

export function SystemStatus() {
  const [isOpen, setIsOpen] = useState(false);

  const { data, isError, isLoading } = useQuery<HealthResponse>({
    queryKey: ['health'],
    queryFn: async () => {
      const res = await fetch('/api/health');
      if (!res.ok) throw new Error('Health check failed');
      return res.json();
    },
    refetchInterval: 5000,
  });

  if (isLoading) return null;
  if (isError || !data) {
    return (
      <div className="system-status error">
        <span className="status-dot red" />
        <span className="status-text">System: Error</span>
      </div>
    );
  }

  // Determine global status
  let statusColor = 'green';
  let statusText = 'System: OK';

  // Alerts
  const diskFreeGB = data.system.disk.free_bytes ? data.system.disk.free_bytes / 1e9 : 0;
  const memUsedPercent = data.system.memory.percent_used || 0;

  if (data.system.disk.error || (data.system.disk.free_bytes !== undefined && diskFreeGB < 0.5)) { // < 500MB
    statusColor = 'red';
    statusText = 'System: Disk Full';
  } else if (data.system.disk.free_bytes !== undefined && diskFreeGB < 2.0) { // < 2GB
    statusColor = 'yellow';
    statusText = 'System: Low Disk';
  } else if (memUsedPercent > 90) {
    statusColor = 'yellow';
    statusText = 'System: High Mem';
  } else if (!data.system.gpu.available) {
    statusText = 'System: OK (CPU)';
  }

  return (
    <div
      className="system-status-container"
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
    >
      <div className={`system-status ${statusColor}`}>
        <span className={`status-dot ${statusColor}`} />
        <span className="status-text">{statusText}</span>
      </div>

      {isOpen && (
        <div className="system-popover">
          <div className="popover-section">
            <h4>Storage</h4>
            {data.system.disk.error ? (
              <span className="error-text">Unknown</span>
            ) : (
              <>
                <div className="progress-bar">
                  <div
                    className={`progress-fill ${statusColor === 'red' && diskFreeGB < 0.5 ? 'red' : 'blue'}`}
                    style={{ width: `${data.system.disk.percent_used}%` }}
                  />
                </div>
                <div className="stat-detail">
                  {diskFreeGB.toFixed(1)}GB free of {(data.system.disk.total_bytes! / 1e9).toFixed(1)}GB
                </div>
              </>
            )}
          </div>

          <div className="popover-section">
            <h4>Memory</h4>
            {data.system.memory.error ? (
              <span className="error-text">Unknown</span>
            ) : (
              <>
                <div className="progress-bar">
                  <div
                    className={`progress-fill ${memUsedPercent > 90 ? 'yellow' : 'purple'}`}
                    style={{ width: `${memUsedPercent}%` }}
                  />
                </div>
                <div className="stat-detail">
                  {memUsedPercent}% used ({(data.system.memory.available_bytes! / 1e9).toFixed(1)}GB free)
                </div>
              </>
            )}
          </div>

          <div className="popover-section">
            <h4>Compute</h4>
            <div className="stat-row">
              <span>GPU:</span>
              <span className={data.system.gpu.available ? 'good' : 'neutral'}>
                {data.system.gpu.available ? data.system.gpu.name : 'CPU Only'}
              </span>
            </div>
          </div>

          <div className="popover-section">
            <h4>Queue</h4>
            <div className="stat-row">
              <span>Active:</span>
              <span>{data.queue.active} / {data.queue.slots_total}</span>
            </div>
            <div className="stat-row">
              <span>Pending:</span>
              <span>{data.queue.pending}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
