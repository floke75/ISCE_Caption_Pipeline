import { useMemo } from 'react';
import clsx from 'clsx';
import { useArtifact } from '../hooks/useArtifacts';

interface DataQualityDashboardProps {
  artifactPath: string;
}

interface Token {
  w: string;
  start?: number;
  end?: number;
  pause_after_ms?: number;
  pause_before_ms?: number;
  speaker?: string;
  speaker_change?: boolean;
  break_type?: 'O' | 'LB' | 'SB';
  pos?: string;
}

interface Stats {
  totalTokens: number;
  durationSeconds: number;
  avgCps: number;
  speakerChanges: number;
  longPauses: number;
  breakCounts: { O: number; LB: number; SB: number };
  pauseBuckets: number[];
  pauseBucketLabels: string[];
}

function calculateStats(tokens: Token[]): Stats {
  if (!tokens.length) {
    return {
      totalTokens: 0,
      durationSeconds: 0,
      avgCps: 0,
      speakerChanges: 0,
      longPauses: 0,
      breakCounts: { O: 0, LB: 0, SB: 0 },
      pauseBuckets: [0, 0, 0, 0, 0],
      pauseBucketLabels: ['0-100', '100-300', '300-500', '500-1s', '1s+'],
    };
  }

  let totalChars = 0;
  let speakerChanges = 0;
  let longPauses = 0;
  const breakCounts = { O: 0, LB: 0, SB: 0 };
  const pauseBuckets = [0, 0, 0, 0, 0]; // 0-100, 100-300, 300-500, 500-1000, >1000

  const start = tokens[0].start || 0;
  const end = tokens[tokens.length - 1].end || 0;
  const durationSeconds = Math.max(0, end - start);

  tokens.forEach((t) => {
    totalChars += t.w.length;
    if (t.speaker_change) speakerChanges++;
    const p = t.pause_after_ms || 0;

    if (p > 500) longPauses++;

    if (p < 100) pauseBuckets[0]++;
    else if (p < 300) pauseBuckets[1]++;
    else if (p < 500) pauseBuckets[2]++;
    else if (p < 1000) pauseBuckets[3]++;
    else pauseBuckets[4]++;

    if (t.break_type) {
      if (t.break_type === 'LB') breakCounts.LB++;
      else if (t.break_type === 'SB') breakCounts.SB++;
      else breakCounts.O++;
    }
  });

  return {
    totalTokens: tokens.length,
    durationSeconds,
    avgCps: durationSeconds > 0 ? totalChars / durationSeconds : 0,
    speakerChanges,
    longPauses,
    breakCounts,
    pauseBuckets,
    pauseBucketLabels: ['0-100', '100-300', '300-500', '500-1s', '1s+'],
  };
}

export function DataQualityDashboard({ artifactPath }: DataQualityDashboardProps) {
  const { data, isLoading, error } = useArtifact<Token[]>(artifactPath);

  const stats = useMemo(() => {
    // Handle both training format (list of TokenRow) and inference (Enriched)
    // Sometimes training data is nested or simple list. Assuming list of dicts based on previous tasks.
    if (!data) return null;
    let tokens: Token[] = [];
    if (Array.isArray(data)) {
      tokens = data as Token[];
    } else if (data && typeof data === 'object' && 'tokens' in data) {
      tokens = (data as { tokens: Token[] }).tokens;
    }
    return calculateStats(tokens);
  }, [data]);

  if (isLoading) return <div className="p-4">Loading artifact data...</div>;
  if (error) return <div className="p-4 text-red-500">Failed to load artifact: {error.message}</div>;
  if (!stats) return <div className="p-4">No data available.</div>;

  const maxBucket = Math.max(...stats.pauseBuckets, 1);
  const totalBreaks = stats.breakCounts.O + stats.breakCounts.LB + stats.breakCounts.SB || 1;

  return (
    <div className="bg-gray-50 p-6 rounded-lg border border-gray-200 mt-4">
      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900">Data Quality Metrics</h2>
        <p className="text-sm text-gray-500 truncate">{artifactPath}</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatCard label="Total Tokens" value={stats.totalTokens.toLocaleString()} />
        <StatCard
          label="Avg CPS"
          value={stats.avgCps.toFixed(1)}
          sub={stats.avgCps > 25 ? 'High' : stats.avgCps < 10 ? 'Low' : 'Normal'}
          color={stats.avgCps > 25 ? 'red' : 'green'}
        />
        <StatCard label="Speaker Changes" value={stats.speakerChanges} />
        <StatCard
          label="Pauses > 500ms"
          value={stats.longPauses}
          sub={`${((stats.longPauses / stats.totalTokens) * 100).toFixed(1)}% of tokens`}
        />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-4 rounded shadow-sm border border-gray-200 md:col-span-2">
          <h3 className="text-sm font-medium text-gray-700 mb-4">Pause Duration Distribution (ms)</h3>
          <div className="flex items-end h-40 gap-2 justify-between px-2">
            {stats.pauseBuckets.map((count, idx) => (
              <div key={idx} className="flex flex-col items-center flex-1 group">
                <div
                  className="w-full bg-blue-500 rounded-t transition-all hover:bg-blue-600 relative"
                  style={{ height: `${(count / maxBucket) * 100}%`, minHeight: '4px' }}
                >
                    <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-xs bg-gray-800 text-white px-1 rounded opacity-0 group-hover:opacity-100 transition-opacity">
                        {count}
                    </div>
                </div>
                <span className="text-xs text-gray-500 mt-2">{stats.pauseBucketLabels[idx]}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-4 rounded shadow-sm border border-gray-200">
          <h3 className="text-sm font-medium text-gray-700 mb-4">Break Types</h3>
          <div className="space-y-3">
             <BreakBar label="None (O)" count={stats.breakCounts.O} total={totalBreaks} color="bg-blue-500" />
             <BreakBar label="Line Break (LB)" count={stats.breakCounts.LB} total={totalBreaks} color="bg-green-500" />
             <BreakBar label="Sentence Break (SB)" count={stats.breakCounts.SB} total={totalBreaks} color="bg-yellow-500" />
          </div>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, sub, color = 'gray' }: { label: string; value: string | number; sub?: string; color?: string }) {
  return (
    <div className="bg-white p-4 rounded shadow-sm border border-gray-200">
      <div className="text-xs font-semibold text-gray-500 uppercase tracking-wider">{label}</div>
      <div className="text-2xl font-bold text-gray-900 mt-1">{value}</div>
      {sub && <div className={`text-xs mt-1 text-${color}-600`}>{sub}</div>}
    </div>
  );
}

function BreakBar({ label, count, total, color }: { label: string; count: number; total: number; color: string }) {
  const pct = total > 0 ? (count / total) * 100 : 0;
  return (
    <div>
      <div className="flex justify-between text-xs mb-1">
        <span className="text-gray-600">{label}</span>
        <span className="font-medium">{pct.toFixed(1)}% ({count})</span>
      </div>
      <div className="w-full bg-gray-100 rounded-full h-2 overflow-hidden">
        <div className={clsx('h-full', color)} style={{ width: `${pct}%` }}></div>
      </div>
    </div>
  );
}
