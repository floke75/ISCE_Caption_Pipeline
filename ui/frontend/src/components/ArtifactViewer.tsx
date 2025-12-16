import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import client from '../api/client';
import { useState, useMemo } from 'react';

type FileContent = {
  path: string;
  content: string;
  size: number;
  mimeType?: string;
  truncated: boolean;
};

export function ArtifactViewer() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const path = searchParams.get('path');
  const [lineWrap, setLineWrap] = useState(false);

  const { data, isLoading, error } = useQuery<FileContent>({
    queryKey: ['fileContent', path],
    queryFn: async () => {
      if (!path) throw new Error('No path provided');
      const res = await client.get('/files/content', { params: { path } });
      return res.data;
    },
    enabled: Boolean(path),
    retry: false
  });

  const displayContent = useMemo(() => {
    if (!data?.content) return '';
    const isJson = data.path.toLowerCase().endsWith('.json') || data.mimeType?.includes('json');
    if (isJson) {
      try {
        const parsed = JSON.parse(data.content);
        return JSON.stringify(parsed, null, 2);
      } catch {
        return data.content;
      }
    }
    return data.content;
  }, [data]);

  if (!path) return <div className="p-4">No file path specified.</div>;
  if (isLoading) return <div className="p-4">Loading file content...</div>;
  if (error) return <div className="p-4 text-red-400">Error loading file: {(error instanceof Error ? error.message : String(error))}</div>;

  const handleDownload = () => {
    const url = `${client.defaults.baseURL}/files/download?path=${encodeURIComponent(path)}`;
    window.location.href = url;
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-200">
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-slate-900">
        <div className="flex items-center gap-4">
          <button
            onClick={() => navigate(-1)}
            className="text-slate-400 hover:text-white transition-colors"
          >
            ← Back
          </button>
          <div>
             <h1 className="text-lg font-semibold truncate max-w-xl" title={path}>{path.split('/').pop()}</h1>
             <p className="text-xs text-slate-500 font-mono">{data?.mimeType} · {data?.size.toLocaleString()} bytes {data?.truncated && '(Truncated preview)'}</p>
          </div>
        </div>
        <div className="flex gap-3">
          <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
             <input type="checkbox" checked={lineWrap} onChange={e => setLineWrap(e.target.checked)} />
             Wrap lines
          </label>
          <button
            onClick={handleDownload}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 text-white rounded text-sm font-medium transition-colors"
          >
            Download Raw
          </button>
        </div>
      </header>

      <div className="flex-1 overflow-auto p-4">
        <pre className={`font-mono text-sm ${lineWrap ? 'whitespace-pre-wrap' : 'whitespace-pre'}`}>
          {displayContent}
        </pre>
      </div>

      {data?.truncated && (
        <div className="bg-yellow-900/20 border-t border-yellow-900/50 p-2 text-center text-yellow-200 text-sm">
           This file is large. Only the first 500KB are shown. Download the file to view full content.
        </div>
      )}
    </div>
  );
}
