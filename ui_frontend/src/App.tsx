import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { isAxiosError } from 'axios';
import {
  cancelJob,
  fetchJob,
  fetchJobs,
  JobDetail as JobDetailType,
  JobSummary,
} from './lib/api';
import JobTable from './components/JobTable';
import JobDetail from './components/JobDetail';
import { InferenceForm } from './components/forms/InferenceForm';
import { TrainingPairForm } from './components/forms/TrainingPairForm';
import { ModelTrainingForm } from './components/forms/ModelTrainingForm';
import { ConfigEditor } from './components/ConfigEditor';
import { copyToClipboard } from './lib/clipboard';

const tabs = [
  { key: 'inference', label: 'Inference' },
  { key: 'training', label: 'Training pair' },
  { key: 'model', label: 'Model training' },
  { key: 'config', label: 'Configuration' },
] as const;

export type TabKey = (typeof tabs)[number]['key'];

function App() {
  const [activeTab, setActiveTab] = useState<TabKey>('inference');
  const [selectedJobId, setSelectedJobId] = useState<string | undefined>();
  const [actionBanner, setActionBanner] = useState<
    { type: 'success' | 'error'; message: string } | null
  >(null);

  const {
    data: jobsData,
    refetch: refetchJobs,
    isLoading: jobsLoading,
    isError: jobsError,
  } = useQuery({
    queryKey: ['jobs'],
    queryFn: fetchJobs,
    refetchInterval: 4000,
  });

  const jobs: JobSummary[] = jobsData ?? [];

  useEffect(() => {
    if (!selectedJobId && jobs.length > 0) {
      setSelectedJobId(jobs[0].id);
    }
  }, [jobs, selectedJobId]);

  useEffect(() => {
    if (!actionBanner) {
      return undefined;
    }
    const timeout = window.setTimeout(() => setActionBanner(null), 4000);
    return () => window.clearTimeout(timeout);
  }, [actionBanner]);

  const jobQuery = useQuery({
    queryKey: ['job', selectedJobId],
    queryFn: () => fetchJob(selectedJobId as string),
    enabled: Boolean(selectedJobId),
    refetchInterval: selectedJobId ? 3000 : false,
  });

  const selectedJob: JobDetailType | undefined = jobQuery.data;

  const cancelMutation = useMutation({
    mutationFn: (jobId: string) => cancelJob(jobId),
    onSuccess: (job) => {
      setActionBanner({ type: 'success', message: 'Cancellation requested.' });
      setSelectedJobId(job.id);
      refetchJobs();
    },
    onError: (error: unknown) => {
      let message = 'Failed to cancel job.';
      if (isAxiosError(error)) {
        const detail =
          typeof error.response?.data === 'object' && error.response?.data !== null
            ? (error.response.data as { detail?: unknown }).detail
            : undefined;
        message =
          (typeof detail === 'string' && detail) || error.message || 'Failed to cancel job.';
      } else if (error instanceof Error && error.message) {
        message = error.message;
      }
      setActionBanner({ type: 'error', message });
    },
  });

  const handleCopyWorkspace = async (job: JobSummary) => {
    const success = await copyToClipboard(job.workspace);
    setActionBanner({
      type: success ? 'success' : 'error',
      message: success ? 'Workspace path copied to clipboard.' : 'Unable to copy workspace path.',
    });
  };

  const handleCancelJob = (job: JobSummary) => {
    if (!['pending', 'running'].includes(job.status)) {
      setActionBanner({ type: 'error', message: 'Job is no longer running.' });
      return;
    }
    if (job.cancel_requested) {
      setActionBanner({ type: 'success', message: 'Cancellation already requested.' });
      return;
    }
    cancelMutation.mutate(job.id);
  };

  const pendingCancelId =
    cancelMutation.isPending && typeof cancelMutation.variables === 'string'
      ? cancelMutation.variables
      : undefined;

  const formContent = useMemo(() => {
    switch (activeTab) {
      case 'inference':
        return (
          <InferenceForm
            onCreated={(job) => {
              setSelectedJobId(job.id);
              refetchJobs();
            }}
          />
        );
      case 'training':
        return (
          <TrainingPairForm
            onCreated={(job) => {
              setSelectedJobId(job.id);
              refetchJobs();
            }}
          />
        );
      case 'model':
        return (
          <ModelTrainingForm
            onCreated={(job) => {
              setSelectedJobId(job.id);
              refetchJobs();
            }}
          />
        );
      case 'config':
        return <ConfigEditor onSaved={() => refetchJobs()} />;
      default:
        return null;
    }
  }, [activeTab, refetchJobs]);

  return (
    <div className="app-shell">
      <header className="app-header">
        <h1>ISCE Pipeline Control Center</h1>
        <p>
          Launch inference, assemble training data, and monitor iterative model training from a single dashboard.
          Configuration edits are version-aware and jobs run in isolated sandboxes with on-disk logs.
        </p>
      </header>

      <main className="app-body">
        <section className="panel">
          <div className="panel-header">
            <h2>Workflows</h2>
          </div>
          <div className="tabs">
            {tabs.map((tab) => (
              <button
                key={tab.key}
                className={`tab-button ${activeTab === tab.key ? 'active' : ''}`}
                onClick={() => setActiveTab(tab.key)}
              >
                {tab.label}
              </button>
            ))}
          </div>
          {formContent}
        </section>

        <section className="panel">
          <div className="panel-header">
            <h2>Job monitor</h2>
          </div>
          <div className="panel-body" style={{ display: 'grid', gap: '1.25rem' }}>
            {actionBanner && (
              <div className={`message-banner ${actionBanner.type}`}>{actionBanner.message}</div>
            )}
            {jobsError && (
              <div className="message-banner error">Failed to load job list.</div>
            )}
            {jobsLoading ? (
              <div style={{ color: '#64748b' }}>Loading jobs…</div>
            ) : (
              <JobTable
                jobs={jobs}
                selectedJobId={selectedJobId}
                onSelect={setSelectedJobId}
                onCopyWorkspace={handleCopyWorkspace}
                onCancel={handleCancelJob}
                pendingCancelId={pendingCancelId}
              />
            )}
            {jobQuery.isError && (
              <div className="message-banner error">Unable to load job details.</div>
            )}
            <JobDetail job={selectedJob} />
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
