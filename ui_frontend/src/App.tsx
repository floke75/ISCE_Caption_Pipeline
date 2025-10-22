import { useEffect, useMemo, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
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

  const jobQuery = useQuery({
    queryKey: ['job', selectedJobId],
    queryFn: () => fetchJob(selectedJobId as string),
    enabled: Boolean(selectedJobId),
    refetchInterval: selectedJobId ? 3000 : false,
  });

  const selectedJob: JobDetailType | undefined = jobQuery.data;

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
            {jobsError && (
              <div className="message-banner error">Failed to load job list.</div>
            )}
            {jobsLoading ? (
              <div style={{ color: '#64748b' }}>Loading jobs…</div>
            ) : (
              <JobTable jobs={jobs} selectedJobId={selectedJobId} onSelect={setSelectedJobId} />
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
