import React, { useEffect, useState } from 'react';

// Define the Job type based on the backend model
interface Job {
    id: string;
    type: string;
    status: string;
    created_at: string;
    error?: string;
}

const JobsDashboard: React.FC = () => {
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchJobs = async () => {
            try {
                const response = await fetch('/api/jobs');
                if (!response.ok) {
                    throw new Error('Failed to fetch jobs');
                }
                const data = await response.json();
                setJobs(data);
            } catch (err: any) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchJobs();
        const interval = setInterval(fetchJobs, 5000); // Poll every 5 seconds
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div>Loading jobs...</div>;
    if (error) return <div>Error: {error}</div>;

    return (
        <div>
            <h2>Jobs Dashboard</h2>
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Type</th>
                        <th>Status</th>
                        <th>Created At</th>
                        <th>Error</th>
                    </tr>
                </thead>
                <tbody>
                    {jobs.map((job) => (
                        <tr key={job.id}>
                            <td>{job.id}</td>
                            <td>{job.type}</td>
                            <td>{job.status}</td>
                            <td>{new Date(job.created_at).toLocaleString()}</td>
                            <td>{job.error}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

export default JobsDashboard;
