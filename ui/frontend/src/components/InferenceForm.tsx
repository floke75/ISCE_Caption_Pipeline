import React, { useState } from 'react';
import FilePicker from './FilePicker';

const InferenceForm: React.FC = () => {
    const [mediaPath, setMediaPath] = useState('');
    const [transcriptPath, setTranscriptPath] = useState('');

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        const response = await fetch('/api/jobs/inference', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ media_path: mediaPath, transcript_path: transcriptPath }),
        });
        if (response.ok) {
            alert('Inference job created successfully!');
        } else {
            alert('Failed to create inference job.');
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2>Create Inference Job</h2>
            <div>
                <label>Media File:</label>
                <input type="text" value={mediaPath} readOnly />
                <FilePicker onFileSelect={setMediaPath} />
            </div>
            <div>
                <label>Transcript File (Optional):</label>
                <input type="text" value={transcriptPath} readOnly />
                <FilePicker onFileSelect={setTranscriptPath} />
            </div>
            <button type="submit">Create Job</button>
        </form>
    );
};

export default InferenceForm;
