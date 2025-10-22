import React, { useState } from 'react';
import FilePicker from './FilePicker';

const TrainingPairForm: React.FC = () => {
    const [mediaPath, setMediaPath] = useState('');
    const [srtPath, setSrtPath] = useState('');

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        const response = await fetch('/api/jobs/training-pair', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ media_path: mediaPath, srt_path: srtPath }),
        });
        if (response.ok) {
            alert('Training pair job created successfully!');
        } else {
            alert('Failed to create training pair job.');
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2>Create Training Pair Job</h2>
            <div>
                <label>Media File:</label>
                <input type="text" value={mediaPath} readOnly />
                <FilePicker onFileSelect={setMediaPath} />
            </div>
            <div>
                <label>SRT File:</label>
                <input type="text" value={srtPath} readOnly />
                <FilePicker onFileSelect={setSrtPath} />
            </div>
            <button type="submit">Create Job</button>
        </form>
    );
};

export default TrainingPairForm;
