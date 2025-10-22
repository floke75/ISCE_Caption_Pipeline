import React, { useState } from 'react';
import FilePicker from './FilePicker';

const ModelTrainingForm: React.FC = () => {
    const [corpusDir, setCorpusDir] = useState('');
    const [constraintsPath, setConstraintsPath] = useState('');
    const [weightsPath, setWeightsPath] = useState('');

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        const response = await fetch('/api/jobs/model-training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                corpus_dir: corpusDir,
                constraints_path: constraintsPath,
                weights_path: weightsPath,
            }),
        });
        if (response.ok) {
            alert('Model training job created successfully!');
        } else {
            alert('Failed to create model training job.');
        }
    };

    return (
        <form onSubmit={handleSubmit}>
            <h2>Create Model Training Job</h2>
            <div>
                <label>Corpus Directory:</label>
                <input type="text" value={corpusDir} readOnly />
                <FilePicker onFileSelect={setCorpusDir} selectDirectory />
            </div>
            <div>
                <label>Constraints Path:</label>
                <input type="text" value={constraintsPath} onChange={e => setConstraintsPath(e.target.value)} />
            </div>
            <div>
                <label>Weights Path:</label>
                <input type="text" value={weightsPath} onChange={e => setWeightsPath(e.target.value)} />
            </div>
            <button type="submit">Create Job</button>
        </form>
    );
};

export default ModelTrainingForm;
