import React, { useState } from 'react';
import JobsDashboard from './components/JobsDashboard';
import './App.css';

import InferenceForm from './components/InferenceForm';

// Placeholder components for each tab
const Inference: React.FC = () => <InferenceForm />;
import TrainingPairForm from './components/TrainingPairForm';

const TrainingData: React.FC = () => <TrainingPairForm />;
import ModelTrainingForm from './components/ModelTrainingForm';

const ModelTraining: React.FC = () => <ModelTrainingForm />;
import ConfigurationEditor from './components/Configuration';

const Configuration: React.FC = () => <ConfigurationEditor />;

const App: React.FC = () => {
    const [activeTab, setActiveTab] = useState('jobs');

    const renderTabContent = () => {
        switch (activeTab) {
            case 'inference': return <Inference />;
            case 'training-data': return <TrainingData />;
            case 'model-training': return <ModelTraining />;
            case 'jobs': return <JobsDashboard />;
            case 'configuration': return <Configuration />;
            default: return <JobsDashboard />;
        }
    };

    return (
        <div className="app-container">
            <header className="app-header">
                <h1>ISCE Caption Pipeline</h1>
                <nav className="app-nav">
                    <button onClick={() => setActiveTab('jobs')} className={activeTab === 'jobs' ? 'active' : ''}>Jobs</button>
                    <button onClick={() => setActiveTab('inference')} className={activeTab === 'inference' ? 'active' : ''}>Inference</button>
                    <button onClick={() => setActiveTab('training-data')} className={activeTab === 'training-data' ? 'active' : ''}>Training Data</button>
                    <button onClick={() => setActiveTab('model-training')} className={activeTab === 'model-training' ? 'active' : ''}>Model Training</button>
                    <button onClick={() => setActiveTab('configuration')} className={activeTab === 'configuration' ? 'active' : ''}>Configuration</button>
                </nav>
            </header>
            <main className="app-main">
                {renderTabContent()}
            </main>
        </div>
    );
};

export default App;
