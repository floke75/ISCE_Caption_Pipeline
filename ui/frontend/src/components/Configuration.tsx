import React, { useEffect, useState } from 'react';
import ConfigForm from './ConfigForm';

const ConfigurationEditor: React.FC = () => {
    const [pipelineConfig, setPipelineConfig] = useState<any>(null);
    const [modelConfig, setModelConfig] = useState<any>(null);

    useEffect(() => {
        const fetchConfigs = async () => {
            try {
                const pipelineResponse = await fetch('/api/config/pipeline');
                const pipelineData = await pipelineResponse.json();
                setPipelineConfig(pipelineData);

                const modelResponse = await fetch('/api/config/model');
                const modelData = await modelResponse.json();
                setModelConfig(modelData);
            } catch (error) {
                console.error('Failed to fetch configs:', error);
            }
        };

        fetchConfigs();
    }, []);

    const handleSavePipelineConfig = async () => {
        try {
            const response = await fetch('/api/config/pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(pipelineConfig),
            });
            if (response.ok) {
                alert('Pipeline config saved successfully!');
            } else {
                alert('Failed to save pipeline config.');
            }
        } catch (error) {
            alert('Failed to save pipeline config.');
        }
    };

    const handleSaveModelConfig = async () => {
        try {
            const response = await fetch('/api/config/model', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(modelConfig),
            });
            if (response.ok) {
                alert('Model config saved successfully!');
            } else {
                alert('Failed to save model config.');
            }
        } catch (error) {
            alert('Failed to save model config.');
        }
    };

    return (
        <div>
            <h2>Configuration</h2>
            <div>
                <h3>pipeline_config.yaml</h3>
                {pipelineConfig && <ConfigForm config={pipelineConfig} onConfigChange={setPipelineConfig} />}
                <button onClick={handleSavePipelineConfig}>Save</button>
            </div>
            <div>
                <h3>config.yaml</h3>
                {modelConfig && <ConfigForm config={modelConfig} onConfigChange={setModelConfig} />}
                <button onClick={handleSaveModelConfig}>Save</button>
            </div>
        </div>
    );
};

export default ConfigurationEditor;
