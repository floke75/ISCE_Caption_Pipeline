import React from 'react';

interface ConfigFormProps {
    config: any;
    onConfigChange: (newConfig: any) => void;
}

const ConfigForm: React.FC<ConfigFormProps> = ({ config, onConfigChange }) => {
    const handleChange = (key: string, value: any) => {
        onConfigChange({ ...config, [key]: value });
    };

    return (
        <div style={{ marginLeft: '20px', borderLeft: '1px solid #eee', paddingLeft: '15px' }}>
            {Object.entries(config).map(([key, value]) => (
                <div key={key} style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                    <label style={{ minWidth: '200px', fontWeight: '500' }}>{key}:</label>
                    {(() => {
                        if (typeof value === 'boolean') {
                            return (
                                <input
                                    type="checkbox"
                                    checked={value}
                                    onChange={(e) => handleChange(key, e.target.checked)}
                                    style={{ height: '20px', width: '20px' }}
                                />
                            );
                        }
                        if (typeof value === 'number') {
                            return (
                                <input
                                    type="number"
                                    value={value}
                                    onChange={(e) => handleChange(key, parseFloat(e.target.value) || 0)}
                                    style={{ padding: '5px', width: '100px' }}
                                />
                            );
                        }
                        if (typeof value === 'object' && value !== null) {
                            return <ConfigForm config={value} onConfigChange={(newValue) => handleChange(key, newValue)} />;
                        }
                        return (
                            <input
                                type="text"
                                value={String(value ?? '')}
                                onChange={(e) => handleChange(key, e.target.value)}
                                style={{ padding: '5px', width: '400px' }}
                            />
                        );
                    })()}
                </div>
            ))}
        </div>
    );
};

export default ConfigForm;
