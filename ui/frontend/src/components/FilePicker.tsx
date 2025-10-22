import React, { useState, useEffect } from 'react';

interface File {
    name: string;
    path: string;
    is_dir: boolean;
}

interface FilePickerProps {
    onFileSelect: (path: string) => void;
    selectDirectory?: boolean;
}

const FilePicker: React.FC<FilePickerProps> = ({ onFileSelect, selectDirectory = false }) => {
    const [path, setPath] = useState('');
    const [files, setFiles] = useState<File[]>([]);
    const [parent, setParent] = useState<File | null>(null);

    const fetchFiles = async (newPath: string) => {
        try {
            const response = await fetch(`/api/files?path=${encodeURIComponent(newPath)}`);
            const data = await response.json();
            setPath(data.path);
            setFiles(data.items);
            setParent(data.parent);
        } catch (error) {
            console.error('Failed to fetch files:', error);
        }
    };

    useEffect(() => {
        fetchFiles(''); // Fetch root by default
    }, []);

    const handleSelect = (file: File) => {
        if (file.is_dir) {
            fetchFiles(file.path);
        } else if (!selectDirectory) {
            onFileSelect(file.path);
        }
    };

    return (
        <div>
            <h4>Current Path: {path}</h4>
            {selectDirectory && <button type="button" onClick={() => onFileSelect(path)}>Select Current Directory</button>}
            <ul>
                {parent && <li onClick={() => fetchFiles(parent.path)}>..</li>}
                {files.map(file => (
                    <li key={file.path} onClick={() => handleSelect(file)}>
                        {file.is_dir ? `${file.name}/` : file.name}
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default FilePicker;
