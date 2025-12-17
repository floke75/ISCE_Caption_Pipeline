import React, { useState, useEffect, useCallback } from 'react';
import toast from 'react-hot-toast';
import { templateService, type JobTemplate, type TemplateData, type JobType } from '../services/templateService';
import '../styles/forms.css';

interface Props {
  type: JobType;
  onLoad: (data: TemplateData) => void;
  getDataToSave: () => TemplateData;
}

export function TemplateSelector({ type, onLoad, getDataToSave }: Props) {
  const [templates, setTemplates] = useState<JobTemplate[]>([]);
  const [isSaving, setIsSaving] = useState(false);
  const [newTemplateName, setNewTemplateName] = useState('');
  const [selectedId, setSelectedId] = useState('');

  const refresh = useCallback(() => {
    setTemplates(templateService.getTemplates(type));
  }, [type]);

  useEffect(() => {
    refresh();
    setSelectedId(''); // Reset selection on type change
  }, [refresh]);

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTemplateName.trim()) {
      toast.error('Please enter a name');
      return;
    }
    const data = getDataToSave();
    try {
      const saved = templateService.saveTemplate(newTemplateName.trim(), type, data);
      toast.success('Template saved');
      setIsSaving(false);
      setNewTemplateName('');
      refresh();
      setSelectedId(saved.id); // Auto-select the new one
    } catch (err) {
      console.error(err);
      toast.error('Failed to save template');
    }
  };

  const handleDelete = () => {
    if (!selectedId) return;
    if (confirm('Are you sure you want to delete this template?')) {
      templateService.deleteTemplate(selectedId);
      toast.success('Template deleted');
      refresh();
      setSelectedId('');
    }
  };

  const handleLoad = () => {
    if (!selectedId) return;
    const template = templates.find(t => t.id === selectedId);
    if (template) {
      onLoad(template.data);
      toast.success(`Loaded template: ${template.name}`);
    }
  };

  return (
    <div className="template-selector">
      <div className="template-controls">
        <div className="template-group">
          <label htmlFor="template-select" className="template-label">Template:</label>
          <select
            id="template-select"
            value={selectedId}
            onChange={(e) => setSelectedId(e.target.value)}
            className="template-dropdown"
          >
            <option value="">-- Select a template --</option>
            {templates.map(t => (
              <option key={t.id} value={t.id}>{t.name} ({new Date(t.createdAt).toLocaleDateString()})</option>
            ))}
          </select>
          <button
            type="button"
            className="secondary small"
            onClick={handleLoad}
            disabled={!selectedId}
          >
            Load
          </button>
          {selectedId && (
             <button
                type="button"
                className="danger small"
                onClick={handleDelete}
                title="Delete selected template"
             >
                ×
             </button>
          )}
        </div>

        {!isSaving ? (
          <button
            type="button"
            className="secondary small"
            onClick={() => setIsSaving(true)}
          >
            Save current as...
          </button>
        ) : (
          <form onSubmit={handleSave} className="template-save-form">
            <input
              type="text"
              value={newTemplateName}
              onChange={(e) => setNewTemplateName(e.target.value)}
              placeholder="Template name"
              autoFocus
              className="template-input"
            />
            <button type="submit" className="primary small">Save</button>
            <button
              type="button"
              className="secondary small"
              onClick={() => { setIsSaving(false); setNewTemplateName(''); }}
            >
              Cancel
            </button>
          </form>
        )}
      </div>
    </div>
  );
}
