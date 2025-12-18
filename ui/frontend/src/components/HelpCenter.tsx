import { useState } from 'react';
import '../styles/help-center.css';

type Section = 'quickstart' | 'glossary' | 'faq';

const GLOSSARY = [
  { term: 'Beam Width', def: 'How many alternative segmentation paths the model explores. Higher values (e.g., 5-10) improve accuracy but increase processing time.' },
  { term: 'Diarization', def: 'The process of partitioning an audio stream into homogeneous segments according to the speaker identity.' },
  { term: 'Enriched Tokens', def: 'The intermediate data format where words are aligned with their timing, pause duration, and linguistic features.' },
  { term: 'Pause Z-Score', def: 'A normalized measure of silence duration between words, used to detect natural breaks in speech.' },
  { term: 'SRT', def: 'SubRip Subtitle file format, a standard text file format for subtitles.' },
];

const QUICKSTART = [
  { id: 1, text: 'Upload your media file (audio/video) or use an existing workspace.' },
  { id: 2, text: 'Choose the "Inference" tab for generating subtitles.' },
  { id: 3, text: 'Select a Preset (e.g., "Standard") to configure optimal settings.' },
  { id: 4, text: 'Click "Run Inference" and monitor progress in the Job Board.' },
  { id: 5, text: 'Once complete, download the SRT file or use the "Visualise Alignment" tool.' },
];

const FAQ = [
  { q: 'Why is my job stuck in "Pending"?', a: 'The backend worker might be busy with another job or not running. Check the System Status indicator.' },
  { q: 'What does "Error Boost" do?', a: 'It increases the penalty for misclassifying breaks during training, forcing the model to be more conservative.' },
  { q: 'Can I edit the subtitles?', a: 'Yes, download the SRT file and edit it in any text editor or subtitle software.' },
];

export function HelpCenter() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<Section>('quickstart');
  const [searchTerm, setSearchTerm] = useState('');

  const toggleOpen = () => setIsOpen(!isOpen);

  const filteredGlossary = GLOSSARY.filter(item =>
    item.term.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.def.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <>
      <button
        type="button"
        className="help-toggle-btn"
        onClick={toggleOpen}
        title="Open Help Center"
      >
        ?
      </button>

      {isOpen && (
        <div className="help-drawer-overlay" onClick={toggleOpen}>
          <div className="help-drawer" onClick={e => e.stopPropagation()}>
            <div className="drawer-header">
              <h2>Help Center</h2>
              <button type="button" className="close-btn" onClick={toggleOpen}>×</button>
            </div>

            <div className="search-bar">
              <input
                type="text"
                className="search-input"
                placeholder="Search glossary..."
                value={searchTerm}
                onChange={e => {
                    setSearchTerm(e.target.value);
                    if (e.target.value) setActiveSection('glossary');
                }}
              />
            </div>

            <div className="drawer-nav">
              <button
                className={`nav-item ${activeSection === 'quickstart' ? 'active' : ''}`}
                onClick={() => setActiveSection('quickstart')}
              >
                Quickstart Checklist
              </button>
              <button
                className={`nav-item ${activeSection === 'glossary' ? 'active' : ''}`}
                onClick={() => setActiveSection('glossary')}
              >
                Glossary
              </button>
              <button
                className={`nav-item ${activeSection === 'faq' ? 'active' : ''}`}
                onClick={() => setActiveSection('faq')}
              >
                FAQ / Troubleshooting
              </button>
            </div>

            <div className="drawer-content">
              {activeSection === 'quickstart' && (
                <div className="section-content">
                  <h3>Getting Started</h3>
                  <ul className="checklist">
                    {QUICKSTART.map(item => (
                      <li key={item.id}>
                        <span className="step-num">{item.id}</span>
                        <span>{item.text}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {activeSection === 'glossary' && (
                <div className="section-content">
                  <h3>Glossary</h3>
                  {filteredGlossary.length === 0 ? (
                    <p className="empty-state">No terms found.</p>
                  ) : (
                    <dl className="glossary-list">
                      {filteredGlossary.map(item => (
                        <div key={item.term} className="glossary-item">
                          <dt>{item.term}</dt>
                          <dd>{item.def}</dd>
                        </div>
                      ))}
                    </dl>
                  )}
                </div>
              )}

              {activeSection === 'faq' && (
                <div className="section-content">
                  <h3>Common Questions</h3>
                  <div className="faq-list">
                    {FAQ.map((item, idx) => (
                      <div key={idx} className="faq-item">
                        <div className="question">{item.q}</div>
                        <div className="answer">{item.a}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="drawer-footer">
              <a
                href="https://github.com/floke75/ISCE_Caption_Pipeline"
                target="_blank"
                rel="noreferrer"
                className="repo-link"
              >
                View Full Documentation
              </a>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
