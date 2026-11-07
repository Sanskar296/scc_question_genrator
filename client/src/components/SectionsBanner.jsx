import React from 'react';
import { Compass, FileText, Brain, Calculator, Info } from 'lucide-react';

export default function SectionsBanner({ sectionsData }) {
  const sections = sectionsData?.sections || [
    {
      id: 'general_awareness',
      name: 'General Awareness',
      status: 'Active Focus',
      is_active: true,
      weightage: '25 Questions (50 Marks)',
      sources: ['Lucent GK Book (672 Pages)', '12,968 Verified CHSL PYQs'],
      topics_count: 14
    },
    {
      id: 'english',
      name: 'English',
      status: 'Coming Soon',
      is_active: false,
      weightage: '25 Questions (50 Marks)',
      sources: ['Syllabus Configured'],
      topics_count: 14
    },
    {
      id: 'general_intelligence',
      name: 'General Intelligence',
      status: 'Coming Soon',
      is_active: false,
      weightage: '25 Questions (50 Marks)',
      sources: ['Syllabus Configured'],
      topics_count: 18
    },
    {
      id: 'quantitative_aptitude',
      name: 'Quantitative Aptitude',
      status: 'Coming Soon',
      is_active: false,
      weightage: '25 Questions (50 Marks)',
      sources: ['Syllabus Configured'],
      topics_count: 20
    }
  ];

  const getSectionIcon = (id) => {
    switch (id) {
      case 'general_awareness':
        return <Compass size={18} />;
      case 'english':
        return <FileText size={18} />;
      case 'general_intelligence':
        return <Brain size={18} />;
      case 'quantitative_aptitude':
        return <Calculator size={18} />;
      default:
        return <Compass size={18} />;
    }
  };

  return (
    <div style={{ marginBottom: '2rem' }}>
      {/* Black & White Directive Notice Box */}
      <div className="notice-box">
        <Info size={20} style={{ shrink: 0, marginTop: '2px', color: '#ffffff' }} />
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
            <span style={{ fontWeight: 700, fontSize: '0.85rem' }}>Curriculum Directive:</span>
            <span className="badge badge-active">General Awareness Active Focus</span>
          </div>
          <p style={{ fontSize: '0.78rem', color: '#a3a3a3', lineHeight: 1.6 }}>
            While English, General Intelligence, and Quantitative Aptitude sections are listed, <strong>General Awareness is the primarily active focus right now</strong>. The question bank is exclusively powered by the <strong>Lucent GK book (672 pages)</strong> and <strong>12,968 verified previous year questions</strong> indexed with ChromaDB vector embeddings.
          </p>
        </div>
      </div>

      {/* 4 Sections Grid */}
      <div className="sections-grid">
        {sections.map((sec) => (
          <div
            key={sec.id}
            className={`card ${sec.is_active ? 'card-active' : ''}`}
            style={{ opacity: sec.is_active ? 1 : 0.6 }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
              <div style={{
                width: '32px',
                height: '32px',
                background: sec.is_active ? '#ffffff' : '#171717',
                color: sec.is_active ? '#000000' : '#888888',
                borderRadius: '6px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}>
                {getSectionIcon(sec.id)}
              </div>
              <span className={`badge ${sec.is_active ? 'badge-active' : 'badge-inactive'}`}>
                {sec.status}
              </span>
            </div>

            <h3 style={{ fontSize: '0.92rem', fontWeight: 700, marginBottom: '2px' }}>{sec.name}</h3>
            <p className="mono" style={{ fontSize: '0.72rem', color: '#888888', marginBottom: '12px' }}>{sec.weightage}</p>

            <div style={{ borderTop: '1px solid #222222', paddingTop: '10px', fontSize: '0.74rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span className="text-muted">Topics:</span>
                <span style={{ fontWeight: 600 }}>{sec.topics_count} Subtopics</span>
              </div>
              <div style={{ color: '#777777', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {sec.sources ? sec.sources.join(', ') : 'Standard Tier I'}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
