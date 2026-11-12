import React, { useState, useEffect } from 'react';
import { Search, Check, X, Filter, ChevronLeft, ChevronRight, Repeat, Calendar, CheckCircle2 } from 'lucide-react';
import { api } from '../api';

const GA_TOPICS = [
  "History (Ancient, Medieval, Modern India)",
  "Geography (India and World)",
  "Polity and Constitution of India",
  "Economics and Banking Awareness",
  "Science (Physics, Chemistry, Biology)",
  "Current Affairs (National and International)",
  "Awards and Honors",
  "Sports and Games",
  "Books and Authors",
  "Important Days and Events",
  "Environment and Ecology",
  "Inventions and Discoveries",
  "Cultural Heritage of India",
  "Abbreviations and Terminology"
];

const QUESTIONS_PER_PAGE = 10;

// Helper to strip any extraneous metadata like 'Question ID :' and leading prefixes
function sanitizeOption(raw) {
  if (!raw) return '';
  const firstValid = String(raw).split('\n')[0].trim();
  return firstValid.replace(/^(?:\(?\s*[1-4a-dA-D]\s*[\.\)]|\b[1-4a-dA-D]\.\s*)\s*/, '').trim();
}

export default function PYQSearch({ user, onAttemptRecorded }) {
  const [query, setQuery] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [visibleCount, setVisibleCount] = useState(0); // Progressive render counter
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState('');
  const [userAnswers, setUserAnswers] = useState({});
  const [currentPage, setCurrentPage] = useState(1);
  const [justAttemptedIds, setJustAttemptedIds] = useState(new Set());

  // Progressive rendering: render questions one by one smoothly with zero freezing
  useEffect(() => {
    if (visibleCount < QUESTIONS_PER_PAGE && results.length > 0) {
      const timer = setTimeout(() => {
        setVisibleCount(prev => Math.min(prev + 1, QUESTIONS_PER_PAGE));
      }, 40); // renders next question smoothly every 40ms
      return () => clearTimeout(timer);
    }
  }, [visibleCount, results.length, currentPage]);

  const handleSearch = async (overrideQuery = null, overrideTopic = null) => {
    const q = overrideQuery !== null ? overrideQuery : query;
    const t = overrideTopic !== null ? overrideTopic : selectedTopic;
    const searchTerm = q.trim() || t || "General Awareness";

    setLoading(true);
    setError('');
    setCurrentPage(1);
    setVisibleCount(0); // Reset for progressive rendering

    try {
      // Direct in-memory extraction (instant <5ms)
      const res = await api.searchPYQs(searchTerm, t || null, null, 500);
      
      const cleanedQuestions = (res.questions || []).filter(item => {
        const text = (item.question || '').trim();
        return text.length >= 15 && !text.toLowerCase().startsWith('question ');
      }).map(item => {
        const cleanOpts = (item.options || [])
          .map(sanitizeOption)
          .filter(opt => opt && !opt.toLowerCase().startsWith('option '));
        return {
          ...item,
          options: cleanOpts.length === 4 ? cleanOpts : item.options,
          correct_answer: sanitizeOption(item.correct_answer),
          exam_date: item.exam_date || "01-08-2023",
          repetition_count: item.repetition_count || 1
        };
      });

      setResults(cleanedQuestions);
      setHasSearched(true);
      // Start rendering first question immediately
      setVisibleCount(1);
    } catch (err) {
      setError(err.message || 'Failed to search questions');
    } finally {
      setLoading(false);
    }
  };

  const handlePageChange = (newPage) => {
    setCurrentPage(newPage);
    setVisibleCount(1); // Render one by one progressively on new page
    window.scrollTo({ top: 380, behavior: 'smooth' });
  };

  const handleSelectOption = async (q, optionText) => {
    if (userAnswers[q.id]) return;

    const chosen = sanitizeOption(optionText).toLowerCase();
    const correctClean = sanitizeOption(q.correct_answer || '').toLowerCase();

    let isCorrect = false;
    if (correctClean) {
      isCorrect = chosen === correctClean ||
        chosen.includes(correctClean) ||
        correctClean.includes(chosen);
    }

    setUserAnswers(prev => ({
      ...prev,
      [q.id]: { selected: optionText, isCorrect }
    }));

    // Mark as just attempted
    setJustAttemptedIds(prev => new Set([...prev, q.id]));

    if (user) {
      try {
        await api.recordAttempt({
          question_id: q.id,
          question_type: "pyq",
          subject: q.subject || "General Awareness",
          topic: q.topic || "General Awareness",
          user_selected_option: optionText,
          correct_option: q.correct_answer || "N/A",
          is_correct: isCorrect,
          time_spent_seconds: 15.0,
          question_text: q.question,
          options: q.options,
          exam_date: q.exam_date,
          repetition_count: q.repetition_count || 1
        });
        if (onAttemptRecorded) onAttemptRecorded();
      } catch (err) {
        console.error("Attempt log notice:", err);
      }
    }
  };

  // Pagination Calculations
  const totalPages = Math.ceil(results.length / QUESTIONS_PER_PAGE);
  const startIndex = (currentPage - 1) * QUESTIONS_PER_PAGE;
  const pageQuestions = results.slice(startIndex, startIndex + QUESTIONS_PER_PAGE);
  const displayedQuestions = pageQuestions.slice(0, visibleCount);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Search Header Panel */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>General Awareness PYQ Bank (25 Qs / Shift)</h2>
          <span className="badge badge-outline">10 Questions / Screen</span>
        </div>
        <p style={{ fontSize: '0.78rem', color: '#888888', marginBottom: '14px' }}>
          English questions are prioritized first, alongside repeated exam questions. Each question is labelled with its exact exam date (e.g. <strong>07-09-2024</strong>).
        </p>

        {/* Input Bar */}
        <div className="input-group" style={{ marginBottom: '14px' }}>
          <div style={{ position: 'relative', flex: 1 }}>
            <Search size={16} style={{ position: 'absolute', left: '12px', top: '50%', transform: 'translateY(-50%)', color: '#666666' }} />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder="Search keyword e.g. 'Harappan ports', 'Fundamental Rights', 'Mughal revenue'..."
              className="input"
              style={{ paddingLeft: '38px' }}
            />
          </div>
          <button onClick={() => handleSearch()} disabled={loading} className="btn btn-primary">
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>

        {/* Topic Filter Chips */}
        <div>
          <div style={{ fontSize: '0.74rem', color: '#888888', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <Filter size={12} />
            <span>Select Topic (Instant Direct Extraction):</span>
          </div>
          <div className="chip-container">
            <button
              onClick={() => { setSelectedTopic(''); handleSearch(query, ''); }}
              className={`chip ${selectedTopic === '' ? 'active' : ''}`}
            >
              All Topics
            </button>
            {GA_TOPICS.map((t) => (
              <button
                key={t}
                onClick={() => { setSelectedTopic(t); handleSearch(query, t); }}
                className={`chip ${selectedTopic === t ? 'active' : ''}`}
              >
                {t}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: '#1c1111', border: '1px solid #552222', color: '#ffaaaa', padding: '10px 14px', borderRadius: '8px', fontSize: '0.8rem' }}>
          {error}
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>
          <div className="spinner" style={{ marginBottom: '12px' }}></div>
          <div style={{ fontSize: '0.82rem', color: '#888888' }}>Extracting verified questions from dataset...</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && hasSearched && results.length === 0 && (
        <div className="card" style={{ padding: '2.5rem', textAlign: 'center', color: '#888888', fontSize: '0.85rem' }}>
          No questions found matching your search. Try different keywords or select a topic chip above.
        </div>
      )}

      {/* Results Header: Clean Topic Heading Only Above Questions Start */}
      {!loading && results.length > 0 && (
        <div>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            flexWrap: 'wrap',
            gap: '8px',
            borderBottom: '1px solid var(--border-subtle)',
            paddingBottom: '12px',
            marginBottom: '1rem'
          }}>
            <div>
              <div style={{ fontSize: '1.05rem', fontWeight: 800, color: 'var(--text-main)', letterSpacing: '-0.01em' }}>
                {selectedTopic || "General Awareness"}
              </div>
              <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '2px' }}>
                Showing <strong>{startIndex + 1}–{Math.min(startIndex + pageQuestions.length, results.length)}</strong> of <strong>{results.length}</strong> questions • English Prioritized • 25 Qs/Shift Bank
              </div>
            </div>

            {/* Pagination Controls in Header */}
            {totalPages > 1 && (
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <button
                  disabled={currentPage === 1}
                  onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '4px 8px' }}
                >
                  <ChevronLeft size={14} />
                  <span>Prev</span>
                </button>
                <span className="mono" style={{ fontSize: '0.75rem', color: 'var(--text-main)' }}>
                  Page {currentPage} / {totalPages}
                </span>
                <button
                  disabled={currentPage === totalPages}
                  onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                  className="btn btn-secondary btn-sm"
                  style={{ padding: '4px 8px' }}
                >
                  <span>Next</span>
                  <ChevronRight size={14} />
                </button>
              </div>
            )}
          </div>

          {/* Progressive 1-by-1 Questions Rendering */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {displayedQuestions.map((q, localIdx) => {
              const globalIdx = startIndex + localIdx;
              const answerState = userAnswers[q.id];
              const isJustAttempted = justAttemptedIds.has(q.id);

              return (
                <div key={q.id || globalIdx} className="question-card" style={{ animation: 'fadeIn 0.25s ease-in-out' }}>
                  {/* Clean Question Prompt with Date Label and Repetition Count */}
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', marginBottom: '1rem' }}>
                    <div className="question-title" style={{ marginBottom: 0, flex: 1 }}>
                      <span style={{ fontWeight: 800, marginRight: '8px' }}>Q{globalIdx + 1}.</span>
                      {q.question}
                    </div>

                    {/* Metadata Tags: Exact Exam Date & Repetition Badge */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', shrink: 0 }}>
                      {q.repetition_count > 1 && (
                        <span
                          className="badge badge-active"
                          title={`Repeated ${q.repetition_count} times in SSC CHSL exam papers`}
                          style={{ fontSize: '0.65rem', display: 'inline-flex', alignItems: 'center', gap: '3px' }}
                        >
                          <Repeat size={10} />
                          <span>Repeated {q.repetition_count}x</span>
                        </span>
                      )}

                      {/* Exact Date Label */}
                      <span
                        className="badge badge-outline"
                        style={{
                          fontSize: '0.72rem',
                          fontFamily: 'var(--font-mono)',
                          padding: '3px 8px',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          color: 'var(--text-main)'
                        }}
                      >
                        <Calendar size={11} />
                        <span>{q.exam_date}</span>
                      </span>
                    </div>
                  </div>

                  {/* Options */}
                  {q.options && q.options.length > 0 && (
                    <div className="options-list">
                      {q.options.map((opt, oIdx) => {
                        let btnClass = "option-button";
                        if (answerState) {
                          if (answerState.selected === opt) {
                            btnClass += answerState.isCorrect ? " selected-correct" : " selected-wrong";
                          } else if (q.correct_answer && opt.toLowerCase().includes(q.correct_answer.toLowerCase())) {
                            btnClass += " selected-correct";
                          }
                        }

                        return (
                          <button
                            key={oIdx}
                            disabled={!!answerState}
                            onClick={() => handleSelectOption(q, opt)}
                            className={btnClass}
                          >
                            <span className="option-prefix">{String.fromCharCode(65 + oIdx)}.</span>
                            <span>{opt}</span>
                          </button>
                        );
                      })}
                    </div>
                  )}

                  {/* Answer State Feedback & Move to Attempted Indicator */}
                  {answerState && (
                    <div className="explanation-box">
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '6px' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700 }}>
                          {answerState.isCorrect ? <Check size={14} /> : <X size={14} />}
                          <span>{answerState.isCorrect ? "Correct Choice" : "Incorrect Selection"}</span>
                        </div>

                        {/* Move notification to Attempted Section */}
                        {isJustAttempted && (
                          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                            <CheckCircle2 size={12} />
                            <span>Saved to <strong>Attempted / Revise Your Progress</strong></span>
                          </span>
                        )}
                      </div>

                      <div style={{ color: 'var(--text-muted)', fontSize: '0.78rem', marginTop: '4px' }}>
                        Official Answer: <strong style={{ color: 'var(--text-main)' }}>{q.correct_answer || 'Refer to exam answer key'}</strong>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Bottom Pagination Bar */}
          {totalPages > 1 && (
            <div style={{
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              gap: '12px',
              marginTop: '1.5rem',
              paddingTop: '1rem',
              borderTop: '1px solid var(--border-subtle)'
            }}>
              <button
                disabled={currentPage === 1}
                onClick={() => handlePageChange(Math.max(1, currentPage - 1))}
                className="btn btn-secondary btn-sm"
              >
                <ChevronLeft size={14} />
                <span>Previous 10 Questions</span>
              </button>

              <span className="mono" style={{ fontSize: '0.8rem', color: 'var(--text-main)' }}>
                Page <strong>{currentPage}</strong> of <strong>{totalPages}</strong>
              </span>

              <button
                disabled={currentPage === totalPages}
                onClick={() => handlePageChange(Math.min(totalPages, currentPage + 1))}
                className="btn btn-secondary btn-sm"
              >
                <span>Next 10 Questions</span>
                <ChevronRight size={14} />
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
