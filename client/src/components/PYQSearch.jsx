import React, { useState } from 'react';
import { Search, Check, X, Filter } from 'lucide-react';
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

export default function PYQSearch({ user, onAttemptRecorded }) {
  const [query, setQuery] = useState('');
  const [selectedTopic, setSelectedTopic] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [hasSearched, setHasSearched] = useState(false);
  const [error, setError] = useState('');
  const [userAnswers, setUserAnswers] = useState({});

  const handleSearch = async (overrideQuery = null, overrideTopic = null) => {
    const q = overrideQuery !== null ? overrideQuery : query;
    const t = overrideTopic !== null ? overrideTopic : selectedTopic;
    const searchTerm = q.trim() || t || "General Awareness";

    setLoading(true);
    setError('');

    try {
      const res = await api.searchPYQs(searchTerm, t || null, null, 12);
      setResults(res.questions || []);
      setHasSearched(true);
    } catch (err) {
      setError(err.message || 'Failed to search questions');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectOption = async (q, optionText) => {
    if (userAnswers[q.id]) return;

    const cleanOption = optionText.trim().toLowerCase();
    const correctClean = (q.correct_answer || '').trim().toLowerCase();

    let isCorrect = false;
    if (correctClean) {
      const match = correctClean.match(/\(?([a-d1-4])\)?/);
      if (match) {
        const targetLetter = match[1];
        isCorrect = cleanOption.startsWith(`(${targetLetter})`) ||
                    cleanOption.startsWith(`${targetLetter}.`) ||
                    cleanOption.startsWith(`${targetLetter})`) ||
                    cleanOption.includes(correctClean);
      } else {
        isCorrect = cleanOption.includes(correctClean) || correctClean.includes(cleanOption);
      }
    }

    setUserAnswers(prev => ({
      ...prev,
      [q.id]: { selected: optionText, isCorrect }
    }));

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
          time_spent_seconds: 15.0
        });
        if (onAttemptRecorded) onAttemptRecorded();
      } catch (err) {
        console.error("Attempt log notice:", err);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Search Header Panel */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>ChromaDB Semantic PYQ Search</h2>
          <span className="badge badge-outline">all-MiniLM-L6-v2 (384d)</span>
        </div>
        <p style={{ fontSize: '0.78rem', color: '#888888', marginBottom: '14px' }}>
          Enter a topic, constitutional article, or historical keyword. Dense embeddings will pull matching exam questions from verified SSC papers.
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
              placeholder="e.g. 'Harappan civilization ports', 'Fundamental Rights', 'Mughal revenue system'..."
              className="input"
              style={{ paddingLeft: '38px' }}
            />
          </div>
          <button onClick={() => handleSearch()} disabled={loading} className="btn btn-primary">
            {loading ? 'Searching...' : 'Search PYQs'}
          </button>
        </div>

        {/* Topic Filter Chips */}
        <div>
          <div style={{ fontSize: '0.74rem', color: '#888888', display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '4px' }}>
            <Filter size={12} />
            <span>Select Topic Filter:</span>
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
          <div style={{ fontSize: '0.82rem', color: '#888888' }}>Querying ChromaDB vector embeddings & MongoDB...</div>
        </div>
      )}

      {/* Empty State */}
      {!loading && hasSearched && results.length === 0 && (
        <div className="card" style={{ padding: '2.5rem', textAlign: 'center', color: '#888888', fontSize: '0.85rem' }}>
          No questions found matching your search. Try different keywords or select a topic chip above.
        </div>
      )}

      {/* Results List */}
      {!loading && results.length > 0 && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#888888', marginBottom: '12px' }}>
            <span>Found <strong>{results.length}</strong> verified questions</span>
            <span>ChromaDB Vector Retrieval Active</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {results.map((q, idx) => {
              const answerState = userAnswers[q.id];

              return (
                <div key={q.id || idx} className="question-card">
                  {/* Metadata Header */}
                  <div className="question-header">
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', alignItems: 'center' }}>
                      <span className="badge badge-white-outline">{q.topic || 'General Awareness'}</span>
                      {q.shift && <span className="badge badge-outline">{q.shift}</span>}
                      {q.exam_date && <span className="badge badge-outline">{q.exam_date}</span>}
                    </div>
                    {q.similarity_score !== undefined && (
                      <span className="mono" style={{ fontSize: '0.72rem', color: '#aaaaaa' }}>
                        Score: {Math.round(q.similarity_score * 100)}%
                      </span>
                    )}
                  </div>

                  {/* Question Prompt */}
                  <div className="question-title">
                    <span style={{ fontWeight: 800, marginRight: '8px' }}>Q{idx + 1}.</span>
                    {q.question}
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

                  {/* Answer State Feedback */}
                  {answerState && (
                    <div className="explanation-box">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, marginBottom: '2px' }}>
                        {answerState.isCorrect ? <Check size={14} /> : <X size={14} />}
                        <span>{answerState.isCorrect ? "Correct Choice" : "Incorrect Selection"}</span>
                      </div>
                      <div style={{ color: '#aaaaaa', fontSize: '0.78rem' }}>
                        Official Answer: <strong style={{ color: '#ffffff' }}>{q.correct_answer || 'Refer to exam answer key'}</strong>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
