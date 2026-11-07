import React, { useState } from 'react';
import { BookOpen, Check, X, FileText } from 'lucide-react';
import { api } from '../api';

const QUICK_PAGES = [
  { page: 6, label: "Indus Valley (P. 6)" },
  { page: 12, label: "Vedic Culture (P. 12)" },
  { page: 24, label: "Buddhism & Jainism (P. 24)" },
  { page: 55, label: "Delhi Sultanate (P. 55)" },
  { page: 85, label: "Mughal Empire (P. 85)" },
  { page: 140, label: "Geography of India (P. 140)" },
  { page: 210, label: "Constitution of India (P. 210)" }
];

export default function LucentQuestionGen({ user, onAttemptRecorded }) {
  const [pageNumber, setPageNumber] = useState(6);
  const [questionCount, setQuestionCount] = useState(3);
  const [loading, setLoading] = useState(false);
  const [pagePreview, setPagePreview] = useState(null);
  const [generatedData, setGeneratedData] = useState(null);
  const [error, setError] = useState('');
  const [userAnswers, setUserAnswers] = useState({});

  const handleGenerate = async (targetPage = null) => {
    const page = targetPage !== null ? targetPage : parseInt(pageNumber, 10);
    if (isNaN(page) || page < 1 || page > 672) {
      setError("Please specify a valid page number between 1 and 672.");
      return;
    }

    setLoading(true);
    setError('');
    setUserAnswers({});

    try {
      const preview = await api.getLucentPage(page);
      setPagePreview(preview);

      const res = await api.generateLucentQuestions(page, questionCount);
      setGeneratedData(res);
    } catch (err) {
      setError(err.message || 'Failed to generate questions from Lucent page.');
    } finally {
      setLoading(false);
    }
  };

  const handleSelectOption = async (q, optionText) => {
    if (userAnswers[q.id]) return;

    const isCorrect = optionText === q.correct_answer || optionText.startsWith("(a)");

    setUserAnswers(prev => ({
      ...prev,
      [q.id]: { selected: optionText, isCorrect }
    }));

    if (user) {
      try {
        await api.recordAttempt({
          question_id: q.id,
          question_type: "lucent_generated",
          subject: "General Awareness",
          topic: q.topic || "General Awareness",
          user_selected_option: optionText,
          correct_option: q.correct_answer,
          is_correct: isCorrect,
          time_spent_seconds: 20.0
        });
        if (onAttemptRecorded) onAttemptRecorded();
      } catch (err) {
        console.error("Attempt log notice:", err);
      }
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Control Panel */}
      <div className="card">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 700 }}>Lucent GK Question Generator</h2>
          <span className="badge badge-outline">672 Total Pages</span>
        </div>
        <p style={{ fontSize: '0.78rem', color: '#888888', marginBottom: '14px' }}>
          Specify any page number from the Lucent GK English Edition. The system reads the extracted page text, identifies factual statements, and formulates multiple-choice questions.
        </p>

        {/* Input Bar */}
        <div className="input-group" style={{ marginBottom: '14px', flexWrap: 'wrap' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: '#888888', fontWeight: 600 }}>Page Number:</span>
            <input
              type="number"
              min={1}
              max={672}
              value={pageNumber}
              onChange={(e) => setPageNumber(e.target.value)}
              className="input mono"
              style={{ width: '80px', textAlign: 'center' }}
            />
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.78rem', color: '#888888', fontWeight: 600 }}>Count:</span>
            <select
              value={questionCount}
              onChange={(e) => setQuestionCount(parseInt(e.target.value, 10))}
              className="select"
            >
              <option value={2}>2 Questions</option>
              <option value={3}>3 Questions</option>
              <option value={5}>5 Questions</option>
            </select>
          </div>

          <button onClick={() => handleGenerate()} disabled={loading} className="btn btn-primary">
            {loading ? 'Synthesizing...' : 'Generate Questions'}
          </button>
        </div>

        {/* Quick Page Jump Chips */}
        <div>
          <span style={{ fontSize: '0.72rem', color: '#666666', marginRight: '6px' }}>Curated Pages:</span>
          <div className="chip-container" style={{ display: 'inline-flex' }}>
            {QUICK_PAGES.map((qp) => (
              <button
                key={qp.page}
                onClick={() => { setPageNumber(qp.page); handleGenerate(qp.page); }}
                className="chip"
              >
                {qp.label}
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

      {/* Page Preview Box */}
      {pagePreview && (
        <div className="card" style={{ padding: '1rem', background: '#080808' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', borderBottom: '1px solid #222222', paddingBottom: '6px', marginBottom: '8px' }}>
            <span style={{ fontWeight: 700 }}>Lucent GK Book — Page {pagePreview.page_number}</span>
            <span className="mono text-muted">{pagePreview.text ? pagePreview.text.length : 0} chars</span>
          </div>
          {pagePreview.relative_headings && pagePreview.relative_headings.length > 0 && (
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              <span style={{ fontSize: '0.72rem', color: '#666666' }}>Headings:</span>
              {pagePreview.relative_headings.map((h, i) => (
                <span key={i} className="badge badge-outline" style={{ fontSize: '0.68rem' }}>{h}</span>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Loading Spinner */}
      {loading && (
        <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>
          <div className="spinner" style={{ marginBottom: '12px' }}></div>
          <div style={{ fontSize: '0.82rem', color: '#888888' }}>Reading Page {pageNumber} & synthesizing MCQs...</div>
        </div>
      )}

      {/* Generated Questions List */}
      {!loading && generatedData && (
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', color: '#888888', marginBottom: '12px' }}>
            <span>Generated <strong>{generatedData.questions?.length || 0}</strong> questions from Page {generatedData.page_number}</span>
            <span>Topic: {generatedData.topic}</span>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {generatedData.questions?.map((q, idx) => {
              const answerState = userAnswers[q.id];

              return (
                <div key={q.id || idx} className="question-card">
                  <div className="question-header">
                    <span className="badge badge-white-outline">{q.topic || 'General Awareness'}</span>
                    <span className="mono" style={{ fontSize: '0.72rem', color: '#888888' }}>Page {q.page_number}</span>
                  </div>

                  <div className="question-title">
                    <span style={{ fontWeight: 800, marginRight: '8px' }}>Q{idx + 1}.</span>
                    {q.question}
                  </div>

                  <div className="options-list">
                    {q.options.map((opt, oIdx) => {
                      let btnClass = "option-button";
                      if (answerState) {
                        if (answerState.selected === opt) {
                          btnClass += answerState.isCorrect ? " selected-correct" : " selected-wrong";
                        } else if (opt === q.correct_answer || opt.startsWith("(a)")) {
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

                  {answerState && (
                    <div className="explanation-box">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 700, marginBottom: '2px' }}>
                        {answerState.isCorrect ? <Check size={14} /> : <X size={14} />}
                        <span>{answerState.isCorrect ? "Correct!" : "Incorrect"}</span>
                      </div>
                      <div style={{ color: '#aaaaaa', fontSize: '0.78rem' }}>
                        {q.explanation}
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
