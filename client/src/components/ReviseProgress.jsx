import React, { useState, useEffect } from 'react';
import { api } from '../api';
import { RefreshCw, CheckCircle2, XCircle, RotateCcw, Calendar, Repeat, ArrowRight, Trash2, Award } from 'lucide-react';

export default function ReviseProgress({ user, onAttemptRecorded }) {
  const [attemptedList, setAttemptedList] = useState([]);
  const [filterStatus, setFilterStatus] = useState('all'); // 'all' | 'incorrect' | 'correct'
  const [loading, setLoading] = useState(true);
  const [selectedAnswers, setSelectedAnswers] = useState({});
  const [submittingId, setSubmittingId] = useState(null);

  useEffect(() => {
    loadAttempted();
  }, [filterStatus]);

  const loadAttempted = async () => {
    setLoading(true);
    try {
      const data = await api.getAttemptedQuestions(filterStatus);
      setAttemptedList(data.attempted_questions || []);
    } catch (err) {
      console.error('Failed to load attempted questions:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleSelectOption = (qId, optKey) => {
    setSelectedAnswers(prev => ({ ...prev, [qId]: optKey }));
  };

  const handleReattempt = async (item) => {
    const qId = item.question_id;
    const selectedOpt = selectedAnswers[qId];
    if (!selectedOpt) return;

    setSubmittingId(qId);
    try {
      const res = await api.recordAttempt({
        question_id: qId,
        question_text: item.question_text,
        chosen_option: selectedOpt,
        topic: item.topic || 'General Awareness',
        source_type: 'pyq',
        exam_date: item.exam_date,
        repetition_count: item.repetition_count || 1,
        options: item.options
      });

      // Update local item with new attempt result
      setAttemptedList(prev =>
        prev.map(q => {
          if (q.question_id === qId) {
            return {
              ...q,
              attempt_count: (q.attempt_count || 1) + 1,
              is_correct: res.is_correct,
              last_chosen_option: selectedOpt,
              correct_option: res.correct_option,
              has_reevaluated: true
            };
          }
          return q;
        })
      );

      if (onAttemptRecorded) {
        onAttemptRecorded();
      }
    } catch (err) {
      console.error('Failed to submit reattempt:', err);
    } finally {
      setSubmittingId(null);
    }
  };

  const handleClear = async (qId) => {
    try {
      await api.clearAttemptedQuestion(qId);
      setAttemptedList(prev => prev.filter(q => q.question_id !== qId));
    } catch (err) {
      console.error('Failed to clear question:', err);
    }
  };

  const correctCount = attemptedList.filter(q => q.is_correct).length;
  const incorrectCount = attemptedList.length - correctCount;
  const accuracyPct = attemptedList.length > 0 ? Math.round((correctCount / attemptedList.length) * 100) : 0;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header & Stats Banner */}
      <div className="card" style={{ padding: '1.25rem 1.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700, margin: 0, display: 'flex', alignItems: 'center', gap: '8px' }}>
            <RotateCcw size={18} /> Revise Your Progress
          </h2>
          <p style={{ margin: '4px 0 0 0', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            All your attempted questions are organized here. Re-attempt weak areas and reinforce your accuracy.
          </p>
        </div>

        {/* Quick summary stats */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center' }}>
          <div className="badge badge-outline" style={{ padding: '6px 12px', fontSize: '0.75rem' }}>
            Total: <strong style={{ marginLeft: 4 }}>{attemptedList.length}</strong>
          </div>
          <div className="badge badge-outline" style={{ padding: '6px 12px', fontSize: '0.75rem', borderColor: 'var(--border-subtle)' }}>
            Correct: <strong style={{ marginLeft: 4 }}>{correctCount}</strong>
          </div>
          <div className="badge badge-outline" style={{ padding: '6px 12px', fontSize: '0.75rem' }}>
            Incorrect: <strong style={{ marginLeft: 4 }}>{incorrectCount}</strong>
          </div>
          <div className="badge badge-outline" style={{ padding: '6px 12px', fontSize: '0.75rem', fontWeight: 700 }}>
            <Award size={12} style={{ marginRight: 4 }} /> {accuracyPct}% Accuracy
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
        <div className="tab-group">
          <button
            onClick={() => setFilterStatus('all')}
            className={`tab-btn ${filterStatus === 'all' ? 'active' : ''}`}
            style={{ fontSize: '0.78rem', padding: '6px 14px' }}
          >
            All Attempted
          </button>
          <button
            onClick={() => setFilterStatus('incorrect')}
            className={`tab-btn ${filterStatus === 'incorrect' ? 'active' : ''}`}
            style={{ fontSize: '0.78rem', padding: '6px 14px', color: filterStatus === 'incorrect' ? 'var(--text-main)' : 'var(--text-muted)' }}
          >
            Needs Revision (Incorrect)
          </button>
          <button
            onClick={() => setFilterStatus('correct')}
            className={`tab-btn ${filterStatus === 'correct' ? 'active' : ''}`}
            style={{ fontSize: '0.78rem', padding: '6px 14px' }}
          >
            Mastered (Correct)
          </button>
        </div>

        <button
          onClick={loadAttempted}
          className="btn btn-outline"
          style={{ fontSize: '0.74rem', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
        >
          <RefreshCw size={12} className={loading ? 'spin' : ''} /> Refresh
        </button>
      </div>

      {/* Questions List */}
      {loading ? (
        <div className="card" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <RefreshCw size={24} className="spin" style={{ margin: '0 auto 12px auto' }} />
          <div>Loading your attempted questions...</div>
        </div>
      ) : attemptedList.length === 0 ? (
        <div className="card" style={{ padding: '3rem', textAlign: 'center' }}>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', margin: '0 0 12px 0' }}>
            {filterStatus === 'incorrect'
              ? 'No incorrect questions to revise! Great job!'
              : 'No attempted questions found in this view.'}
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Go to the <strong>All PYQs</strong> tab, answer some questions, and they will automatically show up here for re-testing.
          </p>
        </div>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {attemptedList.map((item, idx) => {
            const qId = item.question_id;
            const currentSelected = selectedAnswers[qId] || item.last_chosen_option;
            const isSubmitting = submittingId === qId;

            return (
              <div
                key={qId || idx}
                className="card"
                style={{
                  padding: '1.25rem 1.5rem',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '0.85rem',
                  borderLeft: item.is_correct ? '3px solid var(--text-main)' : '3px solid var(--border-subtle)'
                }}
              >
                {/* Header row: Topic, Exam Date, Repeated badge, Status */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', flexWrap: 'wrap' }}>
                    <span className="badge badge-outline" style={{ fontSize: '0.7rem', fontWeight: 600 }}>
                      {item.topic || 'General Awareness'}
                    </span>

                    {item.exam_date && (
                      <span className="badge badge-outline" style={{ fontSize: '0.68rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Calendar size={10} /> {item.exam_date}
                      </span>
                    )}

                    {item.repetition_count > 1 && (
                      <span className="badge badge-outline" style={{ fontSize: '0.68rem', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <Repeat size={10} /> Repeated {item.repetition_count}x
                      </span>
                    )}

                    <span className="badge badge-outline" style={{ fontSize: '0.65rem' }}>
                      Attempts: {item.attempt_count || 1}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                    {item.is_correct ? (
                      <span style={{ fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-main)', fontWeight: 600 }}>
                        <CheckCircle2 size={13} /> Previously Correct
                      </span>
                    ) : (
                      <span style={{ fontSize: '0.72rem', display: 'flex', alignItems: 'center', gap: '4px', color: 'var(--text-muted)', fontWeight: 600 }}>
                        <XCircle size={13} /> Needs Practice
                      </span>
                    )}

                    <button
                      onClick={() => handleClear(qId)}
                      title="Remove from revision list"
                      style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: '2px' }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                </div>

                {/* Question sentence */}
                <div style={{ fontSize: '0.92rem', fontWeight: 600, lineHeight: 1.45, color: 'var(--text-main)' }}>
                  {item.question_text}
                </div>

                {/* 4 Options Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '8px' }}>
                  {item.options &&
                    Object.entries(item.options).map(([optKey, optText]) => {
                      const isSelected = currentSelected === optKey;
                      const isCorrect = item.has_reevaluated && item.correct_option === optKey;
                      const isWrong = item.has_reevaluated && isSelected && !item.is_correct;

                      let borderColor = 'var(--border-subtle)';
                      let bg = 'var(--bg-main)';

                      if (isSelected) {
                        borderColor = 'var(--text-main)';
                        bg = 'var(--bg-card)';
                      }
                      if (isCorrect) {
                        borderColor = 'var(--text-main)';
                      }

                      return (
                        <div
                          key={optKey}
                          onClick={() => handleSelectOption(qId, optKey)}
                          style={{
                            padding: '10px 14px',
                            borderRadius: '6px',
                            border: `1px solid ${borderColor}`,
                            background: bg,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'flex-start',
                            gap: '10px',
                            fontSize: '0.82rem',
                            transition: 'all 0.15s ease'
                          }}
                        >
                          <span
                            style={{
                              fontWeight: 700,
                              minWidth: '22px',
                              height: '22px',
                              borderRadius: '4px',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'center',
                              background: isSelected ? 'var(--text-main)' : 'var(--bg-card)',
                              color: isSelected ? 'var(--bg-main)' : 'var(--text-main)',
                              border: '1px solid var(--border-subtle)',
                              fontSize: '0.72rem'
                            }}
                          >
                            {optKey}
                          </span>
                          <span style={{ flex: 1, lineHeight: 1.35, color: 'var(--text-main)' }}>
                            {optText}
                          </span>
                          {isCorrect && (
                            <CheckCircle2 size={14} style={{ color: 'var(--text-main)', marginTop: 3 }} />
                          )}
                          {isWrong && (
                            <XCircle size={14} style={{ color: 'var(--text-muted)', marginTop: 3 }} />
                          )}
                        </div>
                      );
                    })}
                </div>

                {/* Reattempt Action Bar */}
                <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '10px', marginTop: '4px' }}>
                  {item.has_reevaluated && (
                    <span style={{ fontSize: '0.75rem', color: item.is_correct ? 'var(--text-main)' : 'var(--text-muted)', fontWeight: 600 }}>
                      {item.is_correct ? '✓ Reattempt Correct!' : '✗ Still Incorrect, review options again.'}
                    </span>
                  )}
                  <button
                    onClick={() => handleReattempt(item)}
                    disabled={isSubmitting || !selectedAnswers[qId]}
                    className="btn btn-primary"
                    style={{
                      fontSize: '0.76rem',
                      padding: '7px 16px',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '6px',
                      opacity: (!selectedAnswers[qId] || isSubmitting) ? 0.6 : 1
                    }}
                  >
                    <RotateCcw size={12} className={isSubmitting ? 'spin' : ''} />
                    {isSubmitting ? 'Checking...' : 'Submit Reattempt'}
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

