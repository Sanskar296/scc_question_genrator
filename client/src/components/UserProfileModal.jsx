import React from 'react';
import { X, LogOut, Database, User, Mail, Target } from 'lucide-react';

export default function UserProfileModal({ isOpen, onClose, user, onLogout }) {
  if (!isOpen || !user) return null;

  const analytics = user.analytics || {};
  const breakdown = analytics.subject_breakdown || {};

  return (
    <div className="modal-overlay">
      <div className="modal-dialog">
        <button onClick={onClose} className="modal-close">
          <X size={18} />
        </button>

        {/* User Card Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', borderBottom: '1px solid #222222', paddingBottom: '14px', marginBottom: '16px' }}>
          <div style={{
            width: '44px',
            height: '44px',
            background: '#ffffff',
            color: '#000000',
            borderRadius: '10px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontWeight: 800,
            fontSize: '1.2rem'
          }}>
            {user.username ? user.username[0].toUpperCase() : 'U'}
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h3 style={{ fontSize: '1.05rem', fontWeight: 700 }}>{user.full_name || user.username}</h3>
              <span className="badge badge-outline" style={{ fontSize: '0.62rem' }}>MongoDB Synced</span>
            </div>
            <div style={{ fontSize: '0.74rem', color: '#888888', display: 'flex', gap: '10px', marginTop: '2px' }}>
              <span>@{user.username}</span>
              <span className="mono">IP: {user.client_ip || '127.0.0.1'}</span>
            </div>
          </div>
        </div>

        {/* Security & Target info */}
        <div style={{ background: '#000000', border: '1px solid #222222', padding: '10px 14px', borderRadius: '8px', marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <div style={{ fontSize: '0.68rem', color: '#888888', textTransform: 'uppercase' }}>Security & Duration</div>
            <div style={{ fontSize: '0.82rem', fontWeight: 600 }}>IP-Bound JWT • 7-Day Session Active</div>
          </div>
          <span className="badge badge-active">Protected</span>
        </div>

        {/* Analytics Numbers */}
        <div className="stats-grid">
          <div className="stat-item">
            <div className="stat-number">{analytics.total_questions_attempted || 0}</div>
            <div className="stat-label">Attempted</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">{analytics.total_correct || 0}</div>
            <div className="stat-label">Correct</div>
          </div>
          <div className="stat-item">
            <div className="stat-number">{analytics.overall_accuracy_pct || 0}%</div>
            <div className="stat-label">Accuracy</div>
          </div>
        </div>

        {/* Subject Breakdown */}
        {Object.keys(breakdown).length > 0 && (
          <div style={{ margin: '14px 0', borderTop: '1px solid #222222', paddingTop: '12px' }}>
            <div style={{ fontSize: '0.74rem', color: '#888888', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
              Subject Performance
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {Object.entries(breakdown).map(([subject, stats]) => (
                <div key={subject} style={{ display: 'flex', justifyContent: 'space-between', background: '#000000', border: '1px solid #1a1a1a', padding: '8px 12px', borderRadius: '6px', fontSize: '0.78rem' }}>
                  <span>{subject}</span>
                  <div style={{ display: 'flex', gap: '10px' }}>
                    <span className="mono text-muted">{stats.correct || 0}/{stats.attempted || 0} Qs</span>
                    <strong className="mono">{stats.accuracy_pct || 0}%</strong>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Sign Out Button */}
        <div style={{ marginTop: '16px', borderTop: '1px solid #222222', paddingTop: '14px' }}>
          <button
            onClick={() => { onLogout(); onClose(); }}
            className="btn btn-secondary"
            style={{ width: '100%', padding: '10px' }}
          >
            <LogOut size={14} />
            <span>Sign Out</span>
          </button>
        </div>
      </div>
    </div>
  );
}
