import React, { useState } from 'react';
import { X, LogIn, UserPlus, ShieldCheck } from 'lucide-react';
import { api } from '../api';

export default function AuthModal({ isOpen, onClose, onAuthSuccess }) {
  const [tab, setTab] = useState('login');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Exactly 2 inputs for login, plus 1 extra (confirm password) for register
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    if (tab === 'register') {
      if (password !== confirmPassword) {
        setError('Passwords do not match. Please enter matching passwords.');
        return;
      }
      if (password.length < 4) {
        setError('Password must be at least 4 characters.');
        return;
      }
    }

    setLoading(true);
    try {
      if (tab === 'login') {
        const res = await api.login(username, password);
        onAuthSuccess(res);
        onClose();
      } else {
        const res = await api.register(username, password, confirmPassword);
        onAuthSuccess(res);
        onClose();
      }
    } catch (err) {
      setError(err.message || 'Authentication failed. Please verify your credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="modal-overlay">
      <div className="modal-dialog">
        <button onClick={onClose} className="modal-close" aria-label="Close modal">
          <X size={18} />
        </button>

        {/* Tab Toggle */}
        <div style={{
          display: 'flex',
          background: '#000000',
          border: '1px solid var(--border-subtle)',
          borderRadius: '8px',
          padding: '3px',
          marginBottom: '1.25rem'
        }}>
          <button
            type="button"
            onClick={() => { setTab('login'); setError(''); }}
            className={`btn btn-sm ${tab === 'login' ? 'btn-primary' : 'btn-outline'}`}
            style={{ flex: 1, border: 'none' }}
          >
            <LogIn size={13} />
            <span>Sign In</span>
          </button>
          <button
            type="button"
            onClick={() => { setTab('register'); setError(''); }}
            className={`btn btn-sm ${tab === 'register' ? 'btn-primary' : 'btn-outline'}`}
            style={{ flex: 1, border: 'none' }}
          >
            <UserPlus size={13} />
            <span>Register</span>
          </button>
        </div>

        {/* Title & Directive */}
        <div style={{ marginBottom: '1.25rem' }}>
          <h3 style={{ fontSize: '1.1rem', fontWeight: 800 }}>
            {tab === 'login' ? 'Sign In to Assistant' : 'Create Candidate Account'}
          </h3>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginTop: '4px' }}>
            <ShieldCheck size={13} style={{ color: '#ffffff' }} />
            <span style={{ fontSize: '0.72rem', color: '#a3a3a3' }}>
              IP-restricted JWT • Active session persisted for 7 days
            </span>
          </div>
        </div>

        {/* Error Alert */}
        {error && (
          <div style={{
            background: '#1a1111',
            border: '1px solid #7f1d1d',
            color: '#fca5a5',
            padding: '8px 12px',
            borderRadius: '6px',
            fontSize: '0.78rem',
            marginBottom: '14px'
          }}>
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit}>
          {/* Input 1: Username */}
          <div className="form-group">
            <label className="form-label">Username</label>
            <input
              type="text"
              required
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="Enter your username"
              className="input"
              autoComplete="username"
            />
          </div>

          {/* Input 2: Password */}
          <div className="form-group">
            <label className="form-label">Password</label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Enter your password"
              className="input"
              autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
            />
          </div>

          {/* One Extra Input for Register: Confirm Password */}
          {tab === 'register' && (
            <div className="form-group">
              <label className="form-label">Confirm Password</label>
              <input
                type="password"
                required
                value={confirmPassword}
                onChange={(e) => setConfirmPassword(e.target.value)}
                placeholder="Confirm your password"
                className="input"
                autoComplete="new-password"
              />
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className="btn btn-primary"
            style={{ width: '100%', marginTop: '14px', padding: '10px' }}
          >
            {loading ? 'Authenticating with MongoDB...' : (tab === 'login' ? 'Sign In' : 'Register Account')}
          </button>
        </form>
      </div>
    </div>
  );
}
