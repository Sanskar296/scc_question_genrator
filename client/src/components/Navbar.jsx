import React from 'react';
import { BookOpen, LogIn, LogOut, User, Sun, Moon } from 'lucide-react';

export default function Navbar({ user, onOpenAuth, onOpenProfile, onLogout, theme, onToggleTheme }) {
  return (
    <header className="header">
      <div className="container header-inner">
        {/* Brand */}
        <div className="brand">
          <div className="brand-icon">
            <BookOpen size={18} />
          </div>
          <div>
            <div className="brand-title">
              <span>Daily Pratice</span>
              <span className="badge badge-outline">SSC CHSL</span>
            </div>
            <div className="brand-subtitle">Intelligent Study & Question Retrieval</div>
          </div>
        </div>

        {/* Center Pill: Focus Indicator */}
        <div className="focus-pill">
          <span className="dot-indicator"></span>
          <span className="text-muted">Active Focus:</span>
          <strong>General Awareness (Lucent GK + PYQs)</strong>
        </div>

        {/* User Account & Theme Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          {/* Black & White Theme Toggle */}
          <button
            onClick={onToggleTheme}
            className="theme-toggle-btn"
            title={`Switch to ${theme === 'dark' ? 'Clean White' : 'Dark Black'} UI`}
            aria-label="Toggle Black/White Theme"
          >
            {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
          </button>

          {user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <button onClick={onOpenProfile} className="btn btn-secondary btn-sm">
                <User size={14} />
                <span>{user.username}</span>
                {user.analytics && (
                  <span className="badge badge-active" style={{ fontSize: '0.65rem', padding: '2px 5px' }}>
                    {user.analytics.overall_accuracy_pct || 0}% Acc
                  </span>
                )}
              </button>
              <button
                onClick={onLogout}
                title="Sign Out"
                className="btn btn-outline btn-sm"
              >
                <LogOut size={14} />
              </button>
            </div>
          ) : (
            <button onClick={onOpenAuth} className="btn btn-primary btn-sm">
              <LogIn size={14} />
              <span>Sign In / Register</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
