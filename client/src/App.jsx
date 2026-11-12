import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import SectionsBanner from './components/SectionsBanner';
import PYQSearch from './components/PYQSearch';
import LucentQuestionGen from './components/LucentQuestionGen';
import ReviseProgress from './components/ReviseProgress';
import AuthModal from './components/AuthModal';
import UserProfileModal from './components/UserProfileModal';
import { api, getStoredToken } from './api';
import { Layers, BookOpen, RotateCcw } from 'lucide-react';

export default function App() {
  const [user, setUser] = useState(null);
  const [sectionsData, setSectionsData] = useState(null);
  const [activeTab, setActiveTab] = useState('pyqs');
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('study_theme') || 'dark';
  });

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('study_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  useEffect(() => {
    fetchInitialData();
  }, []);

  const fetchInitialData = async () => {
    try {
      const sec = await api.getSections();
      setSectionsData(sec);
    } catch (err) {
      console.error('Failed to load sections data:', err);
    }

    if (getStoredToken()) {
      try {
        const profile = await api.getMe();
        setUser(profile);
      } catch (err) {
        console.warn('Session expired or invalid:', err.message);
        api.logout();
        setUser(null);
      }
    }
  };

  const handleAuthSuccess = (authData) => {
    setUser({
      user_id: authData.user_id,
      username: authData.username,
      client_ip: authData.client_ip,
      email: authData.email,
      full_name: authData.full_name || authData.username,
      target_exam: authData.target_exam || 'SSC CHSL 2026 Dec',
      analytics: {
        total_questions_attempted: 0,
        total_correct: 0,
        overall_accuracy_pct: 0
      }
    });
    fetchInitialData();
  };

  const handleLogout = () => {
    api.logout();
    setUser(null);
  };

  const refreshUserProfile = async () => {
    if (getStoredToken()) {
      try {
        const profile = await api.getMe();
        setUser(profile);
      } catch (err) {
        console.error('Failed to refresh user stats:', err);
      }
    }
  };

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Navbar
        user={user}
        onOpenAuth={() => setIsAuthOpen(true)}
        onOpenProfile={() => setIsProfileOpen(true)}
        onLogout={handleLogout}
        theme={theme}
        onToggleTheme={toggleTheme}
      />

      <main className="container" style={{ flex: 1, padding: '2rem 1.5rem' }}>
        {/* Sections Banner with active focus on General Awareness */}
        <SectionsBanner sectionsData={sectionsData} />

        {/* Tab Switcher */}
        <div className="tab-bar">
          <div className="tab-group">
            <button
              onClick={() => setActiveTab('pyqs')}
              className={`tab-btn ${activeTab === 'pyqs' ? 'active' : ''}`}
            >
              <Layers size={14} />
              <span>All PYQs</span>
              <span className="badge badge-outline" style={{ fontSize: '0.62rem' }}>12,968 Qs</span>
            </button>
            <button
              onClick={() => setActiveTab('lucent')}
              className={`tab-btn ${activeTab === 'lucent' ? 'active' : ''}`}
            >
              <BookOpen size={14} />
              <span>NEW Question Generator from Lucent</span>
              <span className="badge badge-outline" style={{ fontSize: '0.62rem' }}>672 Pages</span>
            </button>
            <button
              onClick={() => setActiveTab('revise')}
              className={`tab-btn ${activeTab === 'revise' ? 'active' : ''}`}
            >
              <RotateCcw size={14} />
              <span>Revise Your Progress</span>
            </button>
          </div>

          {!user && (
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>• Tip:</span>
              <button
                onClick={() => setIsAuthOpen(true)}
                style={{ background: 'none', border: 'none', color: 'var(--text-main)', textDecoration: 'underline', cursor: 'pointer', fontWeight: 600 }}
              >
                Sign In
              </button>
              <span>to save your test attempts and Scores in your Account.</span>
            </div>
          )}
        </div>

        {/* Active Tool View */}
        {activeTab === 'pyqs' && (
          <PYQSearch user={user} onAttemptRecorded={refreshUserProfile} />
        )}
        {activeTab === 'lucent' && (
          <LucentQuestionGen user={user} onAttemptRecorded={refreshUserProfile} />
        )}
        {activeTab === 'revise' && (
          <ReviseProgress user={user} onAttemptRecorded={refreshUserProfile} />
        )}
      </main>

      <footer style={{ borderTop: '1px solid var(--border-subtle)', background: 'var(--bg-main)', padding: '1.25rem 0', fontSize: '0.72rem', color: 'var(--text-muted)', transition: 'background-color 0.2s ease, border-color 0.2s ease' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div>Study Assistant </div>

          <div>Focus: General Awareness + English + Verified CHSL PYQs</div>
        </div>
      </footer>

      {/* Modals */}
      <AuthModal
        isOpen={isAuthOpen}
        onClose={() => setIsAuthOpen(false)}
        onAuthSuccess={handleAuthSuccess}
      />

      <UserProfileModal
        isOpen={isProfileOpen}
        onClose={() => setIsProfileOpen(false)}
        user={user}
        onLogout={handleLogout}
      />
    </div>
  );
}
