import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import SectionsBanner from './components/SectionsBanner';
import PYQSearch from './components/PYQSearch';
import LucentQuestionGen from './components/LucentQuestionGen';
import AuthModal from './components/AuthModal';
import UserProfileModal from './components/UserProfileModal';
import { api, getStoredToken } from './api';
import { Layers, BookOpen } from 'lucide-react';

export default function App() {
  const [user, setUser] = useState(null);
  const [sectionsData, setSectionsData] = useState(null);
  const [activeTab, setActiveTab] = useState('pyqs');
  const [isAuthOpen, setIsAuthOpen] = useState(false);
  const [isProfileOpen, setIsProfileOpen] = useState(false);

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
      target_exam: authData.target_exam || 'SSC CHSL 2024 Tier I',
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
              <span>ChromaDB Semantic PYQs</span>
              <span className="badge badge-outline" style={{ fontSize: '0.62rem' }}>12,968 Qs</span>
            </button>
            <button
              onClick={() => setActiveTab('lucent')}
              className={`tab-btn ${activeTab === 'lucent' ? 'active' : ''}`}
            >
              <BookOpen size={14} />
              <span>Lucent GK Question Generator</span>
              <span className="badge badge-outline" style={{ fontSize: '0.62rem' }}>672 Pages</span>
            </button>
          </div>

          {!user && (
            <div style={{ fontSize: '0.75rem', color: '#888888', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <span>• Tip:</span>
              <button
                onClick={() => setIsAuthOpen(true)}
                style={{ background: 'none', border: 'none', color: '#ffffff', textDecoration: 'underline', cursor: 'pointer', fontWeight: 600 }}
              >
                Sign In
              </button>
              <span>to store your test attempts in MongoDB Atlas.</span>
            </div>
          )}
        </div>

        {/* Active Tool View */}
        {activeTab === 'pyqs' ? (
          <PYQSearch user={user} onAttemptRecorded={refreshUserProfile} />
        ) : (
          <LucentQuestionGen user={user} onAttemptRecorded={refreshUserProfile} />
        )}
      </main>

      <footer style={{ borderTop: '1px solid #1a1a1a', background: '#000000', padding: '1.25rem 0', fontSize: '0.72rem', color: '#555555' }}>
        <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
          <div>Antigravity Study Assistant • Connected to MongoDB Atlas & ChromaDB</div>
          <div>Focus: Lucent GK Book (English 2026) + Verified CHSL PYQs</div>
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
