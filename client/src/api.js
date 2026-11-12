/**
 * API Client for Antigravity Intelligent Study Assistant
 * Manages JWT tokens in localStorage and attaches Authorization headers.
 */

// If VITE_API_URL is set (e.g. on Vercel deployment), point to it; otherwise fallback to relative '/api' for local Vite dev proxy.
const RAW_API_BASE = import.meta.env.VITE_API_URL || '/api';
const API_BASE = RAW_API_BASE.replace(/\/+$/, '');
const TOKEN_KEY = 'antigravity_token';

export const getStoredToken = () => localStorage.getItem(TOKEN_KEY);
export const setStoredToken = (token) => localStorage.setItem(TOKEN_KEY, token);
export const removeStoredToken = () => localStorage.removeItem(TOKEN_KEY);

const request = async (endpoint, options = {}) => {
  const token = getStoredToken();
  const headers = {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...(options.headers || {})
  };

  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  const response = await fetch(`${API_BASE}${cleanEndpoint}`, {
    ...options,
    headers
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || 'An error occurred during request');
  }
  return data;
};

export const api = {
  // Authentication
  login: async (username, password) => {
    const res = await request('/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username: username.trim(), password })
    });
    if (res.token) setStoredToken(res.token);
    return res;
  },

  register: async (username, password, confirmPassword) => {
    const res = await request('/auth/register', {
      method: 'POST',
      body: JSON.stringify({
        username: username.trim(),
        password,
        confirm_password: confirmPassword
      })
    });
    if (res.token) setStoredToken(res.token);
    return res;
  },

  getMe: async () => {
    return request('/auth/me');
  },

  logout: () => {
    removeStoredToken();
  },

  // Syllabus Sections
  getSections: async () => {
    return request('/sections');
  },

  // PYQ Retrieval (ChromaDB + MongoDB + JSON Direct)
  searchPYQs: async (query, topicFilter = null, shiftFilter = null, nResults = 500) => {
    const params = new URLSearchParams({ query, n_results: nResults });
    if (topicFilter) params.append('topic_filter', topicFilter);
    if (shiftFilter) params.append('shift_filter', shiftFilter);
    return request(`/pyqs/search?${params.toString()}`);
  },

  // Lucent GK Page & Dynamic Question Generator
  getLucentPage: async (pageNumber) => {
    return request(`/lucent/page/${pageNumber}`);
  },

  generateLucentQuestions: async (pageNumber, count = 3) => {
    return request('/lucent/generate-questions', {
      method: 'POST',
      body: JSON.stringify({ page_number: pageNumber, count })
    });
  },

  // Performance Logging & Revision (MongoDB)
  recordAttempt: async (attemptData) => {
    return request('/user/record-attempt', {
      method: 'POST',
      body: JSON.stringify(attemptData)
    });
  },

  getAttemptedQuestions: async (filterStatus = 'all') => {
    return request(`/user/attempted-questions?filter_status=${filterStatus}`);
  },

  clearAttemptedQuestion: async (questionId) => {
    return request(`/user/attempted-questions/${questionId}`, {
      method: 'DELETE'
    });
  }
};

