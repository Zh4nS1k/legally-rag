import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useNavigate,
} from 'react-router-dom';
import Header from './components/Header';
import UploadSection from './components/UploadSection';
import LoadingSection from './components/LoadingSection';
import ResultSection from './components/ResultSection';
import HistorySection from './components/HistorySection';
import Footer from './components/Footer';
import AuthPage from './components/AuthPage';
import ProtectedRoute from './components/ProtectedRoute';
import ProfileSection from './components/ProfileSection';
import './styles/index.css';

function App() {
  const navigate = useNavigate();
  const abortControllerRef = useRef(null);

  const [appState, setAppState] = useState({
    isLoading: false,
    fileInfo: null,
    analysisData: null,
    error: null,
    isAuthenticated: false,
    userData: null,
  });

  const handleLogout = useCallback(() => {
    localStorage.removeItem('token');
    setAppState({
      isLoading: false,
      fileInfo: null,
      analysisData: null,
      error: null,
      isAuthenticated: false,
      userData: null,
    });
    navigate('/login');
  }, [navigate]);

  const validateToken = useCallback(
    async (token) => {
      try {
        const response = await fetch(
          'http://localhost:8080/api/validate-token',
          {
            headers: { Authorization: `Bearer ${token}` },
          }
        );
        if (response.ok) {
          setAppState((prev) => ({ ...prev, isAuthenticated: true }));
        } else {
          handleLogout();
        }
      } catch {
        handleLogout();
      }
    },
    [handleLogout]
  );

  const fetchUserData = useCallback(async (token) => {
    try {
      const response = await fetch('http://localhost:8080/api/user', {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        const data = await response.json();
        setAppState((prev) => ({ ...prev, userData: data }));
      }
    } catch (err) {
      console.error('Error fetching user data:', err);
    }
  }, []);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      validateToken(token);
      fetchUserData(token);
    }
  }, [validateToken, fetchUserData]);

  const handleFileUpload = async (file) => {
    if (!file) return;

    const abortController = new AbortController();
    abortControllerRef.current = abortController;

    setAppState((prev) => ({
      ...prev,
      isLoading: true,
      error: null,
      fileInfo: { name: file.name, size: file.size },
    }));

    try {
      const token = localStorage.getItem('token');
      if (!token) throw new Error('Требуется авторизация');

      const formData = new FormData();
      formData.append('document', file);

      const response = await fetch('http://localhost:8080/api/analyze', {
        method: 'POST',
        body: formData,
        headers: {
          Authorization: `Bearer ${token}`,
        },
        signal: abortController.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Ошибка при анализе документа');
      }

      const data = await response.json();
      setAppState((prev) => ({
        ...prev,
        isLoading: false,
        analysisData: data,
      }));
    } catch (err) {
      if (err.name === 'AbortError') {
        console.warn('Анализ отменен пользователем');
        setAppState((prev) => ({
          ...prev,
          isLoading: false,
          fileInfo: null,
          analysisData: null,
          error: 'Анализ отменен пользователем',
        }));
      } else {
        setAppState((prev) => ({
          ...prev,
          isLoading: false,
          error: err.message,
        }));
        if (
          err.message.includes('Сессия истекла') ||
          err.message.includes('Требуется авторизация')
        ) {
          handleLogout();
        }
      }
    }
  };

  const handleCancelUpload = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
  };

  const handleRemoveFile = async () => {
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8080/api/cache/clear', {
        method: 'POST',
        headers: {
          Authorization: `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error('Не удалось удалить файл');
      }

      setAppState((prev) => ({
        ...prev,
        fileInfo: null,
        error: null,
      }));
    } catch (err) {
      setAppState((prev) => ({
        ...prev,
        error: err.message,
      }));
    }
  };

  return (
    <div className="App">
      <Header
        isAuthenticated={appState.isAuthenticated}
        onLogout={handleLogout}
        userData={appState.userData}
      />
      <main className="container">
        <Routes>
          <Route
            path="/login"
            element={
              <AuthPage
                onSuccess={() => {
                  const token = localStorage.getItem('token');
                  setAppState((prev) => ({ ...prev, isAuthenticated: true }));
                  fetchUserData(token);
                  navigate('/');
                }}
                type="login"
              />
            }
          />
          <Route
            path="/register"
            element={
              <AuthPage onSuccess={() => navigate('/login')} type="register" />
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                <ProfileSection
                  userData={appState.userData}
                  onLogout={handleLogout}
                  onBack={() => navigate('/')}
                />
              </ProtectedRoute>
            }
          />
          <Route
            path="/"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                {appState.isLoading ? (
                  <LoadingSection onCancel={handleCancelUpload} />
                ) : appState.analysisData ? (
                  <ResultSection
                    data={appState.analysisData}
                    onBackClick={() =>
                      setAppState((prev) => ({ ...prev, analysisData: null }))
                    }
                  />
                ) : (
                  <UploadSection
                    onFileUpload={handleFileUpload}
                    fileInfo={appState.fileInfo}
                    error={appState.error}
                    onClearError={() =>
                      setAppState((prev) => ({ ...prev, error: null }))
                    }
                    onHistoryClick={() => navigate('/history')}
                    onCancelUpload={handleCancelUpload}
                    onRemoveFile={handleRemoveFile}
                  />
                )}
              </ProtectedRoute>
            }
          />
          <Route
            path="/history"
            element={
              <ProtectedRoute isAuthenticated={appState.isAuthenticated}>
                <HistorySection onBackClick={() => navigate('/')} />
              </ProtectedRoute>
            }
          />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}

export default AppWrapper;
