import React, { useState } from 'react';

const ChatSection = ({ isAuthenticated, onBackClick }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSend = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    const userMessage = { role: 'user', content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    try {
      const token = localStorage.getItem('token');
      const response = await fetch('http://localhost:8000/chat/consult', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ message: userMessage.content }),
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Ошибка запроса к AI');
      }
      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.answer,
          reliability: data.reliability,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="chat-section">
      <button className="back-btn" onClick={onBackClick}>
        &larr; Назад
      </button>
      <h2>Юридический AI-чат</h2>
      <div className="chat-history">
        {messages.map((msg, idx) => (
          <div key={idx} className={`chat-msg ${msg.role}`}>
            <div className="msg-content">{msg.content}</div>
            {msg.role === 'assistant' && (
              <div className="msg-meta">
                <span>Надежность: {(msg.reliability * 100).toFixed(1)}%</span>
                {msg.sources && msg.sources.length > 0 && (
                  <details>
                    <summary>Источники</summary>
                    <ul>
                      {msg.sources.map((src, i) => (
                        <li key={i}>{src}</li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            )}
          </div>
        ))}
        {loading && <div className="chat-msg assistant">AI печатает...</div>}
      </div>
      <div className="chat-input-row">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleInputKeyDown}
          placeholder="Введите ваш юридический вопрос..."
          rows={2}
          disabled={loading}
        />
        <button onClick={handleSend} disabled={loading || !input.trim()}>
          Отправить
        </button>
      </div>
      {error && <div className="chat-error">{error}</div>}
    </div>
  );
};

export default ChatSection;
