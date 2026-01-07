import { useState, useEffect } from "react";
import { Upload, FileText, Trash2, Database, Search, AlertCircle, CheckCircle, Loader } from 'lucide-react';

export default function RAGDemo() {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState('');
  const [indexes, setIndexes] = useState([]);
  const [uploadedDocs, setUploadedDocs] = useState([]);
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState('');
  const [statusType, setStatusType] = useState('info');

  const API_BASE = "https://rag-q1tl.onrender.com";

`${API_BASE}/ask_question`


  const fetchIndexes = async () => {
    try {
      const res = await fetch(`${API_BASE}/list_indexes`);
      const data = await res.json();
      setIndexes(data.indexes || []);
    } catch (err) {
      console.error('Failed to fetch indexes:', err);
    }
  };

  const fetchUploadedDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/list_uploaded_docs`);
      const data = await res.json();
      setUploadedDocs(data.files || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  useEffect(() => {
    fetchIndexes();
    fetchUploadedDocs();
  }, []);

  const askQuestion = async () => {
    if (!query.trim() || !selectedIndex) return;
    
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/ask_question`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, index_name: selectedIndex })
      });
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setStatus('Failed to get answer');
      setStatusType('error');
    }
    setLoading(false);
  };

  const handleUpload = async () => {
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    setStatus('Uploading...');
    setStatusType('info');
    
    try {
      await fetch(`${API_BASE}/upload_document`, {
        method: 'POST',
        body: formData
      });
      setStatus('Document uploaded successfully!');
      setStatusType('success');
      setFile(null);
      fetchUploadedDocs();
    } catch (err) {
      setStatus('Upload failed');
      setStatusType('error');
    }
  };

  const handleBuildIndex = async () => {
    setStatus('Building index...');
    setStatusType('info');
    
    try {
      const res = await fetch(`${API_BASE}/build_index`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ index_name: "default" }),
      });

      const data = await res.json();
    //   setStatus(data.message);
      setStatus('Index built successfully!');
      setStatusType('success');
      fetchIndexes();
    } catch (err) {
      setStatus('Failed to build index');
      setStatusType('error');
    }
  };

  const handleDeleteIndex = async (indexName) => {
    try {
      await fetch(`${API_BASE}/delete_index/${indexName}`, {
        method: 'DELETE'
      });
      if (selectedIndex === indexName) setSelectedIndex('');
      fetchIndexes();
    } catch (err) {
      setStatus('Failed to delete index');
      setStatusType('error');
    }
  };

  const handleDeleteDoc = async (docName) => {
    try {
      await fetch(`${API_BASE}/delete_uploaded_doc/${docName}`, {
        method: 'DELETE'
      });
      fetchUploadedDocs();
    } catch (err) {
      setStatus('Failed to delete document');
      setStatusType('error');
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      minWidth: '100wh',
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      padding: '2rem',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    }}>
      <div style={{
        margin: '0 auto',
        background: 'rgba(255, 255, 255, 0.95)',
        borderRadius: '20px',
        boxShadow: '0 20px 60px rgba(0, 0, 0, 0.3)',
        overflow: 'hidden'
      }}>
        {/* Header */}
        <div style={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          padding: '2rem',
          color: 'white'
        }}>
          <h1 style={{
            margin: 0,
            fontSize: '2.5rem',
            fontWeight: '700',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem'
          }}>
            <Database size={40} />
            Local RAG Demo
          </h1>
          <p style={{ margin: '0.5rem 0 0 0', opacity: 0.9, fontSize: '1.1rem' }}>
            Upload documents, build indexes, and ask questions
          </p>
        </div>

        <div style={{ padding: '2rem' }}>
          {/* Status Message */}
          {status && (
            <div style={{
              padding: '1rem',
              borderRadius: '10px',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem',
              background: statusType === 'success' ? '#d4edda' : statusType === 'error' ? '#f8d7da' : '#d1ecf1',
              color: statusType === 'success' ? '#155724' : statusType === 'error' ? '#721c24' : '#0c5460',
              border: `1px solid ${statusType === 'success' ? '#c3e6cb' : statusType === 'error' ? '#f5c6cb' : '#bee5eb'}`
            }}>
              {statusType === 'success' ? <CheckCircle size={20} /> : statusType === 'error' ? <AlertCircle size={20} /> : <Loader size={20} />}
              <span>{status}</span>
            </div>
          )}

          {/* Query Section */}
          <div style={{
            background: '#f8f9fa',
            padding: '1.5rem',
            borderRadius: '12px',
            marginBottom: '1.5rem',
            border: '2px solid #e9ecef'
          }}>
            <h2 style={{ margin: '0 0 1rem 0', fontSize: '1.5rem', color: '#2d3748', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Search size={24} />
              Ask a Question
            </h2>
            
            <select
              value={selectedIndex}
              onChange={(e) => setSelectedIndex(e.target.value)}
              style={{
                width: '100%',
                padding: '0.75rem 1rem',
                borderRadius: '8px',
                border: '2px solid #cbd5e0',
                background: 'white',
                fontSize: '1rem',
                cursor: 'pointer',
                marginBottom: '1rem',
                color: '#2d3748',
                transition: 'all 0.2s ease'
              }}
              onFocus={(e) => e.target.style.borderColor = '#667eea'}
              onBlur={(e) => e.target.style.borderColor = '#cbd5e0'}
            >
              <option value="" disabled>Select an index...</option>
              {indexes.map((idx) => (
                <option key={idx} value={idx}>{idx}</option>
              ))}
            </select>

            <textarea
              rows={4}
              style={{
                width: '100%',
                padding: '1rem',
                borderRadius: '8px',
                border: '2px solid #cbd5e0',
                fontSize: '1rem',
                fontFamily: 'inherit',
                resize: 'vertical',
                marginBottom: '1rem',
                transition: 'all 0.2s ease',
                boxSizing: 'border-box'
              }}
              placeholder="Type your question here..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={(e) => e.target.style.borderColor = '#667eea'}
              onBlur={(e) => e.target.style.borderColor = '#cbd5e0'}
            />
            
            <button
              onClick={askQuestion}
              disabled={loading || !selectedIndex || !query.trim()}
              style={{
                width: '100%',
                padding: '1rem',
                borderRadius: '8px',
                border: 'none',
                background: loading || !selectedIndex || !query.trim() ? '#cbd5e0' : 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                fontSize: '1.1rem',
                fontWeight: '600',
                cursor: loading || !selectedIndex || !query.trim() ? 'not-allowed' : 'pointer',
                transition: 'all 0.2s ease',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.5rem'
              }}
              onMouseOver={(e) => !loading && selectedIndex && query.trim() && (e.target.style.transform = 'translateY(-2px)')}
              onMouseOut={(e) => e.target.style.transform = 'translateY(0)'}
            >
              {loading ? <Loader size={20} className="spin" /> : <Search size={20} />}
              {loading ? 'Thinking...' : 'Ask Question'}
            </button>
          </div>

          {/* Response Section */}
          {response && (
            <div style={{
              background: '#f8f9fa',
              padding: '1.5rem',
              borderRadius: '12px',
              marginBottom: '1.5rem',
              border: '2px solid #e9ecef'
            }}>
              <h2 style={{ margin: '0 0 1rem 0', fontSize: '1.5rem', color: '#2d3748' }}>Answer</h2>
              <div style={{
                background: 'white',
                padding: '1.5rem',
                borderRadius: '8px',
                border: '2px solid #667eea',
                fontSize: '1.1rem',
                lineHeight: '1.6',
                color: '#2d3748'
              }}>
                {response.answer}
              </div>

              {response.context && response.context.length > 0 && (
                <>
                  <h3 style={{ margin: '1.5rem 0 1rem 0', fontSize: '1.25rem', color: '#2d3748' }}>
                    Context Used ({response.context.length} chunks)
                  </h3>
                  {response.context.map((chunk, i) => (
                    <div
                      key={i}
                      style={{
                        marginBottom: '1rem',
                        padding: '1rem',
                        background: 'white',
                        borderRadius: '8px',
                        border: '1px solid #e2e8f0',
                        fontSize: '0.95rem',
                        lineHeight: '1.5',
                        color: '#4a5568'
                      }}
                    >
                      <div style={{ fontWeight: '600', marginBottom: '0.5rem', color: '#667eea' }}>
                        Chunk {i + 1}
                      </div>
                      <pre style={{
                        whiteSpace: 'pre-wrap',
                        wordBreak: 'break-word',
                        margin: 0,
                        fontFamily: 'inherit'
                      }}>
                        {chunk}
                      </pre>
                    </div>
                  ))}
                </>
              )}
            </div>
          )}

          {/* Upload Section */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: window.innerWidth > 768 ? '1fr 1fr' : '1fr',
            gap: '1.5rem',
            marginBottom: '1.5rem'
          }}>
            <div style={{
              background: '#f8f9fa',
              padding: '1.5rem',
              borderRadius: '12px',
              border: '2px solid #e9ecef'
            }}>
              <h2 style={{ margin: '0 0 1rem 0', fontSize: '1.5rem', color: '#2d3748', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={24} />
                Upload Document
              </h2>
              
              <label style={{
                display: 'block',
                padding: '2rem',
                border: '2px dashed #cbd5e0',
                borderRadius: '8px',
                textAlign: 'center',
                cursor: 'pointer',
                background: 'white',
                transition: 'all 0.2s ease',
                marginBottom: '1rem'
              }}
              onMouseOver={(e) => {
                e.currentTarget.style.borderColor = '#667eea';
                e.currentTarget.style.background = '#f7fafc';
              }}
              onMouseOut={(e) => {
                e.currentTarget.style.borderColor = '#cbd5e0';
                e.currentTarget.style.background = 'white';
              }}>
                <FileText size={32} style={{ margin: '0 auto 0.5rem', display: 'block', color: '#667eea' }} />
                <div style={{ fontSize: '1rem', color: '#4a5568' }}>
                  {file ? file.name : 'Click to upload PDF or TXT file'}
                </div>
                <input
                  type="file"
                  accept=".pdf,.txt"
                  onChange={(e) => setFile(e.target.files[0])}
                  style={{ display: 'none' }}
                />
              </label>

              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <button
                  onClick={handleUpload}
                  disabled={!file}
                  style={{
                    flex: 1,
                    padding: '0.75rem',
                    borderRadius: '8px',
                    border: 'none',
                    background: file ? '#667eea' : '#cbd5e0',
                    color: 'white',
                    fontSize: '1rem',
                    fontWeight: '600',
                    cursor: file ? 'pointer' : 'not-allowed',
                    transition: 'all 0.2s ease'
                  }}
                >
                  Upload
                </button>
                
                <button
                  onClick={handleBuildIndex}
                  disabled={uploadedDocs.length === 0}
                  style={{
                    flex: 1,
                    padding: '0.75rem',
                    borderRadius: '8px',
                    border: 'none',
                    background: uploadedDocs.length > 0 ? '#764ba2' : '#cbd5e0',
                    color: 'white',
                    fontSize: '1rem',
                    fontWeight: '600',
                    cursor: uploadedDocs.length > 0 ? 'pointer' : 'not-allowed',
                    transition: 'all 0.2s ease'
                  }}
                >
                  Build Index
                </button>
              </div>
            </div>

            <div style={{
              background: '#f8f9fa',
              padding: '1.5rem',
              borderRadius: '12px',
              border: '2px solid #e9ecef'
            }}>
              <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', color: '#2d3748' }}>
                Uploaded Documents ({uploadedDocs.length})
              </h3>
              <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
                {uploadedDocs.length === 0 ? (
                  <p style={{ color: '#718096', textAlign: 'center', padding: '1rem' }}>No documents uploaded</p>
                ) : (
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    {uploadedDocs.map(doc => (
                      <li key={doc} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        padding: '0.75rem',
                        background: 'white',
                        borderRadius: '6px',
                        marginBottom: '0.5rem',
                        border: '1px solid #e2e8f0'
                      }}>
                        <span style={{ color: '#2d3748', fontSize: '0.95rem', wordBreak: 'break-all' }}>{doc}</span>
                        <button
                          onClick={() => handleDeleteDoc(doc)}
                          style={{
                            padding: '0.5rem',
                            borderRadius: '6px',
                            border: 'none',
                            background: '#fed7d7',
                            color: '#c53030',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            transition: 'all 0.2s ease',
                            marginLeft: '0.5rem',
                            flexShrink: 0
                          }}
                          onMouseOver={(e) => e.target.style.background = '#fc8181'}
                          onMouseOut={(e) => e.target.style.background = '#fed7d7'}
                        >
                          <Trash2 size={16} />
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            </div>
          </div>

          {/* Indexes Section */}
          <div style={{
            background: '#f8f9fa',
            padding: '1.5rem',
            borderRadius: '12px',
            border: '2px solid #e9ecef'
          }}>
            <h3 style={{ margin: '0 0 1rem 0', fontSize: '1.25rem', color: '#2d3748' }}>
              Available Indexes ({indexes.length})
            </h3>
            {indexes.length === 0 ? (
              <p style={{ color: '#718096', textAlign: 'center', padding: '1rem' }}>No indexes found</p>
            ) : (
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {indexes.map((idx) => (
                  <li key={idx} style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    padding: '0.75rem',
                    background: 'white',
                    borderRadius: '6px',
                    marginBottom: '0.5rem',
                    border: '1px solid #e2e8f0'
                  }}>
                    <span style={{ color: '#2d3748', fontSize: '0.95rem', fontWeight: '500' }}>{idx}</span>
                    <button
                      onClick={() => handleDeleteIndex(idx)}
                      style={{
                        padding: '0.5rem 1rem',
                        borderRadius: '6px',
                        border: 'none',
                        background: '#fed7d7',
                        color: '#c53030',
                        cursor: 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        fontSize: '0.9rem',
                        fontWeight: '600',
                        transition: 'all 0.2s ease'
                      }}
                      onMouseOver={(e) => e.target.style.background = '#fc8181'}
                      onMouseOut={(e) => e.target.style.background = '#fed7d7'}
                    >
                      <Trash2 size={16} />
                      Delete
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        .spin {
          animation: spin 1s linear infinite;
        }
      `}</style>
    </div>
  );
}