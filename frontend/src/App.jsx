import { useState, useEffect } from "react";
import UploadDocument from "./UploadFile";

export default function App() {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState(null);
  const [loading, setLoading] = useState(false);
  const [indexes, setIndexes] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState("");
  const [uploadedDocs, setUploadedDocs] = useState([]);
  
    const fetchUploadedDocs = async () => {
      const res = await fetch("http://localhost:8000/list_uploaded_docs");
      const data = await res.json();
      setUploadedDocs(data.files);
    };
    
    useEffect(() => {
      fetchUploadedDocs();
    }, []);

  const fetchIndexes = () => {
    fetch("http://localhost:8000/list_indexes")
    .then(res => res.json())
    .then(data => setIndexes(data.indexes));
  };
  
  useEffect(() => {
    fetchIndexes();
  }, []);
  
  async function askQuestion() {
    setLoading(true);
    setResponse(null);

    const res = await fetch("http://localhost:8000/ask_question", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query,
        index_name: selectedIndex
      })
    });

    const data = await res.json();
    setResponse(data);
    setLoading(false);
  }

  return (
      <div style={{ padding: "2rem", fontFamily: "sans-serif" }}>
        <h1>Local RAG Demo</h1>

        <textarea
          rows={3}
          style={{ width: "100%", marginBottom: "1rem" }}
          placeholder="Ask a question..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <button
          onClick={askQuestion}
          disabled={loading || !selectedIndex || !query.trim()}
        >
          {loading ? "Thinking..." : "Ask"}
        </button>

      <select
        value={selectedIndex}
        onChange={(e) => setSelectedIndex(e.target.value)}
        style={{
          marginBottom: "1rem",
          padding: "0.5rem 0.75rem",
          borderRadius: "6px",
          border: "1px solid #ccc",
          background: "#fff",
          fontSize: "1rem",
          cursor: "pointer",
          outline: "none",
          transition: "border-color 0.2s ease",
          width: "100%",
          maxWidth: "300px",
          display: "block",
          color: "#222" // <-- darker text
        }}
        onFocus={(e) => (e.target.style.borderColor = "#888")}
        onBlur={(e) => (e.target.style.borderColor = "#ccc")}
      >
        <option value="" disabled style={{ color: "#555" }}>
          Select an index…
        </option>
        {indexes.map((idx) => (
          <option key={idx} value={idx} style={{ color: "#222" }}>
            {idx}
          </option>
        ))}
      </select>

      <UploadDocument uploadedDocs={uploadedDocs} refreshUploadedDocs={fetchUploadedDocs} indexes={indexes} refreshIndexes={fetchIndexes}></UploadDocument>

      {response && (
        <div style={{ marginTop: "2rem" }}>
          <h2>Answer</h2>
          <p>{response.answer}</p>
      
          <h2>Context Used</h2>
          {response.context?.map((chunk, i) => (
            <div
              key={i}
              style={{
                marginBottom: "1rem",
                padding: "1rem",
                background: "#f4f4f4",
                borderRadius: "6px",
                overflowWrap: "break-word",
                wordBreak: "break-word",
                maxWidth: "100%",
                color: "#222" // <-- darker text for the whole block
              }}
            >
              <pre
                style={{
                  whiteSpace: "pre-wrap",
                  overflowWrap: "break-word",
                  wordBreak: "break-word",
                  margin: 0,
                  maxWidth: "100%",
                  color: "#222" // <-- ensure pre text is dark
                }}
              >
                {chunk}
              </pre>
            </div>
          ))}
        </div>
      )}

      </div>
  );
}
