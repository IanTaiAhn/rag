import { useState } from "react";

export default function UploadDocument({ uploadedDocs, refreshUploadedDocs, indexes, refreshIndexes }) {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState("");

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("http://localhost:8000/upload_document", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    setStatus(data.message || "Upload complete");

    await refreshUploadedDocs();
  };

  const handleBuildIndex = async () => {
    const res = await fetch("http://localhost:8000/build_index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ index_name: "default" }),
    });

    const data = await res.json();
    setStatus(data.message);

    await refreshIndexes();
  };

  const handleDeleteIndex = async (indexName) => {
    await fetch(`http://localhost:8000/delete_index/${indexName}`, {
      method: "DELETE",
    });

    await refreshIndexes();
  };
  
  return (
    <div>
      <input
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => setFile(e.target.files[0])}
      />

        <button onClick={handleUpload}>Upload</button>
        <button
          onClick={handleBuildIndex}
          disabled={uploadedDocs.length === 0}
        >
          Build Index
        </button>

      <p>{status}</p>
    <div>
      <h3>Available Indexes</h3>
      {indexes.length === 0 && <p>No indexes found</p>}
      <ul>
        {indexes.map((idx) => (
          <li key={idx}>
            {idx}
            <button
              style={{ marginLeft: "10px" }}
              onClick={() => handleDeleteIndex(idx)}
            >
              Delete
            </button>
          </li>
        ))}
      </ul>
    </div>
      {/* <ul>
        {uploadedDocs.map(doc => (
          <li key={doc}>{doc}</li>
        ))}
      </ul> */}
        <div>
          <h3>Uploaded Documents</h3>
          {uploadedDocs.length === 0 && <p>No documents uploaded</p>}

          <ul>
            {uploadedDocs.map(doc => (
              <li key={doc}>
                {doc}
                <button
                  onClick={async () => {
                    await fetch(`http://localhost:8000/delete_uploaded_doc/${doc}`, {
                      method: "DELETE"
                    });
                    refreshUploadedDocs();
                  }}
                >
                  Delete
                </button>
              </li>
            ))}
          </ul>
        </div>


    </div>
    
  );
}
