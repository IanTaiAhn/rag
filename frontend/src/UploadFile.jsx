import { useState } from "react";

export default function UploadDocument({ refreshIndexes }) {
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

  return (
    <div>
      <input
        type="file"
        accept=".pdf,.txt"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={handleUpload}>Upload</button>
      <button onClick={handleBuildIndex}>Build Index</button>

      <p>{status}</p>
    </div>
  );
}
