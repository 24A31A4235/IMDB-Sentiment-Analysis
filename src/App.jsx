import { useState } from "react"

export default function App() {
  const [text, setText] = useState("")
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  const analyze = async () => {
    if (!text.trim()) return
    setLoading(true)
    setResult(null)
    setError("")
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      })
      const data = await res.json()
      setResult(data)
    } catch {
      setError("Backend not running. Make sure uvicorn is still running.")
    }
    setLoading(false)
  }

  const isPos = result?.label === "POSITIVE"

  const getBarColor = (score) => {
    if (score >= 90) return "#16a34a"
    if (score >= 75) return "#15803d"
    if (score >= 60) return "#ca8a04"
    return "#dc2626"
  }

  const getBadgeColor = (score) => {
    if (score >= 90) return "#166534"
    if (score >= 75) return "#15803d"
    if (score >= 60) return "#ca8a04"
    return "#dc2626"
  }

  return (
    <div style={{
      maxWidth: 620,
      margin: "60px auto",
      padding: "0 20px",
      fontFamily: "system-ui, sans-serif"
    }}>
      <h1 style={{ fontSize: 26, fontWeight: 700, marginBottom: 6 }}>
        Movie Review Sentiment
      </h1>
      <p style={{ color: "#666", fontSize: 14, marginBottom: 28 }}>
        Paste any movie review below. we will analze the sentiment and give you a confidence score!
      </p>

      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        placeholder="e.g. This movie was absolutely brilliant! The acting was superb..."
        style={{
          width: "100%",
          height: 150,
          padding: 14,
          fontSize: 14,
          borderRadius: 10,
          border: "1.5px solid #ddd",
          resize: "vertical",
          outline: "none",
          lineHeight: 1.6
        }}
      />

      <button
        onClick={analyze}
        disabled={loading || !text.trim()}
        style={{
          marginTop: 12,
          padding: "11px 30px",
          fontSize: 14,
          fontWeight: 700,
          background: loading ? "#999" : "#111",
          color: "#fff",
          border: "none",
          borderRadius: 8,
          cursor: loading ? "not-allowed" : "pointer"
        }}
      >
        {loading ? "Analyzing..." : "Analyze Sentiment"}
      </button>

      {error && (
        <div style={{
          marginTop: 20,
          padding: 14,
          borderRadius: 10,
          background: "#fff1f2",
          border: "1.5px solid #fca5a5",
          color: "#dc2626",
          fontSize: 13
        }}>
          {error}
        </div>
      )}

      {result && (
        <div style={{
          marginTop: 20,
          padding: 20,
          borderRadius: 12,
          background: isPos ? "#f0fdf4" : "#fff1f2",
          border: `2px solid ${isPos ? "#86efac" : "#fca5a5"}`
        }}>
          <div style={{
            fontSize: 24,
            fontWeight: 800,
            color: isPos ? "#16a34a" : "#dc2626"
          }}>
            {isPos ? "POSITIVE" : "NEGATIVE"}
          </div>

          <div style={{ fontSize: 14, color: "#555", marginTop: 6 }}>
            Confidence: <b>{result.score}%</b>
          </div>

          <div style={{
            marginTop: 8,
            padding: "4px 12px",
            borderRadius: 8,
            display: "inline-block",
            fontSize: 12,
            fontWeight: 600,
            background: getBadgeColor(result.score),
            color: "#fff"
          }}>
            {result.interpretation}
          </div>

          <div style={{ marginTop: 12 }}>
            <div style={{
              height: 8,
              borderRadius: 99,
              background: "#e5e7eb",
              overflow: "hidden"
            }}>
              <div style={{
                height: "100%",
                width: result.score + "%",
                borderRadius: 99,
                background: getBarColor(result.score),
                transition: "width 0.6s ease"
              }} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}