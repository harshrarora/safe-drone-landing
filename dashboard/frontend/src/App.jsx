import React, { useState } from "react";

export default function App() {
  const [imageFile, setImageFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [status, setStatus] = useState("Connected • Demo Mode");

  const runRealInference = async () => {
    if (!imageFile) {
      alert("Please upload an image first");
      return;
    }

    setStatus("Running inference...");

    const formData = new FormData();
    formData.append("image", imageFile);

    try {
      const response = await fetch("http://localhost:8000/infer", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend error");
      }

      const data = await response.json();
      console.log("Backend response:", data);

      // ✅ SAFETY NORMALIZATION (VERY IMPORTANT)
      const consensus = Number(data.uncertainty?.consensus ?? 0);
      const uncertaintyValue = Number(data.uncertainty?.bbox_uncertainty ?? 0);
      const safetyScore = Number(data.final_score ?? 0);

      const riskFactors =
        data.safety_assessment?.risk_factors?.map((r) => ({
          type: String(r.type || "unknown").replace("_", " "),
          severity: String(r.severity || "low").toUpperCase(),
        })) ?? [];

      setPrediction({
  decision: data.final_decision || "ABORT",
  safety_score: data.final_score ?? 0,

  uncertainty: {
    consensus: data.uncertainty?.consensus ?? 0,
    uncertainty: data.uncertainty?.bbox_uncertainty ?? 0,
    type: "MC Dropout",
  },

  risk_factors: (data.risk_factors || []).map((r) => {
    // 🔹 HUMAN-FRIENDLY MAPPING
    if (r.type === "safety_module_error") {
      return {
        type: "No UAP / UAI detected",
        severity: "LOW",
      };
    }

    return {
      type: r.type.replaceAll("_", " "),
      severity: r.severity.toUpperCase(),
    };
  }),

  timestamp: new Date().toLocaleTimeString(),
});


      setStatus("Connected • Live Inference");
    } catch (error) {
      console.error("Inference error:", error);
      alert("Inference failed. Check backend logs.");
      setStatus("Disconnected");
    }
  };

  const decisionStyles = {
    SAFE: "from-emerald-500 to-emerald-700",
    CAUTION: "from-amber-400 to-amber-600",
    ABORT: "from-rose-500 to-rose-700",
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-900/40 via-slate-900 to-slate-950 text-slate-100 p-10">
      {/* Header */}
      <header className="mb-10 flex items-center justify-between">
        <div>
          <h1 className="text-5xl font-extrabold tracking-tight">
            Landing Zone Safety Dashboard
          </h1>
          <p className="mt-2 text-slate-400 max-w-xl">
            Uncertainty-aware decision intelligence for autonomous aerial landing
          </p>
        </div>
        <div className="flex items-center gap-2 text-sm text-emerald-400">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          {status}
        </div>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Inference Control */}
        <div className="bg-slate-900/70 backdrop-blur rounded-2xl p-8 border border-slate-800">
          <h2 className="text-xl font-semibold mb-6">Inference Control</h2>

          <input
            type="file"
            accept="image/*"
            onChange={(e) => {
              const file = e.target.files[0];
              setImageFile(file);
              setPreview(URL.createObjectURL(file));
            }}
            className="mb-4 block w-full text-sm"
          />

          {preview && (
            <img
              src={preview}
              alt="Preview"
              className="mt-4 rounded-lg max-h-40 object-contain border border-slate-700"
            />
          )}

          <button
            onClick={runRealInference}
            className="mt-6 w-full py-4 rounded-xl font-semibold bg-gradient-to-r from-sky-500 to-indigo-600"
          >
            Run Analysis
          </button>
        </div>

        {/* Decision */}
        {prediction && (
          <div className="bg-slate-900/70 rounded-2xl p-8 border border-slate-800">
            <h2 className="text-xl font-semibold mb-6">System Decision</h2>

            <div
              className={`rounded-2xl p-8 text-center bg-gradient-to-br ${decisionStyles[prediction.decision]}`}
            >
              <div className="text-6xl font-black">
                {prediction.decision}
              </div>
              <div className="mt-4 text-lg">
                Safety Score: {prediction.safety_score.toFixed(3)}
              </div>
            </div>
          </div>
        )}

        {/* Uncertainty */}
        {prediction && (
          <div className="bg-slate-900/70 rounded-2xl p-8 border border-slate-800">
            <h2 className="text-xl font-semibold mb-6">Model Uncertainty</h2>

            <p>
              Consensus: {(prediction.uncertainty.consensus * 100).toFixed(1)}%
            </p>
            <p>
              Uncertainty: {prediction.uncertainty.uncertainty.toFixed(3)}
            </p>
          </div>
        )}
      </div>

      {/* Risk Factors */}
      {prediction && (
        <div className="mt-10 bg-slate-900/70 rounded-2xl p-8 border border-slate-800">
          <h2 className="text-xl font-semibold mb-4">Risk Factors</h2>

          {prediction.risk_factors.length === 0 ? (
            <p className="text-emerald-400">✓ No operational risks detected</p>
          ) : (
            prediction.risk_factors.map((r, i) => (
              <div key={i} className="text-rose-400">
                {r.type} ({r.severity})
              </div>
            ))
          )}
        </div>
      )}

      <footer className="mt-16 text-center text-sm text-slate-500 border-t border-slate-800 pt-6">
        Made by <span className="font-semibold text-slate-300">Team INIT to Win It</span>
      </footer>
    </div>
  );
}
