"use client";

import { useCallback, useRef, useState } from "react";
import { apiUrl } from "@/lib/api";
import type { MatchResult } from "@/lib/api";

interface DrawingResult {
  index: number;
  job_id: string;
  batch_id: string;
  crop_url: string;
  result: MatchResult | null;
  status: "pending" | "processing" | "done" | "error";
}

export default function BatchPage() {
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [useLlm, setUseLlm] = useState(true);
  const [processing, setProcessing] = useState(false);
  const [total, setTotal] = useState(0);
  const [drawings, setDrawings] = useState<DrawingResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<DrawingResult | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const onFile = (f: File) => {
    setFile(f);
    setDrawings([]);
    setError(null);
    setTotal(0);
    setSelected(null);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, []);

  const handleProcess = async () => {
    if (!file) return;
    setProcessing(true);
    setDrawings([]);
    setError(null);
    setSelected(null);

    const form = new FormData();
    form.append("file", file);
    form.append("use_llm", String(useLlm));

    try {
      const res = await fetch(apiUrl("/api/v1/batch/upload"), {
        method: "POST",
        body: form,
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Upload failed" }));
        throw new Error(err.detail || "Upload failed");
      }

      const reader = res.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          if (!part.trim()) continue;
          let eventType = "message";
          let dataLine = "";
          for (const line of part.split("\n")) {
            if (line.startsWith("event:")) eventType = line.slice(6).trim();
            if (line.startsWith("data:")) dataLine = line.slice(5).trim();
          }
          if (!dataLine) continue;
          const parsed = JSON.parse(dataLine);

          if (eventType === "start") {
            setTotal(parsed.total);
            setDrawings(
              Array.from({ length: parsed.total }, (_, i) => ({
                index: i,
                job_id: "",
                batch_id: parsed.batch_id,
                crop_url: "",
                result: null,
                status: "pending",
              }))
            );
          } else if (eventType === "drawing") {
            setDrawings((prev) =>
              prev.map((d) =>
                d.index === parsed.index
                  ? {
                      ...d,
                      job_id: parsed.job_id,
                      batch_id: parsed.batch_id,
                      crop_url: parsed.crop_url,
                      result: parsed.result,
                      status: parsed.result ? "done" : "error",
                    }
                  : d
              )
            );
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Processing failed");
    } finally {
      setProcessing(false);
    }
  };

  const downloadJson = (d: DrawingResult) => {
    if (!d.result?.filled_json) return;
    const blob = new Blob([JSON.stringify(d.result.filled_json, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const key = d.result.matched_master?.key?.replace("/", "-") ?? `drawing-${d.index + 1}`;
    a.download = `${key}-filled.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const done = drawings.filter((d) => d.status === "done").length;

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl font-semibold">Batch Drawing Processor</h1>
          <p className="text-slate-400 text-sm mt-1">
            Upload a PDF or image with multiple drawings — each is detected, matched, and filled automatically.
          </p>
        </div>

        {/* Upload zone */}
        <div
          className={`rounded-xl border-2 border-dashed p-10 text-center cursor-pointer transition-colors ${
            dragging ? "border-blue-500 bg-blue-950/20" : "border-slate-700 hover:border-slate-500"
          }`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.webp"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); }}
          />
          {file ? (
            <div>
              <p className="text-white font-medium">{file.name}</p>
              <p className="text-slate-400 text-sm mt-1">{(file.size / 1024).toFixed(0)} KB — click to change</p>
            </div>
          ) : (
            <div>
              <p className="text-slate-300">Drop a PDF or image here, or click to browse</p>
              <p className="text-slate-500 text-sm mt-1">PDF, PNG, JPG, WEBP — up to 50 MB</p>
            </div>
          )}
        </div>

        {/* Controls */}
        <div className="flex items-center gap-6">
          <label className="flex items-center gap-3 cursor-pointer">
            <div
              className={`relative w-11 h-6 rounded-full transition-colors ${useLlm ? "bg-blue-600" : "bg-slate-700"}`}
              onClick={() => setUseLlm((v) => !v)}
            >
              <div className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${useLlm ? "left-6" : "left-1"}`} />
            </div>
            <span className="text-sm text-slate-300">Use LLM for matching</span>
          </label>

          <button
            onClick={handleProcess}
            disabled={!file || processing}
            className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium transition-colors"
          >
            {processing ? `Processing… (${done}/${total || "?"})` : "Process File"}
          </button>
        </div>

        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/30 p-4 text-red-300 text-sm">{error}</div>
        )}

        {/* Results grid */}
        {drawings.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-medium text-lg">
                {done} / {drawings.length} drawings processed
              </h2>
              {done === drawings.length && !processing && (
                <button
                  onClick={() => drawings.filter(d => d.status === "done").forEach(downloadJson)}
                  className="text-sm px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors"
                >
                  Download all JSONs
                </button>
              )}
            </div>

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {drawings.map((d) => (
                <div
                  key={d.index}
                  onClick={() => d.status === "done" ? setSelected(d) : null}
                  className={`rounded-xl border overflow-hidden cursor-pointer transition-all ${
                    d.status === "done"
                      ? "border-slate-700 hover:border-slate-500 bg-slate-900"
                      : d.status === "error"
                      ? "border-red-800 bg-red-950/20"
                      : "border-slate-800 bg-slate-900/50"
                  } ${selected?.index === d.index ? "ring-2 ring-blue-500" : ""}`}
                >
                  {/* Crop image */}
                  <div className="bg-white aspect-square flex items-center justify-center overflow-hidden">
                    {d.crop_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img
                        src={apiUrl(d.crop_url)}
                        alt={`Drawing ${d.index + 1}`}
                        className="w-full h-full object-contain"
                      />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center bg-slate-800">
                        {d.status === "pending" ? (
                          <div className="w-6 h-6 border-2 border-slate-600 border-t-blue-400 rounded-full animate-spin" />
                        ) : (
                          <span className="text-slate-500 text-xs">No image</span>
                        )}
                      </div>
                    )}
                  </div>

                  {/* Info */}
                  <div className="p-3">
                    <p className="text-xs text-slate-400 mb-1">Drawing {d.index + 1}</p>
                    {d.status === "pending" && (
                      <p className="text-xs text-slate-500">Waiting…</p>
                    )}
                    {d.status === "processing" && (
                      <p className="text-xs text-blue-400">Processing…</p>
                    )}
                    {d.status === "error" && (
                      <p className="text-xs text-red-400">Match failed</p>
                    )}
                    {d.status === "done" && d.result && (
                      <>
                        <p className="text-sm font-medium truncate">
                          {d.result.matched_master?.name ?? "No match"}
                        </p>
                        <p className="text-xs text-slate-400 mt-0.5">
                          {Math.round(d.result.confidence * 100)}% confidence
                        </p>
                      </>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Detail panel */}
        {selected && selected.result && (
          <div className="rounded-xl border border-slate-700 bg-slate-900 p-6 space-y-4">
            <div className="flex items-start justify-between gap-4">
              <div>
                <h3 className="font-semibold text-lg">
                  Drawing {selected.index + 1} — {selected.result.matched_master?.name ?? "No match"}
                </h3>
                <p className="text-slate-400 text-sm mt-1">
                  {selected.result.matched_master?.category} · {Math.round(selected.result.confidence * 100)}% confidence
                </p>
              </div>
              <div className="flex gap-2">
                <button
                  onClick={() => downloadJson(selected)}
                  className="px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-sm font-medium transition-colors"
                >
                  Download JSON
                </button>
                <button
                  onClick={() => setSelected(null)}
                  className="px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 text-sm transition-colors"
                >
                  Close
                </button>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-6">
              {/* Crop */}
              <div>
                <p className="text-xs text-slate-400 mb-2">Your drawing</p>
                <div className="rounded-lg bg-white p-3 flex items-center justify-center" style={{ minHeight: 200 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={apiUrl(selected.crop_url)}
                    alt="Crop"
                    className="max-h-64 max-w-full object-contain"
                  />
                </div>
              </div>
              {/* Master */}
              <div>
                <p className="text-xs text-slate-400 mb-2">Matched master</p>
                <div className="rounded-lg bg-white p-3 flex items-center justify-center" style={{ minHeight: 200 }}>
                  {selected.result.matched_master && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={apiUrl(selected.result.matched_master.image_url)}
                      alt="Master"
                      className="max-h-64 max-w-full object-contain"
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Extracted lengths */}
            {selected.result.extracted_lengths?.length > 0 && (
              <div>
                <p className="text-xs text-slate-400 mb-2">Extracted lengths</p>
                <div className="flex flex-wrap gap-2">
                  {selected.result.extracted_lengths.map((l, i) => (
                    <span key={i} className="px-3 py-1 rounded-lg bg-slate-800 text-sm font-mono">
                      {l}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Filled JSON preview */}
            {selected.result.filled_json && (
              <div>
                <p className="text-xs text-slate-400 mb-2">Filled JSON</p>
                <pre className="rounded-lg bg-slate-950 border border-slate-800 p-3 text-xs overflow-auto max-h-48 text-slate-300">
                  {JSON.stringify(selected.result.filled_json, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
