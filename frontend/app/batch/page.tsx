"use client";

import { useCallback, useRef, useState } from "react";
import { apiUrl, triggerRetrain } from "@/lib/api";
import type { MatchResult } from "@/lib/api";
import BoxEditor, { type EditorBox } from "@/components/BoxEditor";
import CorrectionPanel from "@/components/CorrectionPanel";

interface PageData {
  index: number;
  image_url: string;
  width: number;
  height: number;
  boxes: EditorBox[];
}

interface DrawingResult {
  index: number;
  job_id: string;
  batch_id: string;
  crop_url: string;
  result: MatchResult | null;
  status: "pending" | "processing" | "done" | "error";
}

type Stage = "upload" | "review" | "results";

let _bid = 0;
const mkId = () => `b_${Date.now()}_${_bid++}`;

export default function BatchPage() {
  const [stage, setStage] = useState<Stage>("upload");
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [detecting, setDetecting] = useState(false);
  const [processing, setProcessing] = useState(false);

  const [batchId, setBatchId] = useState<string>("");
  const [pages, setPages] = useState<PageData[]>([]);
  const [activePage, setActivePage] = useState(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [total, setTotal] = useState(0);
  const [drawings, setDrawings] = useState<DrawingResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<DrawingResult | null>(null);
  const [correcting, setCorrecting] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  const reset = () => {
    setStage("upload");
    setPages([]);
    setBatchId("");
    setDrawings([]);
    setTotal(0);
    setError(null);
    setSelected(null);
    setSelectedId(null);
    setActivePage(0);
  };

  const onFile = (f: File) => {
    reset();
    setFile(f);
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) onFile(f);
  }, []);

  // ── Stage 1: detect boxes ────────────────────────────────────────────────
  const handleDetect = async () => {
    if (!file) return;
    setDetecting(true);
    setError(null);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(apiUrl("/api/v1/batch/detect"), { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Detection failed" }));
        throw new Error(err.detail || "Detection failed");
      }
      const data = await res.json();
      setBatchId(data.batch_id);
      setPages(
        data.pages.map((p: PageData) => ({
          ...p,
          boxes: p.boxes.map((b) => ({ ...b, id: mkId() })),
        }))
      );
      setActivePage(0);
      setStage("review");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Detection failed");
    } finally {
      setDetecting(false);
    }
  };

  const updateBoxes = (pageIdx: number, boxes: EditorBox[]) => {
    setPages((prev) => prev.map((p) => (p.index === pageIdx ? { ...p, boxes } : p)));
  };

  const totalBoxes = pages.reduce((n, p) => n + p.boxes.length, 0);

  // ── Stage 2: classify confirmed boxes ────────────────────────────────────
  const handleClassify = async () => {
    if (!batchId || totalBoxes === 0) return;
    setProcessing(true);
    setStage("results");
    setDrawings([]);
    setError(null);
    setSelected(null);

    const body = {
      pages: pages
        .filter((p) => p.boxes.length > 0)
        .map((p) => ({
          index: p.index,
          boxes: p.boxes.map((b) => ({
            x: Math.round(b.x),
            y: Math.round(b.y),
            w: Math.round(b.w),
            h: Math.round(b.h),
          })),
        })),
    };

    try {
      const res = await fetch(apiUrl(`/api/v1/batch/${batchId}/classify`), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: "Classification failed" }));
        throw new Error(err.detail || "Classification failed");
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
                status: "pending" as const,
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
      setError(e instanceof Error ? e.message : "Classification failed");
    } finally {
      setProcessing(false);
    }
  };

  const applyCorrection = (updated: MatchResult, _message: string) => {
    if (!selected) return;
    setDrawings((prev) =>
      prev.map((d) => (d.index === selected.index ? { ...d, result: updated, status: "done" } : d))
    );
    setSelected((prev) => (prev ? { ...prev, result: updated, status: "done" } : prev));
    setCorrecting(false);
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg(null);
    try {
      const r = await triggerRetrain();
      setRetrainMsg(r.message);
    } catch (e) {
      setRetrainMsg(e instanceof Error ? e.message : "Retrain failed");
    } finally {
      setRetraining(false);
    }
  };

  const downloadJson = (d: DrawingResult) => {
    if (!d.result?.filled_json) return;
    const blob = new Blob([JSON.stringify(d.result.filled_json, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    const key = d.result.matched_master?.key?.replace("/", "-") ?? `drawing-${d.index + 1}`;
    a.download = `${key}-filled.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const done = drawings.filter((d) => d.status === "done").length;
  const page = pages[activePage];

  return (
    <div className="min-h-screen bg-slate-950 text-white">
      <div className="max-w-6xl mx-auto px-6 py-10 space-y-8">
        <div>
          <h1 className="text-2xl font-semibold">Batch Drawing Processor</h1>
          <p className="text-slate-400 text-sm mt-1">
            Upload a multi-drawing PDF or image — boxes are auto-detected, you review &amp; adjust them, then each drawing is matched.
          </p>
        </div>

        {/* Stepper */}
        <div className="flex items-center gap-2 text-sm">
          {(["upload", "review", "results"] as Stage[]).map((s, i) => (
            <div key={s} className="flex items-center gap-2">
              <span
                className={`px-3 py-1 rounded-full capitalize ${
                  stage === s ? "bg-blue-600 text-white" : "bg-slate-800 text-slate-400"
                }`}
              >
                {i + 1}. {s === "review" ? "Review boxes" : s}
              </span>
              {i < 2 && <span className="text-slate-600">→</span>}
            </div>
          ))}
        </div>

        {/* ── Stage: upload ─────────────────────────────────────────────── */}
        {stage === "upload" && (
          <>
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

            <button
              onClick={handleDetect}
              disabled={!file || detecting}
              className="px-6 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium transition-colors"
            >
              {detecting ? "Detecting drawings…" : "Detect drawings"}
            </button>
          </>
        )}

        {/* ── Stage: review boxes ───────────────────────────────────────── */}
        {stage === "review" && page && (
          <div className="space-y-4">
            <div className="rounded-lg border border-slate-800 bg-slate-900/60 p-4 text-sm text-slate-300 space-y-1">
              <p className="font-medium text-white">Review the detected boxes</p>
              <p className="text-slate-400">
                Drag a box to move it, drag its corner handles to resize. Draw on empty space to add a box.
                Select a box and press <kbd className="px-1 rounded bg-slate-800 border border-slate-700">Delete</kbd> to remove it.
              </p>
            </div>

            {/* page tabs */}
            {pages.length > 1 && (
              <div className="flex flex-wrap gap-2">
                {pages.map((p) => (
                  <button
                    key={p.index}
                    onClick={() => { setActivePage(p.index); setSelectedId(null); }}
                    className={`px-3 py-1.5 rounded-lg text-sm ${
                      p.index === activePage ? "bg-blue-600" : "bg-slate-800 hover:bg-slate-700"
                    }`}
                  >
                    Page {p.index + 1} · {p.boxes.length} {p.boxes.length === 1 ? "box" : "boxes"}
                  </button>
                ))}
              </div>
            )}

            <div className="flex items-center justify-between">
              <p className="text-sm text-slate-400">
                Page {activePage + 1} of {pages.length} · {page.boxes.length} boxes on this page · {totalBoxes} total
              </p>
              <button
                onClick={() => updateBoxes(page.index, [])}
                className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors"
              >
                Clear boxes on page
              </button>
            </div>

            <div className="rounded-xl border border-slate-700 bg-white overflow-hidden">
              <BoxEditor
                imageUrl={apiUrl(page.image_url)}
                naturalWidth={page.width}
                naturalHeight={page.height}
                boxes={page.boxes}
                onChange={(boxes) => updateBoxes(page.index, boxes)}
                selectedId={selectedId}
                onSelect={setSelectedId}
              />
            </div>

            <div className="flex items-center gap-6">
              <button onClick={reset} className="px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 text-sm transition-colors">
                Start over
              </button>
              <button
                onClick={handleClassify}
                disabled={totalBoxes === 0}
                className="px-6 py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium transition-colors"
              >
                Proceed — classify {totalBoxes} {totalBoxes === 1 ? "drawing" : "drawings"}
              </button>
            </div>
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/30 p-4 text-red-300 text-sm">{error}</div>
        )}

        {/* ── Stage: results ────────────────────────────────────────────── */}
        {stage === "results" && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-medium text-lg">
                {processing ? `Processing… (${done}/${total || "?"})` : `${done} / ${drawings.length} drawings processed`}
              </h2>
              <div className="flex gap-2">
                {done === drawings.length && !processing && drawings.length > 0 && (
                  <button
                    onClick={() => drawings.filter((d) => d.status === "done").forEach(downloadJson)}
                    className="text-sm px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors"
                  >
                    Download all JSONs
                  </button>
                )}
                <button
                  onClick={handleRetrain}
                  disabled={retraining}
                  className="text-sm px-4 py-2 rounded-lg bg-purple-700 hover:bg-purple-600 disabled:opacity-50 transition-colors"
                  title="Retrain the classifier on all saved corrections"
                >
                  {retraining ? "Starting…" : "Retrain model"}
                </button>
                <button
                  onClick={() => setStage("review")}
                  className="text-sm px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 transition-colors"
                >
                  ← Back to boxes
                </button>
              </div>
            </div>

            {retrainMsg && (
              <div className="mb-4 rounded-lg border border-purple-800 bg-purple-950/30 p-3 text-purple-200 text-sm">
                {retrainMsg}
              </div>
            )}

            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {drawings.map((d) => (
                <div
                  key={d.index}
                  onClick={() => (d.status === "done" ? (setSelected(d), setCorrecting(false)) : null)}
                  className={`rounded-xl border overflow-hidden cursor-pointer transition-all ${
                    d.status === "done"
                      ? "border-slate-700 hover:border-slate-500 bg-slate-900"
                      : d.status === "error"
                      ? "border-red-800 bg-red-950/20"
                      : "border-slate-800 bg-slate-900/50"
                  } ${selected?.index === d.index ? "ring-2 ring-blue-500" : ""}`}
                >
                  <div className="bg-white aspect-square flex items-center justify-center overflow-hidden">
                    {d.crop_url ? (
                      // eslint-disable-next-line @next/next/no-img-element
                      <img src={apiUrl(d.crop_url)} alt={`Drawing ${d.index + 1}`} className="w-full h-full object-contain" />
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
                  <div className="p-3">
                    <p className="text-xs text-slate-400 mb-1">Drawing {d.index + 1}</p>
                    {d.status === "pending" && <p className="text-xs text-slate-500">Waiting…</p>}
                    {d.status === "error" && <p className="text-xs text-red-400">Match failed</p>}
                    {d.status === "done" && d.result && (
                      <>
                        <p className="text-sm font-medium truncate">{d.result.matched_master?.name ?? "No match"}</p>
                        <p className="text-xs text-slate-400 mt-0.5">{Math.round(d.result.confidence * 100)}% confidence</p>
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
                  onClick={() => setCorrecting((v) => !v)}
                  className="px-4 py-2 rounded-lg bg-blue-700 hover:bg-blue-600 text-sm font-medium transition-colors"
                >
                  {correcting ? "Hide correction" : "Fix match"}
                </button>
                <button onClick={() => downloadJson(selected)} className="px-4 py-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 text-sm font-medium transition-colors">
                  Download JSON
                </button>
                <button onClick={() => { setSelected(null); setCorrecting(false); }} className="px-4 py-2 rounded-lg border border-slate-700 hover:border-slate-500 text-sm transition-colors">
                  Close
                </button>
              </div>
            </div>

            {correcting && (
              <CorrectionPanel
                result={selected.result}
                onCorrected={applyCorrection}
                onCancel={() => setCorrecting(false)}
              />
            )}

            <div className="grid grid-cols-2 gap-6">
              <div>
                <p className="text-xs text-slate-400 mb-2">Your drawing</p>
                <div className="rounded-lg bg-white p-3 flex items-center justify-center" style={{ minHeight: 200 }}>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={apiUrl(selected.crop_url)} alt="Crop" className="max-h-64 max-w-full object-contain" />
                </div>
              </div>
              <div>
                <p className="text-xs text-slate-400 mb-2">Matched master</p>
                <div className="rounded-lg bg-white p-3 flex items-center justify-center" style={{ minHeight: 200 }}>
                  {selected.result.matched_master && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={apiUrl(selected.result.matched_master.image_url)} alt="Master" className="max-h-64 max-w-full object-contain" />
                  )}
                </div>
              </div>
            </div>

            {selected.result.extracted_lengths?.length > 0 && (
              <div>
                <p className="text-xs text-slate-400 mb-2">Extracted lengths</p>
                <div className="flex flex-wrap gap-2">
                  {selected.result.extracted_lengths.map((l, i) => (
                    <span key={i} className="px-3 py-1 rounded-lg bg-slate-800 text-sm font-mono">{l}</span>
                  ))}
                </div>
              </div>
            )}

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
