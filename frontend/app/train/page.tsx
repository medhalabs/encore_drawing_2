"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ClassStat,
  type TrainingStatus,
  getTrainingStatus,
  triggerRetrain,
  uploadTrainingImages,
} from "@/lib/api";

const CATEGORIES = ["Aprons", "Capping", "FootMoulds", "Gutters", "Misc", "RidgeValley", "Soakers"];

function ProgressBar({ value, max }: { value: number; max: number }) {
  const pct = max === 0 ? 0 : Math.min((value / max) * 100, 100);
  const color = value >= 10 ? "bg-emerald-500" : value >= 5 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="w-full h-1.5 bg-slate-700 rounded-full overflow-hidden">
      <div className={`h-full rounded-full transition-all ${color}`} style={{ width: `${pct}%` }} />
    </div>
  );
}

function Badge({ count }: { count: number }) {
  const cls =
    count >= 10
      ? "bg-emerald-900/60 text-emerald-300 border-emerald-700"
      : count >= 5
      ? "bg-amber-900/60 text-amber-300 border-amber-700"
      : "bg-red-900/60 text-red-300 border-red-700";
  return (
    <span className={`text-xs font-mono px-1.5 py-0.5 rounded border ${cls}`}>{count}</span>
  );
}

export default function TrainPage() {
  const [status, setStatus] = useState<TrainingStatus | null>(null);
  const [loadingStatus, setLoadingStatus] = useState(true);
  const [selectedKey, setSelectedKey] = useState<string>("");
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [retraining, setRetraining] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [search, setSearch] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const loadStatus = useCallback(async () => {
    try {
      const s = await getTrainingStatus();
      setStatus(s);
    } catch {
      // ignore
    } finally {
      setLoadingStatus(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // Poll while training is active
  useEffect(() => {
    if (status?.is_training) {
      pollRef.current = setInterval(loadStatus, 3000);
    } else {
      if (pollRef.current) clearInterval(pollRef.current);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [status?.is_training, loadStatus]);

  const handleFiles = (incoming: FileList | null) => {
    if (!incoming) return;
    const valid = Array.from(incoming).filter((f) =>
      ["image/png", "image/jpeg", "image/webp"].includes(f.type)
    );
    setFiles((prev) => [...prev, ...valid]);
    setUploadMsg(null);
    setUploadError(null);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    handleFiles(e.dataTransfer.files);
  };

  const removeFile = (i: number) => setFiles((prev) => prev.filter((_, idx) => idx !== i));

  const handleUpload = async () => {
    if (!selectedKey || files.length === 0) return;
    setUploading(true);
    setUploadMsg(null);
    setUploadError(null);
    try {
      const res = await uploadTrainingImages(selectedKey, files);
      setUploadMsg(`Saved ${res.saved} image${res.saved !== 1 ? "s" : ""} for ${selectedKey}${res.skipped ? ` (${res.skipped} skipped)` : ""}`);
      setFiles([]);
      await loadStatus();
    } catch (e) {
      setUploadError(e instanceof Error ? e.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleRetrain = async () => {
    setRetraining(true);
    setRetrainMsg(null);
    try {
      const res = await triggerRetrain();
      setRetrainMsg(res.message);
      await loadStatus();
    } catch (e) {
      setRetrainMsg(e instanceof Error ? e.message : "Retrain failed");
    } finally {
      setRetraining(false);
    }
  };

  const filteredClasses: ClassStat[] = (status?.classes ?? []).filter((c) => {
    const catMatch = filterCategory === "All" || c.category === filterCategory;
    const searchMatch = search === "" || c.name.toLowerCase().includes(search.toLowerCase()) || c.key.toLowerCase().includes(search.toLowerCase());
    return catMatch && searchMatch;
  });

  const grouped = CATEGORIES.reduce<Record<string, ClassStat[]>>((acc, cat) => {
    acc[cat] = filteredClasses.filter((c) => c.category === cat);
    return acc;
  }, {});

  const ready = (status?.classes ?? []).filter((c) => c.total >= 10).length;
  const total = status?.total_classes ?? 0;

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-slate-100">EfficientNet Training</h2>
        <p className="text-slate-400 mt-1 text-sm">
          Upload hand-drawn sketches per master shape, then retrain the classifier.
          Aim for <span className="text-amber-400 font-medium">10–20 images per class</span> for reliable fast-path matching.
        </p>
      </div>

      {/* Stats bar */}
      {status && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Total images</p>
            <p className="text-2xl font-bold text-slate-100 mt-1">{status.total_training_images}</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Classes ready (≥10)</p>
            <p className="text-2xl font-bold text-emerald-400 mt-1">{ready} / {total}</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Model version</p>
            <p className="text-2xl font-bold text-slate-100 mt-1 truncate">{status.model_version ?? "none"}</p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-slate-900/60 p-4">
            <p className="text-xs text-slate-500 uppercase tracking-wide">Status</p>
            <p className={`text-lg font-bold mt-1 ${status.is_training ? "text-amber-400 animate-pulse" : "text-emerald-400"}`}>
              {status.is_training ? "Training…" : "Idle"}
            </p>
          </div>
        </div>
      )}

      <div className="grid lg:grid-cols-2 gap-8">
        {/* Upload panel */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-slate-200">Upload sketches</h3>

          {/* Master selector */}
          <div>
            <label className="block text-sm text-slate-400 mb-1">Select master drawing</label>
            <select
              value={selectedKey}
              onChange={(e) => setSelectedKey(e.target.value)}
              className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 focus:outline-none focus:ring-2 focus:ring-blue-600"
            >
              <option value="">— choose a master —</option>
              {CATEGORIES.map((cat) => {
                const options = (status?.classes ?? []).filter((c) => c.category === cat);
                if (!options.length) return null;
                return (
                  <optgroup key={cat} label={cat}>
                    {options.map((c) => (
                      <option key={c.key} value={c.key}>
                        {c.name} ({c.correction_count} sketches)
                      </option>
                    ))}
                  </optgroup>
                );
              })}
            </select>
          </div>

          {/* Drop zone */}
          <div
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
            onClick={() => fileRef.current?.click()}
            className="border-2 border-dashed border-slate-700 hover:border-blue-600 rounded-xl p-8 text-center cursor-pointer transition-colors"
          >
            <p className="text-slate-400 text-sm">
              Drag & drop sketches here, or <span className="text-blue-400 underline">browse</span>
            </p>
            <p className="text-slate-600 text-xs mt-1">PNG, JPG, WEBP</p>
            <input
              ref={fileRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              multiple
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />
          </div>

          {/* File list */}
          {files.length > 0 && (
            <div className="space-y-1 max-h-40 overflow-y-auto pr-1">
              {files.map((f, i) => (
                <div key={i} className="flex items-center justify-between rounded-lg bg-slate-800 px-3 py-1.5">
                  <span className="text-xs text-slate-300 truncate max-w-[80%]">{f.name}</span>
                  <button
                    onClick={() => removeFile(i)}
                    className="text-slate-500 hover:text-red-400 text-xs ml-2 shrink-0"
                  >
                    ✕
                  </button>
                </div>
              ))}
            </div>
          )}

          <button
            onClick={handleUpload}
            disabled={!selectedKey || files.length === 0 || uploading}
            className="w-full py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors"
          >
            {uploading ? "Uploading…" : `Upload ${files.length > 0 ? files.length : ""} image${files.length !== 1 ? "s" : ""}`}
          </button>

          {uploadMsg && (
            <div className="rounded-lg border border-emerald-800 bg-emerald-950/40 px-4 py-2 text-sm text-emerald-300">
              {uploadMsg}
            </div>
          )}
          {uploadError && (
            <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-2 text-sm text-red-300">
              {uploadError}
            </div>
          )}

          {/* Retrain */}
          <div className="border-t border-slate-800 pt-4 space-y-3">
            <h3 className="text-lg font-semibold text-slate-200">Retrain classifier</h3>
            <p className="text-xs text-slate-500">
              Runs 10 epochs in a background thread. The server stays live and hot-swaps the new weights when done. Auto-triggers every 10 corrections from the Match page.
            </p>
            <button
              onClick={handleRetrain}
              disabled={retraining || status?.is_training}
              className="w-full py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors"
            >
              {status?.is_training ? "Training in progress…" : retraining ? "Starting…" : "Retrain now"}
            </button>
            {retrainMsg && (
              <div className="rounded-lg border border-slate-700 bg-slate-800/60 px-4 py-2 text-sm text-slate-300">
                {retrainMsg}
              </div>
            )}
          </div>
        </div>

        {/* Class coverage table */}
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-slate-200">Class coverage</h3>

          <div className="flex gap-2 flex-wrap">
            {["All", ...CATEGORIES].map((cat) => (
              <button
                key={cat}
                onClick={() => setFilterCategory(cat)}
                className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                  filterCategory === cat
                    ? "bg-blue-600 text-white"
                    : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                }`}
              >
                {cat}
              </button>
            ))}
          </div>

          <input
            type="text"
            placeholder="Search master name…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-800 px-3 py-2 text-sm text-slate-200 placeholder-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-600"
          />

          {loadingStatus ? (
            <p className="text-slate-500 text-sm">Loading…</p>
          ) : (
            <div className="space-y-4 max-h-[520px] overflow-y-auto pr-1">
              {CATEGORIES.filter((cat) => grouped[cat]?.length > 0).map((cat) => (
                <div key={cat}>
                  <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-2">{cat}</p>
                  <div className="space-y-1">
                    {grouped[cat].map((c) => (
                      <div
                        key={c.key}
                        onClick={() => setSelectedKey(c.key)}
                        className={`flex items-center gap-3 rounded-lg px-3 py-2 cursor-pointer transition-colors ${
                          selectedKey === c.key
                            ? "bg-blue-900/40 border border-blue-700"
                            : "bg-slate-800/60 hover:bg-slate-800"
                        }`}
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-xs text-slate-200 truncate">{c.name}</p>
                          <ProgressBar value={c.total} max={20} />
                        </div>
                        <Badge count={c.correction_count} />
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              {filteredClasses.length === 0 && (
                <p className="text-slate-600 text-sm">No classes match your filter.</p>
              )}
            </div>
          )}

          <div className="flex gap-4 pt-2 text-xs text-slate-500">
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-red-500 inline-block" /> &lt;5 sketches</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-amber-500 inline-block" /> 5–9</span>
            <span className="flex items-center gap-1.5"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" /> ≥10 ready</span>
          </div>
        </div>
      </div>
    </div>
  );
}
