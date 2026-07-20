"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  type ClassStat,
  type TrainingImage,
  type TrainingProgress,
  type TrainingStatus,
  apiUrl,
  deleteTrainingImage,
  getTrainingProgress,
  getTrainingStatus,
  listTrainingImages,
  restartTraining,
  stopTraining,
  triggerRetrain,
  uploadTrainingImages,
} from "@/lib/api";

const CATEGORIES = ["Aprons", "Capping", "FootMoulds", "Gutters", "Misc", "RidgeValley", "Soakers"];

// ── Loss curve SVG chart ──────────────────────────────────────────────────────
function LossCurve({ losses, totalEpochs, currentEpoch, imagesProcessed, imagesPerEpoch }: {
  losses: number[];
  totalEpochs: number;
  currentEpoch: number;
  imagesProcessed: number;
  imagesPerEpoch: number;
}) {
  const W = 540, H = 180, PAD = { top: 16, right: 16, bottom: 36, left: 52 };
  const innerW = W - PAD.left - PAD.right;
  const innerH = H - PAD.top - PAD.bottom;

  const allLosses = losses.length > 0 ? losses : [];
  const maxL = allLosses.length > 0 ? Math.max(...allLosses) * 1.1 : 2;
  const minL = allLosses.length > 0 ? Math.max(0, Math.min(...allLosses) * 0.9) : 0;
  const range = maxL - minL || 1;

  const toX = (epoch: number) => PAD.left + (epoch / (totalEpochs - 1 || 1)) * innerW;
  const toY = (loss: number) => PAD.top + ((maxL - loss) / range) * innerH;

  // Build SVG path for completed epochs
  const points = allLosses.map((l, i) => `${toX(i).toFixed(1)},${toY(l).toFixed(1)}`);
  const linePath = points.length > 1 ? `M ${points.join(" L ")}` : "";

  // Partial line for the current in-progress epoch (straight line extending to current batch)
  const inProgressX = imagesPerEpoch > 0
    ? PAD.left + ((currentEpoch - 1 + imagesProcessed / imagesPerEpoch) / (totalEpochs - 1 || 1)) * innerW
    : null;
  const lastY = allLosses.length > 0 ? toY(allLosses[allLosses.length - 1]) : null;

  // Gradient fill under curve
  const areaPath = points.length > 1
    ? `M ${points[0]} L ${points.join(" L ")} L ${toX(allLosses.length - 1).toFixed(1)},${(PAD.top + innerH).toFixed(1)} L ${toX(0).toFixed(1)},${(PAD.top + innerH).toFixed(1)} Z`
    : "";

  // Y-axis labels
  const yTicks = 4;
  const yLabels = Array.from({ length: yTicks + 1 }, (_, i) => {
    const val = minL + (range * i) / yTicks;
    const y = toY(val);
    return { val, y };
  });

  // X-axis labels (epoch numbers)
  const xLabels = Array.from({ length: totalEpochs }, (_, i) => ({
    epoch: i + 1,
    x: toX(i),
  }));

  return (
    <div className="rounded-xl bg-slate-900/80 border border-slate-700 p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-slate-300 tracking-wide uppercase">Loss Curve</span>
        {allLosses.length > 0 && (
          <span className="text-xs text-slate-400 font-mono">
            latest: <span className="text-amber-300">{allLosses[allLosses.length - 1].toFixed(4)}</span>
            {allLosses.length > 1 && (
              <span className={`ml-2 ${allLosses[allLosses.length - 1] < allLosses[allLosses.length - 2] ? "text-emerald-400" : "text-red-400"}`}>
                {allLosses[allLosses.length - 1] < allLosses[allLosses.length - 2] ? "▼" : "▲"}
                {Math.abs(allLosses[allLosses.length - 1] - allLosses[allLosses.length - 2]).toFixed(4)}
              </span>
            )}
          </span>
        )}
      </div>
      <svg width="100%" viewBox={`0 0 ${W} ${H}`} className="overflow-visible">
        <defs>
          <linearGradient id="lossGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.3" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.02" />
          </linearGradient>
          <linearGradient id="inProgressGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.6" />
            <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.1" />
          </linearGradient>
        </defs>

        {/* Grid lines */}
        {yLabels.map(({ y }, i) => (
          <line key={i} x1={PAD.left} y1={y} x2={W - PAD.right} y2={y}
            stroke="#334155" strokeWidth="1" strokeDasharray={i === 0 ? "0" : "3,3"} />
        ))}
        {xLabels.map(({ x }, i) => (
          <line key={i} x1={x} y1={PAD.top} x2={x} y2={PAD.top + innerH}
            stroke="#1e293b" strokeWidth="1" />
        ))}

        {/* Area fill */}
        {areaPath && <path d={areaPath} fill="url(#lossGrad)" />}

        {/* Loss line */}
        {linePath && (
          <path d={linePath} fill="none" stroke="#f59e0b" strokeWidth="2.5"
            strokeLinecap="round" strokeLinejoin="round" />
        )}

        {/* In-progress epoch dotted extension */}
        {inProgressX !== null && lastY !== null && allLosses.length > 0 && (
          <line
            x1={toX(allLosses.length - 1)} y1={lastY}
            x2={inProgressX} y2={lastY}
            stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4,3" opacity="0.5"
          />
        )}

        {/* Epoch dots */}
        {allLosses.map((l, i) => (
          <circle key={i} cx={toX(i)} cy={toY(l)} r="4"
            fill="#f59e0b" stroke="#1e293b" strokeWidth="2">
            <title>Epoch {i + 1}: {l.toFixed(4)}</title>
          </circle>
        ))}

        {/* Current epoch pulse dot */}
        {inProgressX !== null && lastY !== null && allLosses.length > 0 && (
          <circle cx={inProgressX} cy={lastY} r="4" fill="#f59e0b" opacity="0.7">
            <animate attributeName="r" values="3;6;3" dur="1.5s" repeatCount="indefinite" />
            <animate attributeName="opacity" values="0.7;0.2;0.7" dur="1.5s" repeatCount="indefinite" />
          </circle>
        )}

        {/* Y-axis labels */}
        {yLabels.map(({ val, y }, i) => (
          <text key={i} x={PAD.left - 6} y={y + 4} textAnchor="end"
            fontSize="9" fill="#64748b" fontFamily="monospace">
            {val.toFixed(2)}
          </text>
        ))}

        {/* X-axis labels */}
        {xLabels.map(({ epoch, x }) => (
          <text key={epoch} x={x} y={PAD.top + innerH + 16} textAnchor="middle"
            fontSize="9" fill={epoch <= currentEpoch ? "#94a3b8" : "#334155"}
            fontFamily="monospace">
            {epoch}
          </text>
        ))}

        {/* Axes */}
        <line x1={PAD.left} y1={PAD.top} x2={PAD.left} y2={PAD.top + innerH}
          stroke="#475569" strokeWidth="1.5" />
        <line x1={PAD.left} y1={PAD.top + innerH} x2={W - PAD.right} y2={PAD.top + innerH}
          stroke="#475569" strokeWidth="1.5" />

        {/* Axis labels */}
        <text x={PAD.left - 38} y={PAD.top + innerH / 2} textAnchor="middle"
          fontSize="9" fill="#64748b" transform={`rotate(-90, ${PAD.left - 38}, ${PAD.top + innerH / 2})`}>
          Loss
        </text>
        <text x={PAD.left + innerW / 2} y={H - 2} textAnchor="middle"
          fontSize="9" fill="#64748b">
          Epoch
        </text>
      </svg>

      {/* Image throughput mini-stat */}
      {imagesPerEpoch > 0 && (
        <div className="mt-2 flex items-center gap-3 text-xs text-slate-500">
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-sky-500 animate-pulse" />
            <span>
              <span className="text-slate-300 font-mono">{imagesProcessed.toLocaleString()}</span>
              {" / "}
              <span className="font-mono">{imagesPerEpoch.toLocaleString()}</span>
              {" images this epoch"}
            </span>
          </div>
          <div className="ml-auto font-mono">
            {Math.round((imagesProcessed / imagesPerEpoch) * 100)}%
          </div>
        </div>
      )}
    </div>
  );
}

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
  const [stopping, setStopping] = useState(false);
  const [retrainMsg, setRetrainMsg] = useState<string | null>(null);
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const progressPollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [filterCategory, setFilterCategory] = useState<string>("All");
  const [search, setSearch] = useState("");
  const [classImages, setClassImages] = useState<TrainingImage[]>([]);
  const [loadingImages, setLoadingImages] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
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

  const loadClassImages = useCallback(async (key: string) => {
    if (!key) { setClassImages([]); return; }
    setLoadingImages(true);
    try {
      setClassImages(await listTrainingImages(key));
    } catch {
      setClassImages([]);
    } finally {
      setLoadingImages(false);
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  useEffect(() => {
    loadClassImages(selectedKey);
  }, [selectedKey, loadClassImages]);

  const handleDeleteImage = async (feedbackId: string) => {
    setDeletingId(feedbackId);
    try {
      await deleteTrainingImage(feedbackId);
      setClassImages((prev) => prev.filter((img) => img.feedback_id !== feedbackId));
      await loadStatus();
    } catch {
      // ignore
    } finally {
      setDeletingId(null);
    }
  };

  // Poll status while training is active
  useEffect(() => {
    if (status?.is_training) {
      pollRef.current = setInterval(loadStatus, 3000);
    } else {
      if (pollRef.current) clearInterval(pollRef.current);
    }
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [status?.is_training, loadStatus]);

  // Poll fine-grained progress every 1.5 s while training
  useEffect(() => {
    if (!status?.is_training) {
      if (progressPollRef.current) clearInterval(progressPollRef.current);
      return;
    }
    const tick = async () => {
      try { setProgress(await getTrainingProgress()); } catch { /* ignore */ }
    };
    tick();
    progressPollRef.current = setInterval(tick, 1500);
    return () => { if (progressPollRef.current) clearInterval(progressPollRef.current); };
  }, [status?.is_training]);

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
      await Promise.all([loadStatus(), loadClassImages(selectedKey)]);
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

  const handleStop = async () => {
    setStopping(true);
    setRetrainMsg(null);
    try {
      const res = await stopTraining();
      setRetrainMsg(res.message);
      await loadStatus();
    } catch (e) {
      setRetrainMsg(e instanceof Error ? e.message : "Stop failed");
    } finally {
      setStopping(false);
    }
  };

  const handleRestart = async () => {
    setRetraining(true);
    setRetrainMsg(null);
    try {
      const res = await restartTraining();
      setRetrainMsg(res.message);
      await loadStatus();
    } catch (e) {
      setRetrainMsg(e instanceof Error ? e.message : "Restart failed");
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

          {/* Uploaded images for selected class */}
          {selectedKey && (
            <div className="border-t border-slate-800 pt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-sm font-semibold text-slate-300">
                  Uploaded sketches for <span className="text-blue-400">{selectedKey.split("/")[1]}</span>
                </h3>
                <span className="text-xs text-slate-500">{classImages.length} image{classImages.length !== 1 ? "s" : ""}</span>
              </div>
              {loadingImages ? (
                <p className="text-xs text-slate-500">Loading…</p>
              ) : classImages.length === 0 ? (
                <p className="text-xs text-slate-600 italic">No uploaded sketches yet for this class.</p>
              ) : (
                <div className="grid grid-cols-4 gap-2">
                  {classImages.map((img) => (
                    <div key={img.feedback_id} className="relative group rounded-lg overflow-hidden border border-slate-700 bg-slate-900">
                      <img
                        src={apiUrl(img.image_url)}
                        alt={img.filename}
                        className="w-full aspect-square object-cover"
                      />
                      <button
                        onClick={() => handleDeleteImage(img.feedback_id)}
                        disabled={deletingId === img.feedback_id}
                        title="Remove this training image"
                        className="absolute top-1 right-1 w-5 h-5 rounded-full bg-red-600 hover:bg-red-500 text-white text-xs font-bold flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity disabled:opacity-50"
                      >
                        {deletingId === img.feedback_id ? "…" : "×"}
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Retrain / Stop / Restart */}
          <div className="border-t border-slate-800 pt-4 space-y-3">
            <h3 className="text-lg font-semibold text-slate-200">Retrain classifier</h3>
            <p className="text-xs text-slate-500">
              Runs 10 epochs in a background thread. Hot-swaps weights when done. Auto-triggers every 10 corrections from the Match page.
            </p>

            {/* Primary action row */}
            {!status?.is_training ? (
              <button
                onClick={handleRetrain}
                disabled={retraining}
                className="w-full py-2.5 rounded-xl bg-amber-600 hover:bg-amber-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors"
              >
                {retraining ? "Starting…" : "Retrain now"}
              </button>
            ) : (
              /* Training is running */
              <div className="space-y-3">
                {/* Status header */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-amber-400 font-medium">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse inline-block" />
                    Training in progress
                  </div>
                  {progress && (
                    <span className="text-xs text-slate-400 font-mono">
                      {progress.total_images.toLocaleString()} total images
                    </span>
                  )}
                </div>

                {/* Epoch progress bar */}
                {progress && (
                  <div>
                    <div className="flex justify-between text-xs text-slate-400 mb-1">
                      <span>
                        Epoch{" "}
                        <span className="text-slate-200 font-semibold">{progress.current_epoch}</span>
                        {" / "}{progress.total_epochs}
                      </span>
                      {progress.current_loss !== null && (
                        <span>Loss: <span className="text-slate-200 font-mono">{progress.current_loss.toFixed(4)}</span></span>
                      )}
                    </div>
                    <div className="w-full h-2 bg-slate-700 rounded-full overflow-hidden">
                      <div
                        className="h-full rounded-full bg-amber-500 transition-all duration-500"
                        style={{ width: `${(progress.current_epoch / progress.total_epochs) * 100}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Live loss chart */}
                {progress ? (
                  <LossCurve
                    losses={progress.epoch_losses}
                    totalEpochs={progress.total_epochs}
                    currentEpoch={progress.current_epoch}
                    imagesProcessed={progress.images_processed}
                    imagesPerEpoch={progress.images_per_epoch}
                  />
                ) : (
                  <div className="rounded-xl bg-slate-900/80 border border-slate-700 p-4 animate-pulse">
                    <div className="h-4 w-24 bg-slate-700 rounded mb-3" />
                    <div className="h-32 bg-slate-800 rounded" />
                  </div>
                )}

                {/* Stop / Restart */}
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={handleStop}
                    disabled={stopping}
                    className="py-2.5 rounded-xl bg-red-700 hover:bg-red-600 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors"
                  >
                    {stopping ? "Stopping…" : "⏹ Stop"}
                  </button>
                  <button
                    onClick={handleRestart}
                    disabled={retraining}
                    className="py-2.5 rounded-xl bg-slate-600 hover:bg-slate-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium text-sm transition-colors"
                  >
                    {retraining ? "Restarting…" : "↺ Restart"}
                  </button>
                </div>
              </div>
            )}

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
