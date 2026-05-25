"use client";

import { useRef, useState } from "react";
import ImageUpload from "@/components/ImageUpload";
import MatchProgress, { type LiveTraceStep } from "@/components/MatchProgress";
import MatchResults from "@/components/MatchResults";
import { matchDrawingStream, type MatchResult } from "@/lib/api";

const STEP_ORDER = ["upload", "analyze", "retrieve", "compare", "match", "extract", "validate"];

export default function HomePage() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MatchResult | null>(null);
  const [liveTrace, setLiveTrace] = useState<LiveTraceStep[]>([]);
  const [currentStep, setCurrentStep] = useState<string | undefined>();
  const abortRef = useRef<AbortController | null>(null);

  const handleMatch = async () => {
    if (!file) return;
    abortRef.current?.abort();
    abortRef.current = new AbortController();

    setLoading(true);
    setError(null);
    setResult(null);
    setLiveTrace([]);
    setCurrentStep("upload");

    try {
      const data = await matchDrawingStream(
        file,
        {
          onStep: (step) => {
            setLiveTrace((prev) => {
              const next = prev.filter((p) => p.step !== step.step);
              return [...next, step];
            });
            const idx = STEP_ORDER.indexOf(step.step);
            if (idx >= 0 && idx < STEP_ORDER.length - 1) {
              setCurrentStep(STEP_ORDER[idx + 1]);
            } else {
              setCurrentStep(undefined);
            }
          },
          onResult: (r) => setResult(r),
          onError: (msg) => setError(msg),
        },
        abortRef.current.signal
      );
      setResult(data);
      setLiveTrace(data.agent_trace.map((s) => ({ ...s })));
    } catch (e) {
      if (e instanceof Error && e.name !== "AbortError") {
        setError(e.message);
      }
    } finally {
      setLoading(false);
      setCurrentStep(undefined);
    }
  };

  const displayTrace = liveTrace.length > 0 ? liveTrace : result?.agent_trace ?? [];

  return (
    <div className="grid lg:grid-cols-2 gap-8">
      <div className="space-y-6">
        <ImageUpload onFileSelect={setFile} disabled={loading} />
        <button
          onClick={handleMatch}
          disabled={!file || loading}
          className="w-full py-3 rounded-xl bg-blue-600 hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed font-medium transition-colors"
        >
          {loading ? "Processing… watch pipeline on the right" : "Match Drawing"}
        </button>
        {error && (
          <div className="rounded-lg border border-red-800 bg-red-950/40 px-4 py-3 text-sm text-red-200">
            {error}
          </div>
        )}
      </div>

      <div className="space-y-6">
        {(loading || displayTrace.length > 0 || result) && (
          <MatchProgress trace={displayTrace} loading={loading} currentStep={currentStep} />
        )}
        {result && !loading && (
          <MatchResults result={result} onResultUpdate={setResult} />
        )}
        {!loading && !result && displayTrace.length === 0 && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/30 p-8 text-center text-slate-500">
            Upload a sketch and click Match Drawing to see the live debug pipeline
          </div>
        )}
      </div>
    </div>
  );
}
