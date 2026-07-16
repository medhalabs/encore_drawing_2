"use client";

import { useState } from "react";
import type { MatchResult } from "@/lib/api";
import { apiUrl, downloadJson, submitFeedback } from "@/lib/api";
import SideBySideViewer from "./SideBySideViewer";
import JsonPreview from "./JsonPreview";
import CorrectionPanel from "./CorrectionPanel";

interface Props {
  result: MatchResult;
  onResultUpdate?: (updated: MatchResult) => void;
}

function ScoreBreakdownPanel({ result }: { result: MatchResult }) {
  if (!result.score_breakdown) return null;
  const confidence = result.score_breakdown.vision_score;

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-sm">
      <h3 className="font-medium mb-3">Classifier confidence</h3>
      <p className={`text-2xl font-semibold ${confidence >= 0.65 ? "text-emerald-300" : "text-amber-300"}`}>
        {Math.round(confidence * 100)}%
      </p>
    </div>
  );
}

export default function MatchResults({ result, onResultUpdate }: Props) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const [thumbsState, setThumbsState] = useState<"idle" | "loading" | "done">("idle");
  const confidencePct = Math.round(result.confidence * 100);
  const master = result.matched_master;
  const noMatch = result.no_match || !master;

  async function handleThumbsUp() {
    if (!master || thumbsState !== "idle") return;
    setThumbsState("loading");
    try {
      await submitFeedback({
        job_id: result.job_id,
        master_key: master.key,
        lengths: result.extracted_lengths,
        note: "thumbs-up confirmation",
      });
      setThumbsState("done");
      setSavedMessage(`✓ Saved as training example for ${master.name || master.key}`);
    } catch {
      setThumbsState("idle");
      setSavedMessage("Failed to save training example — please try again.");
    }
  }

  if (noMatch) {
    return (
      <div className="space-y-6">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-semibold text-amber-200">No match found</h2>
            <p className="text-slate-400 text-sm mt-1">
              The sketch did not match any master drawing with sufficient confidence.
            </p>
          </div>
          <button
            onClick={() => setShowCorrection(true)}
            className="px-4 py-2 rounded-lg border border-slate-600 hover:border-slate-400 text-sm font-medium transition-colors"
          >
            Assign master manually
          </button>
        </div>

        {result.warnings.length > 0 && (
          <div className="rounded-lg border border-amber-800 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
            {result.warnings.map((w, i) => (
              <p key={i}>⚠ {w}</p>
            ))}
          </div>
        )}

        <ScoreBreakdownPanel result={result} />

        {result.top_candidates && result.top_candidates.length > 0 && (
          <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
            <h3 className="font-medium mb-3">Closest candidates</h3>
            <div className="space-y-3">
              {result.top_candidates.map((c) => (
                <div
                  key={c.key}
                  className="flex items-center gap-4 rounded-lg border border-slate-800 bg-slate-950/50 p-3"
                >
                  <img
                    src={apiUrl(c.image_url)}
                    alt={c.name || c.key}
                    className="h-16 w-16 object-contain rounded bg-slate-900"
                  />
                  <div className="min-w-0 flex-1">
                    <p className="font-medium truncate">{c.name || c.key}</p>
                    <p className="text-xs text-slate-400">
                      {c.category} · {Math.round(c.combined_score * 100)}% combined ·{" "}
                      {Math.round(c.vision_score * 100)}% vision
                    </p>
                    {c.reasoning && <p className="text-xs text-slate-500 mt-1">{c.reasoning}</p>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {savedMessage && (
          <div className="rounded-lg border border-emerald-800 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-200">
            {savedMessage}
          </div>
        )}

        {showCorrection && (
          <CorrectionPanel
            result={result}
            onCancel={() => setShowCorrection(false)}
            onCorrected={(updated, message) => {
              onResultUpdate?.(updated);
              setSavedMessage(message);
              setShowCorrection(false);
            }}
          />
        )}

        <div className="rounded-xl border border-slate-800 overflow-hidden">
          <div className="px-4 py-2 bg-slate-900 border-b border-slate-800 text-sm font-medium">Your Sketch</div>
          <img
            src={apiUrl(result.upload_image_url)}
            alt="Uploaded sketch"
            className="w-full h-64 object-contain bg-slate-950"
          />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold">{master.name || master.key}</h2>
          <p className="text-slate-400 text-sm mt-1">
            {master.category} · ID {master.id}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${
              (result.score_breakdown?.vision_score ?? result.confidence) >= 0.65
                ? "bg-emerald-900/50 text-emerald-300"
                : "bg-amber-900/50 text-amber-300"
            }`}
          >
            {confidencePct}% confidence
          </span>
          <button
            onClick={handleThumbsUp}
            disabled={thumbsState !== "idle"}
            title="This match is correct — save for training"
            className={`px-3 py-2 rounded-lg border text-sm font-medium transition-colors flex items-center gap-1.5 ${
              thumbsState === "done"
                ? "border-emerald-600 bg-emerald-900/40 text-emerald-300 cursor-default"
                : "border-slate-600 hover:border-emerald-500 hover:text-emerald-300 disabled:opacity-50 disabled:cursor-wait"
            }`}
          >
            {thumbsState === "done" ? "👍 Saved for training" : thumbsState === "loading" ? "Saving…" : "👍 Correct"}
          </button>
          <button
            onClick={() => setShowCorrection((v) => !v)}
            className="px-4 py-2 rounded-lg border border-slate-600 hover:border-slate-400 text-sm font-medium transition-colors"
          >
            {showCorrection ? "Hide correction" : "Correct this match"}
          </button>
          <button
            onClick={() => downloadJson(result)}
            className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm font-medium transition-colors"
          >
            Download JSON
          </button>
        </div>
      </div>

      <ScoreBreakdownPanel result={result} />

      {savedMessage && (
        <div className="rounded-lg border border-emerald-800 bg-emerald-950/30 px-4 py-3 text-sm text-emerald-200">
          {savedMessage}
        </div>
      )}

      {showCorrection && (
        <CorrectionPanel
          result={result}
          onCancel={() => setShowCorrection(false)}
          onCorrected={(updated, message) => {
            onResultUpdate?.(updated);
            setSavedMessage(message);
            setShowCorrection(false);
          }}
        />
      )}

      {result.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-800 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
          {result.warnings.map((w, i) => (
            <p key={i}>⚠ {w}</p>
          ))}
        </div>
      )}

      <SideBySideViewer
        uploadUrl={result.upload_image_url}
        masterImageUrl={master.image_url}
        masterName={master.name || master.key}
      />

      <div className="rounded-xl border border-slate-800 overflow-hidden">
        <div className="px-4 py-3 bg-slate-900 border-b border-slate-800 font-medium">Dimensions</div>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400">
              <th className="text-left px-4 py-2">Segment</th>
              <th className="text-left px-4 py-2">Master (template)</th>
              <th className="text-left px-4 py-2">Extracted (handwritten)</th>
            </tr>
          </thead>
          <tbody>
            {result.extracted_lengths.map((len, i) => (
              <tr key={i} className="border-b border-slate-800/50">
                <td className="px-4 py-2">{i + 1}</td>
                <td className="px-4 py-2 text-slate-400">{master.master_lengths[i] ?? "—"}</td>
                <td className="px-4 py-2 font-medium text-emerald-300">{len}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div>
        <h3 className="font-medium mb-2">Filled JSON Output</h3>
        <JsonPreview data={result.filled_json} />
      </div>
    </div>
  );
}
