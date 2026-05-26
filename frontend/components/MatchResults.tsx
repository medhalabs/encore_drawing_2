"use client";

import { useState } from "react";
import type { MatchResult } from "@/lib/api";
import { downloadJson } from "@/lib/api";
import SideBySideViewer from "./SideBySideViewer";
import JsonPreview from "./JsonPreview";
import CorrectionPanel from "./CorrectionPanel";

interface Props {
  result: MatchResult;
  onResultUpdate?: (updated: MatchResult) => void;
}

export default function MatchResults({ result, onResultUpdate }: Props) {
  const [showCorrection, setShowCorrection] = useState(false);
  const [savedMessage, setSavedMessage] = useState<string | null>(null);
  const confidencePct = Math.round(result.confidence * 100);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-semibold">{result.matched_master.name || result.matched_master.key}</h2>
          <p className="text-slate-400 text-sm mt-1">
            {result.matched_master.category} · ID {result.matched_master.id}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <span
            className={`px-3 py-1 rounded-full text-sm font-medium ${
              (result.score_breakdown?.vision_score ?? result.confidence) >= 0.65 ? "bg-emerald-900/50 text-emerald-300" : "bg-amber-900/50 text-amber-300"
            }`}
          >
            {confidencePct}% confidence
          </span>
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

      
      {result.score_breakdown && (
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4 text-sm">
          <h3 className="font-medium mb-3">Match score breakdown</h3>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            <div>
              <p className="text-slate-500 text-xs">Retrieval</p>
              <p className="font-medium">{result.score_breakdown.retrieval_score}</p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Vector similarity</p>
              <p className="font-medium">
                {result.score_breakdown.vector_score != null
                  ? `${Math.round(result.score_breakdown.vector_score * 100)}%`
                  : "—"}
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Vision shape</p>
              <p className={`font-medium ${result.score_breakdown.vision_score >= 0.65 ? "text-emerald-300" : "text-amber-300"}`}>
                {Math.round(result.score_breakdown.vision_score * 100)}%
              </p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Feedback boost</p>
              <p className="font-medium">{result.score_breakdown.feedback_boost || "—"}</p>
            </div>
            <div>
              <p className="text-slate-500 text-xs">Combined</p>
              <p className="font-medium">{Math.round(result.score_breakdown.combined_score * 100)}%</p>
            </div>
          </div>
          {result.score_breakdown.vision_score < 0.65 && (
            <p className="text-amber-300 text-xs mt-3">
              Vision shape match is below 65% — this match may be wrong. Use Correct this match.
            </p>
          )}
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

      {result.warnings.length > 0 && (
        <div className="rounded-lg border border-amber-800 bg-amber-950/30 px-4 py-3 text-sm text-amber-200">
          {result.warnings.map((w, i) => (
            <p key={i}>⚠ {w}</p>
          ))}
        </div>
      )}

      <SideBySideViewer
        uploadUrl={result.upload_image_url}
        masterImageUrl={result.matched_master.image_url}
        masterName={result.matched_master.name || result.matched_master.key}
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
                <td className="px-4 py-2 text-slate-400">{result.matched_master.master_lengths[i] ?? "—"}</td>
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
