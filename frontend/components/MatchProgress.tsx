"use client";

import { useMemo, useState } from "react";
import type { AgentTraceStep } from "@/lib/api";

const PIPELINE_STEPS = [
  "upload",
  "analyze",
  "retrieve",
  "compare",
  "match",
  "extract",
  "validate",
];

export type LiveTraceStep = AgentTraceStep & { received_at?: string };

interface Props {
  trace: LiveTraceStep[];
  loading: boolean;
  currentStep?: string;
}

function statusColor(status: string) {
  if (status === "completed") return "bg-emerald-500";
  if (status === "warning") return "bg-amber-500";
  if (status === "error") return "bg-red-500";
  if (status === "running") return "bg-blue-400 animate-pulse";
  return "bg-slate-700";
}

function StepDetails({ data }: { data: Record<string, unknown> }) {
  if (!data || Object.keys(data).length === 0) {
    return <p className="text-xs text-slate-500 mt-2">No debug data</p>;
  }

  return (
    <pre className="mt-2 rounded-lg bg-slate-950 border border-slate-800 p-3 text-xs overflow-auto max-h-64 text-slate-300">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}

export default function MatchProgress({ trace, loading, currentStep }: Props) {
  const [expanded, setExpanded] = useState<Record<string, boolean>>({});
  const [showAllJson, setShowAllJson] = useState(false);

  const traceByStep = useMemo(() => {
    const map = new Map<string, LiveTraceStep>();
    for (const t of trace) map.set(t.step, t);
    return map;
  }, [trace]);

  const activeStep = currentStep || (loading ? PIPELINE_STEPS.find((s) => !traceByStep.has(s)) : undefined);

  const toggle = (step: string) => setExpanded((e) => ({ ...e, [step]: !e[step] }));

  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900/50 p-6 space-y-4">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h3 className="font-medium">Live Processing Pipeline</h3>
          <p className="text-xs text-slate-400 mt-1">
            {loading ? "Streaming steps from backend…" : `${trace.length} steps recorded`}
          </p>
        </div>
        <button
          type="button"
          onClick={() => setShowAllJson((v) => !v)}
          className="text-xs px-3 py-1.5 rounded-lg border border-slate-700 hover:border-slate-500"
        >
          {showAllJson ? "Hide raw JSON" : "Show raw JSON"}
        </button>
      </div>

      <div className="space-y-2">
        {PIPELINE_STEPS.map((step) => {
          const entry = traceByStep.get(step);
          const isActive = activeStep === step && loading;
          const isDone = !!entry;
          const isOpen = expanded[step] ?? isActive ?? false;

          return (
            <div
              key={step}
              className={`rounded-lg border ${
                isActive ? "border-blue-700 bg-blue-950/20" : "border-slate-800 bg-slate-950/40"
              }`}
            >
              <button
                type="button"
                onClick={() => toggle(step)}
                className="w-full flex items-start gap-3 p-3 text-left"
              >
                <div className={`mt-1 h-3 w-3 rounded-full shrink-0 ${statusColor(entry?.status || (isActive ? "running" : "pending"))}`} />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <p className="text-sm font-medium capitalize">{step}</p>
                    {entry?.status && (
                      <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-slate-800 text-slate-400">
                        {entry.status}
                      </span>
                    )}
                    {isActive && !entry && (
                      <span className="text-[10px] uppercase tracking-wide px-1.5 py-0.5 rounded bg-blue-900 text-blue-300">
                        running
                      </span>
                    )}
                    {entry?.received_at && (
                      <span className="text-[10px] text-slate-500">{entry.received_at.slice(11, 19)}</span>
                    )}
                  </div>
                  {entry?.message && <p className="text-xs text-slate-400 mt-1">{entry.message}</p>}
                  {isActive && !entry && <p className="text-xs text-blue-300 mt-1">Waiting for Ollama…</p>}
                </div>
                <span className="text-slate-500 text-xs">{isOpen ? "▾" : "▸"}</span>
              </button>
              {isOpen && entry && <div className="px-3 pb-3"><StepDetails data={entry.data} /></div>}
            </div>
          );
        })}
      </div>

      {showAllJson && trace.length > 0 && (
        <div>
          <p className="text-xs text-slate-400 mb-2">Full pipeline trace</p>
          <pre className="rounded-lg bg-slate-950 border border-slate-800 p-3 text-xs overflow-auto max-h-96 text-slate-300">
            {JSON.stringify(trace, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}
