"use client";

import { useEffect, useState } from "react";
import type { MasterSummary, MatchResult } from "@/lib/api";
import { apiUrl, listMasters, submitFeedback } from "@/lib/api";

interface Props {
  result: MatchResult;
  onCorrected: (updated: MatchResult, message: string) => void;
  onCancel: () => void;
}

async function fetchMasterTemplateLengths(masterKey: string): Promise<number[]> {
  const [category, basename] = masterKey.split("/");
  const response = await fetch(apiUrl(`/api/v1/masters/${category}/${basename}`));
  if (!response.ok) return [];
  const data = await response.json();
  return data.lengths ?? [];
}

export default function CorrectionPanel({ result, onCorrected, onCancel }: Props) {
  const [masters, setMasters] = useState<MasterSummary[]>([]);
  const [masterKey, setMasterKey] = useState(
    result.matched_master?.key ?? result.top_candidates?.[0]?.key ?? ""
  );
  const [lengths, setLengths] = useState<number[]>([...result.extracted_lengths]);
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listMasters()
      .then(setMasters)
      .catch(() => setError("Could not load master catalog"));
  }, []);

  useEffect(() => {
    const master = masters.find((m) => m.key === masterKey);
    if (master && lengths.length !== master.segment_count) {
      const next = [...lengths];
      while (next.length < master.segment_count) next.push(0);
      setLengths(next.slice(0, master.segment_count));
    }
  }, [masterKey, masters, lengths.length]);

  const handleSave = async () => {
    setSaving(true);
    setError(null);
    try {
      const response = await submitFeedback({
        job_id: result.job_id,
        master_key: masterKey,
        lengths,
        note: note || undefined,
      });

      const master = masters.find((m) => m.key === masterKey);
      const templateLengths = await fetchMasterTemplateLengths(masterKey);

      const updated: MatchResult = {
        ...result,
        matched_master: {
          key: masterKey,
          id: master?.id ?? result.matched_master?.id ?? "",
          name: master?.name || masterKey.split("/")[1],
          category: master?.category ?? masterKey.split("/")[0],
          image_url: `/api/v1/masters/${masterKey}/image`,
          master_lengths: templateLengths.length
            ? templateLengths
            : (result.matched_master?.master_lengths ?? []),
        },
        extracted_lengths: lengths,
        filled_json: response.filled_json,
        confidence: 1,
        no_match: false,
      };
      onCorrected(updated, response.message);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Save failed");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="rounded-xl border border-blue-800 bg-blue-950/20 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-blue-200">Correct this match</h3>
        <button onClick={onCancel} className="text-sm text-slate-400 hover:text-slate-200">
          Cancel
        </button>
      </div>

      <div>
        <label className="text-sm text-slate-400 block mb-1">Correct master drawing</label>
        <select
          value={masterKey}
          onChange={(e) => setMasterKey(e.target.value)}
          className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
        >
          {masters.map((m) => (
            <option key={m.key} value={m.key}>
              {m.category}/{m.name || m.key.split("/")[1]} ({m.segment_count} segments)
            </option>
          ))}
        </select>

        {/* Live preview of the currently-selected master — cross-verify before saving */}
        {masterKey && (
          <div className="mt-2">
            <p className="text-xs text-slate-500 mb-1">Preview of selected master</p>
            <div className="rounded-lg bg-white p-2 flex items-center justify-center" style={{ height: 160 }}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                key={masterKey}
                src={apiUrl(`/api/v1/masters/${masterKey}/image`)}
                alt={masterKey}
                className="max-h-full max-w-full object-contain"
              />
            </div>
          </div>
        )}
      </div>

      <div>
        <label className="text-sm text-slate-400 block mb-2">Corrected lengths (mm)</label>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
          {lengths.map((len, i) => (
            <div key={i}>
              <span className="text-xs text-slate-500">Seg {i + 1}</span>
              <input
                type="number"
                value={len}
                onChange={(e) => {
                  const next = [...lengths];
                  next[i] = Number(e.target.value);
                  setLengths(next);
                }}
                className="w-full rounded-lg bg-slate-900 border border-slate-700 px-2 py-1.5 text-sm mt-0.5"
              />
            </div>
          ))}
        </div>
      </div>

      <div>
        <label className="text-sm text-slate-400 block mb-1">Note (optional)</label>
        <input
          type="text"
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="e.g. confused with apron-2"
          className="w-full rounded-lg bg-slate-900 border border-slate-700 px-3 py-2 text-sm"
        />
      </div>

      {error && <p className="text-sm text-red-300">{error}</p>}

      <button
        onClick={handleSave}
        disabled={saving}
        className="w-full py-2.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-sm font-medium"
      >
        {saving ? "Saving correction…" : "Save correction & train"}
      </button>
      <p className="text-xs text-slate-500">
        Saves sketch + corrected JSON. Future similar uploads will prefer this master.
      </p>
    </div>
  );
}
