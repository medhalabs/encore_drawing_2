"use client";

interface Props {
  data: Record<string, unknown>;
}

export default function JsonPreview({ data }: Props) {
  return (
    <pre className="rounded-xl border border-slate-800 bg-slate-950 p-4 overflow-auto text-xs leading-relaxed max-h-96">
      {JSON.stringify(data, null, 2)}
    </pre>
  );
}
