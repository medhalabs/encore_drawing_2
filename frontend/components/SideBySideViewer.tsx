"use client";

import { apiUrl } from "@/lib/api";

interface Props {
  uploadUrl: string;
  masterImageUrl: string;
  masterName: string;
}

export default function SideBySideViewer({ uploadUrl, masterImageUrl, masterName }: Props) {
  return (
    <div className="grid md:grid-cols-2 gap-4">
      <div className="rounded-xl border border-slate-800 overflow-hidden">
        <div className="px-4 py-2 bg-slate-900 border-b border-slate-800 text-sm font-medium">Your Sketch</div>
        <img src={apiUrl(uploadUrl)} alt="Uploaded sketch" className="w-full h-64 object-contain bg-slate-950" />
      </div>
      <div className="rounded-xl border border-slate-800 overflow-hidden">
        <div className="px-4 py-2 bg-slate-900 border-b border-slate-800 text-sm font-medium">
          Master: {masterName}
        </div>
        <img src={apiUrl(masterImageUrl)} alt="Master drawing" className="w-full h-64 object-contain bg-slate-950" />
      </div>
    </div>
  );
}
