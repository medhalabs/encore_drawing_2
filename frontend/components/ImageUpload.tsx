"use client";

import { useCallback, useState } from "react";

interface Props {
  onFileSelect: (file: File) => void;
  disabled?: boolean;
}

export default function ImageUpload({ onFileSelect, disabled }: Props) {
  const [preview, setPreview] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      setPreview(URL.createObjectURL(file));
      onFileSelect(file);
    },
    [onFileSelect]
  );

  return (
    <div className="space-y-4">
      <label
        className={`flex flex-col items-center justify-center w-full h-56 border-2 border-dashed rounded-xl cursor-pointer transition-colors ${
          dragOver ? "border-blue-400 bg-blue-950/30" : "border-slate-700 hover:border-slate-500"
        } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragOver(false);
          const file = e.dataTransfer.files[0];
          if (file) handleFile(file);
        }}
      >
        <input
          type="file"
          accept="image/*"
          className="hidden"
          disabled={disabled}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
        <div className="text-center p-6">
          <p className="text-lg font-medium">Drop handwritten sketch here</p>
          <p className="text-sm text-slate-400 mt-1">PNG, JPG up to 10MB</p>
        </div>
      </label>
      {preview && (
        <div className="rounded-xl overflow-hidden border border-slate-800">
          <img src={preview} alt="Upload preview" className="w-full max-h-80 object-contain bg-slate-900" />
        </div>
      )}
    </div>
  );
}
