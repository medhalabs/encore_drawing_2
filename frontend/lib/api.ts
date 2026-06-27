const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AgentTraceStep {
  step: string;
  status: string;
  message: string;
  data: Record<string, unknown>;
}

export interface ScoreBreakdown {
  retrieval_score: number;
  vector_score?: number;
  vision_score: number;
  feedback_boost: number;
  combined_score: number;
}

export interface MasterSummary {
  key: string;
  id: string;
  name: string;
  category: string;
  segment_count: number;
  part_class: string;
  image_url: string;
}

export interface MatchedMaster {
  key: string;
  id: string;
  name: string;
  category: string;
  image_url: string;
  master_lengths: number[];
}

export interface TopCandidate {
  key: string;
  name: string;
  category: string;
  image_url: string;
  combined_score: number;
  vision_score: number;
  reasoning: string;
}

export interface MatchResult {
  job_id: string;
  matched_master: MatchedMaster | null;
  no_match?: boolean;
  confidence: number;
  extracted_lengths: number[];
  filled_json: Record<string, unknown>;
  agent_trace: AgentTraceStep[];
  upload_image_url: string;
  warnings: string[];
  score_breakdown?: ScoreBreakdown | null;
  top_candidates?: TopCandidate[];
}

export interface FeedbackResponse {
  entry: {
    feedback_id: string;
    master_key: string;
    previous_master_key: string;
    created_at: string;
  };
  filled_json: Record<string, unknown>;
  message: string;
}

export function apiUrl(path: string): string {
  return `${API_URL}${path}`;
}

export async function listMasters(): Promise<MasterSummary[]> {
  const response = await fetch(apiUrl("/api/v1/masters"));
  if (!response.ok) throw new Error("Failed to load masters");
  return response.json();
}

export async function matchDrawing(file: File): Promise<MatchResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(apiUrl("/api/v1/match"), {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Match failed" }));
    throw new Error(err.detail || "Match failed");
  }

  return response.json();
}

export async function submitFeedback(payload: {
  job_id: string;
  master_key: string;
  lengths: number[];
  note?: string;
}): Promise<FeedbackResponse> {
  const response = await fetch(apiUrl("/api/v1/feedback"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Failed to save correction" }));
    throw new Error(err.detail || "Failed to save correction");
  }

  return response.json();
}

// ── Training API ──────────────────────────────────────────────────────────────

export interface ClassStat {
  key: string;
  category: string;
  name: string;
  master_count: number;
  correction_count: number;
  total: number;
}

export interface TrainingStatus {
  is_training: boolean;
  model_version: string | null;
  total_classes: number;
  total_training_images: number;
  classes: ClassStat[];
}

export interface UploadResult {
  saved: number;
  skipped: number;
  master_key: string;
  filenames: string[];
}

export interface RetrainResponse {
  triggered: boolean;
  message: string;
  total_training_images: number;
}

export interface TrainingProgress {
  is_training: boolean;
  current_epoch: number;
  total_epochs: number;
  epoch_losses: number[];
  current_loss: number | null;
  percent: number;
  total_images: number;
  images_processed: number;
  images_per_epoch: number;
}

export async function getTrainingProgress(): Promise<TrainingProgress> {
  const res = await fetch(apiUrl("/api/v1/training/progress"));
  if (!res.ok) throw new Error("Failed to load progress");
  return res.json();
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  const res = await fetch(apiUrl("/api/v1/training/status"));
  if (!res.ok) throw new Error("Failed to load training status");
  return res.json();
}

export async function uploadTrainingImages(
  masterKey: string,
  files: File[]
): Promise<UploadResult> {
  const form = new FormData();
  form.append("master_key", masterKey);
  for (const f of files) form.append("files", f);
  const res = await fetch(apiUrl("/api/v1/training/upload"), {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Upload failed");
  }
  return res.json();
}

export interface TrainingImage {
  feedback_id: string;
  master_key: string;
  filename: string;
  created_at: string;
  image_url: string;
}

export async function listTrainingImages(masterKey: string): Promise<TrainingImage[]> {
  const res = await fetch(apiUrl(`/api/v1/training/images?master_key=${encodeURIComponent(masterKey)}`));
  if (!res.ok) throw new Error("Failed to load images");
  return res.json();
}

export async function deleteTrainingImage(feedbackId: string): Promise<void> {
  const res = await fetch(apiUrl(`/api/v1/training/images/${feedbackId}`), { method: "DELETE" });
  if (!res.ok) throw new Error("Failed to delete image");
}

export async function stopTraining(): Promise<{ stopped: boolean; message: string }> {
  const res = await fetch(apiUrl("/api/v1/training/stop"), { method: "POST" });
  if (!res.ok) throw new Error("Failed to stop training");
  return res.json();
}

export async function restartTraining(): Promise<RetrainResponse> {
  const res = await fetch(apiUrl("/api/v1/training/restart"), { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Restart failed" }));
    throw new Error(err.detail || "Restart failed");
  }
  return res.json();
}

export async function triggerRetrain(): Promise<RetrainResponse> {
  const res = await fetch(apiUrl("/api/v1/training/retrain"), { method: "POST" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Retrain failed" }));
    throw new Error(err.detail || "Retrain failed");
  }
  return res.json();
}

export function downloadJson(result: MatchResult): void {
  if (!result.matched_master) return;
  const blob = new Blob([JSON.stringify(result.filled_json, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${result.matched_master.key.replace("/", "-")}-filled.json`;
  a.click();
  URL.revokeObjectURL(url);
}


export interface StreamEventHandlers {
  onStep: (step: AgentTraceStep & { received_at?: string }) => void;
  onResult: (result: MatchResult) => void;
  onError: (message: string) => void;
}

export async function matchDrawingStream(
  file: File,
  handlers: StreamEventHandlers,
  signal?: AbortSignal,
  useLlm: boolean = true,
): Promise<MatchResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("use_llm", String(useLlm));

  const response = await fetch(apiUrl("/api/v1/match/stream"), {
    method: "POST",
    body: formData,
    signal,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: "Match stream failed" }));
    throw new Error(err.detail || "Match stream failed");
  }

  if (!response.body) {
    throw new Error("No response stream from server");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResult: MatchResult | null = null;
  let streamError: string | null = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const parts = buffer.split("\n\n");
    buffer = parts.pop() || "";

    for (const part of parts) {
      if (!part.trim()) continue;
      let eventType = "message";
      let dataLine = "";
      for (const line of part.split("\n")) {
        if (line.startsWith("event:")) eventType = line.slice(6).trim();
        if (line.startsWith("data:")) dataLine = line.slice(5).trim();
      }
      if (!dataLine) continue;
      const parsed = JSON.parse(dataLine);

      if (eventType === "step") {
        handlers.onStep({ ...parsed, received_at: new Date().toISOString() });
      } else if (eventType === "result") {
        finalResult = parsed as MatchResult;
        handlers.onResult(finalResult);
      } else if (eventType === "error") {
        streamError = parsed.detail || "Match failed";
        handlers.onError(streamError);
      }
    }
  }

  if (streamError) {
    throw new Error(streamError);
  }
  if (!finalResult) {
    throw new Error("Stream ended without a result");
  }
  return finalResult;
}
