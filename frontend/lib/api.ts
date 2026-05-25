const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface AgentTraceStep {
  step: string;
  status: string;
  message: string;
  data: Record<string, unknown>;
}

export interface ScoreBreakdown {
  retrieval_score: number;
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

export interface MatchResult {
  job_id: string;
  matched_master: {
    key: string;
    id: string;
    name: string;
    category: string;
    image_url: string;
    master_lengths: number[];
  };
  confidence: number;
  extracted_lengths: number[];
  filled_json: Record<string, unknown>;
  agent_trace: AgentTraceStep[];
  upload_image_url: string;
  warnings: string[];
  score_breakdown?: ScoreBreakdown | null;
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

export function downloadJson(result: MatchResult): void {
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
  signal?: AbortSignal
): Promise<MatchResult> {
  const formData = new FormData();
  formData.append("file", file);

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
        handlers.onError(parsed.detail || "Match failed");
      }
    }
  }

  if (!finalResult) {
    throw new Error("Stream ended without a result");
  }
  return finalResult;
}
