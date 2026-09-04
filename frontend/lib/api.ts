// Типизированный клиент API (ARCHITECTURE §API): прямой fetch без AI-SDK.
// Базовый URL — NEXT_PUBLIC_API_BASE_URL (встраивается в клиентский бандл на сборке),
// по умолчанию бэкенд FastAPI на http://localhost:8100 (см. Makefile-цель backend).
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8100";

export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

/** FastAPI-ошибки — {detail: string}; прочие статусы читаем как текст. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { cache: "no-store", ...init });
  if (!response.ok) {
    let detail = `Ошибка запроса (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // тело не JSON — оставляем общий текст со статусом
    }
    throw new ApiError(response.status, detail);
  }
  return (await response.json()) as T;
}


// --- Сессии ---------------------------------------------------------------------

export type Seniority = "junior" | "middle" | "senior" | "lead";
export type InterviewStyle = "friendly" | "strict" | "realistic";
export type SessionStatus = "created" | "in_progress" | "completed" | "failed";

export interface SessionCreateInput {
  vacancy_text: string;
  seniority?: Seniority;
  language?: "ru" | "en";
  style?: InterviewStyle;
  planned_questions?: number;
}

export interface SessionBrief {
  id: string;
  created_at: string;
  status: SessionStatus;
  position_title: string;
  seniority: string | null;
  language: string;
  style: string;
  planned_questions: number;
  overall_score: number | null;
}

export interface Turn {
  idx: number;
  role: "interviewer" | "candidate";
  text: string;
  stt_confidence: number | null;
  llm_trace_id: string | null;
}

export interface SessionState extends SessionBrief {
  vacancy_text: string;
  plan_json: {
    position_title: string;
    competencies: string[];
    rounds: { type: string; questions: { topic: string; question: string; competency: string }[] }[];
  } | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_sec: number | null;
  turns: Turn[];
}

export interface AnswerOut {
  transcript: string | null;
  question_text: string | null;
  done: boolean;
  action: string;
}

export function createSession(input: SessionCreateInput): Promise<{ id: string }> {
  return request("/api/sessions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function startSession(id: string): Promise<SessionState> {
  return request(`/api/sessions/${id}/start`, { method: "POST" });
}

export function sendTextAnswer(id: string, text: string): Promise<AnswerOut> {
  return request(`/api/sessions/${id}/answer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
}

export function sendAudioAnswer(id: string, audio: Blob): Promise<AnswerOut> {
  const form = new FormData();
  form.append("audio", audio, "answer.webm");
  return request(`/api/sessions/${id}/answer`, { method: "POST", body: form });
}

export function finishSession(id: string): Promise<SessionState> {
  return request(`/api/sessions/${id}/finish`, { method: "POST" });
}

export function listSessions(): Promise<SessionBrief[]> {
  return request("/api/sessions");
}

export function getSession(id: string): Promise<SessionState> {
  return request(`/api/sessions/${id}`);
}

// --- Отчёт и прогресс -----------------------------------------------------------

export interface CompetencyScore {
  name: string;
  score: number | null;
  comment: string | null;
}

export interface TurnFeedback {
  turn_idx: number;
  question: string | null;
  answer: string | null;
  score: number | null;
  good: string | null;
  missed: string | null;
  strong_answer: boolean | null;
}

export interface Report {
  overall_score: number | null;
  competencies: CompetencyScore[];
  turn_feedback: TurnFeedback[];
  strengths: string[];
  weaknesses: string[];
  plan: { topic: string; action: string; resources_hint: string | null }[];
  verdict: string | null;
  hire_recommendation: "strong_yes" | "yes" | "no" | "strong_no" | null;
  /** degraded: true — отчёт построен fallback-ом (нарратив без структуры, ARCHITECTURE §Оценщик) */
  degraded?: boolean;
}

export interface Progress {
  sessions: { date: string; position_title: string; overall_score: number | null }[];
  averages: Record<string, number>;
  trend: "up" | "down" | null;
}

export function getReport(id: string): Promise<Report> {
  return request(`/api/sessions/${id}/report`);
}

export function getProgress(): Promise<Progress> {
  return request("/api/progress");
}

// --- Настройки и TTS -------------------------------------------------------------

export interface SettingsRead {
  provider: "local" | "cloud";
  base_url: string;
  api_key: string | null;
  model: string;
  whisper_model: string;
  tts_voice: string;
  updated_at: string;
}

export interface SettingsUpdateInput {
  provider?: "local" | "cloud";
  base_url?: string;
  /** '***' или пустая строка бэкенд игнорирует — хранимый ключ не перетирается */
  api_key?: string;
  model?: string;
  whisper_model?: string;
  tts_voice?: string;
}

export function getSettings(): Promise<SettingsRead> {
  return request("/api/settings");
}

export function updateSettings(input: SettingsUpdateInput): Promise<SettingsRead> {
  return request("/api/settings", {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
}

export function testSettings(): Promise<{ ok: boolean }> {
  return request("/api/settings/test");
}

export async function synthesizeSpeech(text: string, voice?: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, voice }),
  });
  if (!response.ok) {
    let detail = `Ошибка синтеза речи (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      // тело не JSON — оставляем общий текст
    }
    throw new ApiError(response.status, detail);
  }
  return await response.blob();
}
