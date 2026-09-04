"use client";

import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { sendAudioAnswer, sendTextAnswer, ApiError, type AnswerOut } from "@/lib/api";

/**
 * Ход кандидата: hold-to-talk (pointerdown/up/leave) → MediaRecorder('audio/webm;codecs=opus')
 * → POST /answer (multipart, поле audio); fallback — ответ текстом.
 * Весь блок disabled, пока идёт озвучка вопроса (проп isSpeaking от AudioQueue).
 * Пустой STT (422 «Речь не распознана») — inline-сообщение, запись не падает.
 */
export default function Recorder({
  sessionId,
  isSpeaking,
  onAnswered,
}: {
  sessionId: string;
  isSpeaking: boolean;
  onAnswered: (answer: AnswerOut) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [textAnswer, setTextAnswer] = useState("");

  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (recorderRef.current?.state === "recording") recorderRef.current.stop();
      recorderRef.current?.stream.getTracks().forEach((track) => track.stop());
    };
  }, []);

  async function startRecording() {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => {
        stream.getTracks().forEach((track) => track.stop());
        void submitAudio(new Blob(chunksRef.current, { type: "audio/webm;codecs=opus" }));
      };
      recorder.start();
      recorderRef.current = recorder;
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
      setRecording(true);
    } catch {
      setError("Микрофон недоступен. Ответьте текстом или проверьте доступ к микрофону.");
    }
  }

  function stopRecording() {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    setRecording(false);
    if (recorderRef.current?.state === "recording") recorderRef.current.stop();
  }

  async function submitAudio(blob: Blob) {
    setSending(true);
    setError(null);
    try {
      const answer = await sendAudioAnswer(sessionId, blob);
      onAnswered(answer);
    } catch (exc) {
      setError(
        exc instanceof ApiError
          ? exc.message
          : "Не удалось отправить запись. Проверьте соединение с сервером.",
      );
    } finally {
      setSending(false);
    }
  }

  async function submitText() {
    const trimmed = textAnswer.trim();
    if (!trimmed || sending) return;
    setSending(true);
    setError(null);
    try {
      const answer = await sendTextAnswer(sessionId, trimmed);
      setTextAnswer("");
      onAnswered(answer);
    } catch (exc) {
      setError(
        exc instanceof ApiError
          ? exc.message
          : "Не удалось отправить ответ. Проверьте соединение с сервером.",
      );
    } finally {
      setSending(false);
    }
  }

  const disabled = isSpeaking || sending;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <Button
          type="button"
          size="icon"
          aria-label={recording ? "Отпустите, чтобы отправить" : "Удерживайте, чтобы говорить"}
          className="size-14 rounded-full text-base"
          disabled={disabled}
          onPointerDown={() => void startRecording()}
          onPointerUp={stopRecording}
          onPointerLeave={stopRecording}
        >
          {recording ? "●" : "🎙"}
        </Button>
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium flex items-center gap-2">
            {recording && <span className="size-2.5 rounded-full bg-red-600 animate-pulse" />}
            {recording
              ? `Идёт запись: ${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`
              : isSpeaking
                ? "Дождитесь конца озвучки вопроса…"
                : sending
                  ? "Распознаём и обдумываем ответ…"
                  : "Удерживайте кнопку и говорите"}
          </span>
          <span className="text-xs text-muted-foreground">
            Отпустите кнопку — запись уйдёт в распознавание
          </span>
        </div>
      </div>
      {error && <p className="text-sm text-destructive">{error}</p>}
      <div className="flex flex-col gap-2">
        <label className="text-sm font-medium" htmlFor="text-answer">
          Ответить текстом
        </label>
        <Textarea
          id="text-answer"
          placeholder="Напишите ответ…"
          value={textAnswer}
          disabled={disabled}
          onChange={(event) => setTextAnswer(event.target.value)}
        />
        <Button type="button" className="self-start" disabled={disabled || !textAnswer.trim()} onClick={() => void submitText()}>
          Отправить
        </Button>
      </div>
    </div>
  );
}
