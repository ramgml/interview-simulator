"use client";

import { useEffect, useRef, useState } from "react";
import { getStoredDeviceId, inputConstraints, setStoredDeviceId } from "@/lib/audio";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { sendAudioAnswer, sendTextAnswer, ApiError, type AnswerOut } from "@/lib/api";

/**
 * Ход кандидата: toggle-запись (клик — старт, повторный клик — стоп и отправка)
 * → MediaRecorder('audio/webm;codecs=opus') → POST /answer (multipart, поле audio);
 * fallback — ответ текстом.
 * Анти-дрожание: запись короче MIN_RECORDING_MS наружу не уходит (случайный
 * двойной клик не должен отправить пустоту).
 * Запись идёт с выбранного в настройках микрофона (localStorage `audio-input-device`,
 * см. lib/audio.ts); устройство недоступно — fallback на default и сброс выбора.
 * Весь блок disabled, пока идёт озвучка вопроса (проп isSpeaking от AudioQueue).
 * Пустой STT (422 «Речь не распознана») — inline-сообщение, запись не падает.
 */

/** Минимальная длительность записи, мс: короче — подсказка вместо отправки. */
const MIN_RECORDING_MS = 300;

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
  const streamRef = useRef<MediaStream | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const startedAtRef = useRef(0);

  /** Остановка таймера и треков стрима из ref (образец — фикс T169 в настройках аудио). */
  function releaseStream() {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  useEffect(() => {
    return () => {
      // Unmount при активной записи: отключаем колбэки рекордера (чтобы onstop
      // не отправил запись), останавливаем рекордер и треки стрима.
      const recorder = recorderRef.current;
      if (recorder) {
        recorder.ondataavailable = null;
        recorder.onstop = null;
        if (recorder.state === "recording") recorder.stop();
      }
      releaseStream();
      chunksRef.current = [];
    };
  }, []);

  async function startRecording() {
    setError(null);
    let stream: MediaStream;
    try {
      // Точный deviceId выбранного микрофона; при отказе/устаревшем id — default.
      stream = await navigator.mediaDevices.getUserMedia({
        audio: inputConstraints(getStoredDeviceId("input")),
      });
    } catch {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        setError("Микрофон недоступен. Ответьте текстом или проверьте доступ к микрофону.");
        return;
      }
      // Сохранённый id перестал работать — сбрасываем на устройство по умолчанию.
      setStoredDeviceId("input", null);
    }
    const recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
    chunksRef.current = [];
    streamRef.current = stream;
    startedAtRef.current = performance.now();
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) chunksRef.current.push(event.data);
    };
    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/webm;codecs=opus" });
      chunksRef.current = [];
      if (performance.now() - startedAtRef.current < MIN_RECORDING_MS) {
        // Анти-дрожание: короткая запись — подсказка вместо отправки.
        setError("Слишком короткая запись — нажмите и говорите дольше.");
        return;
      }
      void submitAudio(blob);
    };
    recorder.start();
    recorderRef.current = recorder;
    setSeconds(0);
    timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    setRecording(true);
  }

  /** Повторный клик: стоп записи; финальный чанк и решение об отправке — в onstop. */
  function stopRecording() {
    setRecording(false);
    releaseStream();
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (recorder?.state === "recording") recorder.stop();
  }

  /**
   * Toggle-семантика: один onClick без pointer-событий. Старт асинхронный
   * (getUserMedia), повторный клик возможен только при recording=true —
   * двойного срабатывания нет: state-машина idle → recording → sending.
   */
  function handleRecordClick() {
    if (recording) {
      stopRecording();
      return;
    }
    void startRecording();
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
          aria-pressed={recording}
          aria-label={recording ? "Остановить запись и отправить" : "Говорить"}
          className={cn(
            "size-14 rounded-full text-base",
            recording && "bg-destructive text-white hover:bg-destructive/90",
          )}
          disabled={disabled}
          onClick={handleRecordClick}
        >
          {recording ? "■" : "🎙"}
        </Button>
        <div className="flex flex-col gap-1">
          <span className="text-sm font-medium flex items-center gap-2">
            {recording && <span className="size-2.5 rounded-full bg-red-600 animate-pulse" />}
            {recording
              ? `Идёт запись… ${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")} — нажмите, чтобы отправить`
              : isSpeaking
                ? "Дождитесь конца озвучки вопроса…"
                : sending
                  ? "Распознаём и обдумываем ответ…"
                  : "Говорить"}
          </span>
          <span className="text-xs text-muted-foreground">
            {recording
              ? "Нажмите ещё раз — запись уйдёт в распознавание"
              : "Нажмите кнопку и говорите; повторный клик отправит ответ"}
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
