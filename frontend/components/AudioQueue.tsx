"use client";

import { useEffect, useRef, useState } from "react";
import { synthesizeSpeech } from "@/lib/api";

/** Разбиение на предложения: [.!?…]\s, конечная фраза без пунктуации тоже играется. */
export function splitSentences(text: string): string[] {
  return text
    .split(/(?<=[.!?…])\s+/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);
}

/**
 * Озвучка текста вопроса: последовательные POST /api/tts по предложениям,
 * очередь <audio> по цепочке ended → next. isSpeaking=true на всё время очереди,
 * чтобы Recorder был заблокирован во время озвучки; onDone — по завершении.
 * Текст вопроса рендерится сразу, не дожидаясь озвучки.
 */
export default function AudioQueue({
  text,
  voice,
  onDone,
}: {
  text: string;
  voice?: string;
  onDone?: () => void;
}) {
  const [speaking, setSpeaking] = useState(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const urlsRef = useRef<string[]>([]);

  useEffect(() => {
    if (!text) return;
    let cancelled = false;
    setSpeaking(true);

    async function playQueue() {
      const chunks = splitSentences(text);
      for (const chunk of chunks) {
        if (cancelled) return;
        try {
          const blob = await synthesizeSpeech(chunk, voice);
          if (cancelled) return;
          const url = URL.createObjectURL(blob);
          urlsRef.current.push(url);
          await new Promise<void>((resolve) => {
            const audio = new Audio(url);
            audioRef.current = audio;
            audio.onended = () => resolve();
            audio.onerror = () => resolve();
            void audio.play().catch(() => resolve());
          });
        } catch {
          // Синтез одной фразы упал (бэкенд недоступен) — озвучиваем остальные,
          // текст и так виден; падение сессии не устраиваем.
        }
      }
      if (!cancelled) {
        setSpeaking(false);
        onDone?.();
      }
    }

    void playQueue();
    return () => {
      cancelled = true;
      audioRef.current?.pause();
      for (const url of urlsRef.current) URL.revokeObjectURL(url);
      urlsRef.current = [];
      setSpeaking(false);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  return (
    <p className="text-base leading-relaxed whitespace-pre-wrap">
      {text}
      {speaking && (
        <span className="ml-2 inline-flex items-center gap-1 text-muted-foreground align-middle">
          <span className="size-1.5 rounded-full bg-foreground animate-pulse" />
          озвучивается…
        </span>
      )}
    </p>
  );
}
