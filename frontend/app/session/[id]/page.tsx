"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import AudioQueue from "@/components/AudioQueue";
import Recorder from "@/components/Recorder";
import {
  finishSession,
  getSession,
  startSession,
  ApiError,
  type AnswerOut,
  type SessionState,
} from "@/lib/api";

/** Живое интервью: лента ходов, озвучка последнего вопроса, ход кандидата, досрочный финал. */
export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [session, setSession] = useState<SessionState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [speaking, setSpeaking] = useState(true);
  const [finishing, setFinishing] = useState(false);
  const startedRef = useRef(false);

  const goToReport = useCallback(() => {
    router.push(`/session/${id}/report`);
  }, [router, id]);

  useEffect(() => {
    if (startedRef.current) return;
    startedRef.current = true;
    // Вход на уже начатую сессию (refresh/ремаунт): стартуем только created,
    // иначе тянем состояние как есть — иначе бэк отвечает 409 «Сессия уже начата».
    getSession(id)
      .then(async (state) => {
        if (state.status === "created") {
          setSession(await startSession(id));
        } else if (state.status === "completed") {
          goToReport();
        } else {
          setSession(state);
        }
      })
      .catch((exc: unknown) =>
        setError(exc instanceof ApiError ? exc.message : "Не удалось загрузить сессию"),
      );
  }, [id, goToReport]);

  function handleAnswered(answer: AnswerOut) {
    if (answer.done) {
      goToReport();
      return;
    }
    // После хода перечитываем состояние: новый вопрос интервьюера уже в turns.
    void getSession(id)
      .then(setSession)
      .catch((exc: unknown) =>
        setError(exc instanceof Error ? exc.message : "Не удалось обновить сессию"),
      );
  }

  async function handleFinish() {
    setFinishing(true);
    try {
      await finishSession(id);
      goToReport();
    } catch (exc) {
      setFinishing(false);
      setError(exc instanceof ApiError ? exc.message : "Не удалось завершить сессию");
    }
  }

  if (error) {
    return (
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 flex flex-col gap-4">
        <p className="text-sm text-destructive">{error}</p>
        <Button variant="outline" asChild className="self-start">
          <Link href="/">На главную</Link>
        </Button>
      </main>
    );
  }
  if (!session) {
    return (
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
        <p className="text-sm text-muted-foreground">Готовим интервью…</p>
      </main>
    );
  }

  // Последний interviewer-ход — текущий вопрос, его озвучивает AudioQueue.
  const lastInterviewer = [...session.turns].reverse().find((turn) => turn.role === "interviewer");

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 flex flex-col gap-6">
      <div className="flex items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="text-xl font-semibold tracking-tight">{session.position_title}</h1>
          <p className="text-sm text-muted-foreground">
            Вопрос {session.turns.filter((turn) => turn.role === "interviewer").length} из{" "}
            {session.planned_questions}
          </p>
        </div>
        <Dialog>
          <DialogTrigger asChild>
            <Button variant="outline">Завершить досрочно</Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Завершить интервью?</DialogTitle>
              <DialogDescription>
                Будет сформирован отчёт по уже заданным вопросам. Продолжить нельзя.
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <DialogClose asChild>
                <Button variant="outline">Продолжить интервью</Button>
              </DialogClose>
              <Button variant="destructive" disabled={finishing} onClick={() => void handleFinish()}>
                {finishing ? "Готовим отчёт…" : "Завершить"}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      <div className="flex flex-col gap-3">
        {session.turns.map((turn) => {
          const isCurrentQuestion = lastInterviewer !== undefined && turn.idx === lastInterviewer.idx;
          return (
            <Card key={turn.idx} className={turn.role === "interviewer" ? "" : "bg-muted/40"}>
              <CardHeader className="py-3">
                <CardTitle className="flex items-center gap-2 text-sm">
                  {turn.role === "interviewer" ? "Интервьюер" : "Вы"}
                  {turn.role === "candidate" && turn.stt_confidence !== null && (
                    <Badge variant="outline" title="Уверенность распознавания речи">
                      STT {Math.round(turn.stt_confidence * 100)}%
                    </Badge>
                  )}
                </CardTitle>
              </CardHeader>
              <CardContent className="pb-4">
                {turn.role === "interviewer" && isCurrentQuestion ? (
                  <AudioQueue
                    text={turn.text}
                    onSpeakingStart={() => setSpeaking(true)}
                    onDone={() => setSpeaking(false)}
                  />
                ) : (
                  <p className="text-base leading-relaxed whitespace-pre-wrap">{turn.text}</p>
                )}
              </CardContent>
            </Card>
          );
        })}
      </div>

      <Card>
        <CardHeader className="py-3">
          <CardTitle className="text-sm">Ваш ход</CardTitle>
          <CardDescription>Говорите, удерживая кнопку, или ответьте текстом</CardDescription>
        </CardHeader>
        <CardContent className="pb-4">
          <Recorder sessionId={id} isSpeaking={speaking} onAnswered={handleAnswered} />
        </CardContent>
      </Card>
    </main>
  );
}
