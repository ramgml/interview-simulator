"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import ReportView from "@/components/ReportView";
import { getReport, ApiError, type Report } from "@/lib/api";

/** Страница отчёта: 404 → «Отчёт ещё не готов» + рефреш, иначе полный разбор. */
export default function ReportPage() {
  const { id } = useParams<{ id: string }>();
  const [report, setReport] = useState<Report | null>(null);
  const [notReady, setNotReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setNotReady(false);
    setError(null);
    getReport(id)
      .then((data) => {
        if (!cancelled) setReport(data);
      })
      .catch((exc: unknown) => {
        if (cancelled) return;
        if (exc instanceof ApiError && exc.status === 404) {
          setNotReady(true);
        } else {
          setError(exc instanceof Error ? exc.message : "Не удалось загрузить отчёт");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [id, tick]);

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
  if (notReady) {
    return (
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 flex flex-col items-start gap-4">
        <p className="text-sm text-muted-foreground">
          Отчёт ещё не готов — оценщик всё ещё работает. Обновите страницу через минуту.
        </p>
        <Button onClick={() => setTick((value) => value + 1)}>Обновить</Button>
      </main>
    );
  }
  if (!report) {
    return (
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8">
        <p className="text-sm text-muted-foreground">Загрузка отчёта…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 flex flex-col gap-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Отчёт по собеседованию</h1>
        <Button variant="outline" asChild>
          <Link href="/">На главную</Link>
        </Button>
      </div>
      <ReportView report={report} />
    </main>
  );
}
