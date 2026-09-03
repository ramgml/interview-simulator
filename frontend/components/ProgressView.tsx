"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { getProgress, type Progress as ProgressData } from "@/lib/api";

/** Вкладка «Прогресс»: баллы завершённых сессий, средние по компетенциям, тренд. */
export default function ProgressView() {
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getProgress()
      .then(setProgress)
      .catch((exc: unknown) => setError(exc instanceof Error ? exc.message : "Ошибка загрузки"));
  }, []);

  if (error) return <p className="text-sm text-destructive">{error}</p>;
  if (!progress) return <p className="text-sm text-muted-foreground">Загрузка…</p>;

  const completed = progress.sessions;
  const averages = Object.entries(progress.averages);

  return (
    <div className="flex flex-col gap-4">
      {completed.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Пока нет завершённых сессий — пройдите первое собеседование.
        </p>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Баллы по сессиям
              {progress.trend === "up" && <Badge>Рост 📈</Badge>}
              {progress.trend === "down" && <Badge variant="secondary">Снижение 📉</Badge>}
            </CardTitle>
            <CardDescription>Завершённые собеседования в хронологическом порядке</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Дата</TableHead>
                  <TableHead>Позиция</TableHead>
                  <TableHead className="text-right">Балл</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {completed.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell>{new Date(item.date).toLocaleString("ru-RU")}</TableCell>
                    <TableCell>{item.position_title}</TableCell>
                    <TableCell className="text-right">{item.overall_score ?? "—"}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
      {averages.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Средние баллы по компетенциям</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {averages.map(([name, score]) => (
              <div key={name} className="flex items-center gap-3">
                <span className="w-48 shrink-0 text-sm">{name}</span>
                <Progress value={score * 10} className="flex-1" />
                <span className="w-10 text-right text-sm tabular-nums">{score.toFixed(1)}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
