"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { toast } from "sonner";
import ProgressView from "@/components/ProgressView";
import { createSession, startSession, listSessions, ApiError, type SessionBrief } from "@/lib/api";

const SENIORITY_LABELS: Record<string, string> = {
  junior: "Junior",
  middle: "Middle",
  senior: "Senior",
  lead: "Lead",
};

const STYLE_LABELS: Record<string, string> = {
  friendly: "Дружелюбный",
  strict: "Строгий",
  realistic: "Реалистичный",
};

const STATUS_LABELS: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  created: { label: "Создана", variant: "outline" },
  in_progress: { label: "Идёт", variant: "secondary" },
  completed: { label: "Завершена", variant: "default" },
  failed: { label: "Ошибка", variant: "destructive" },
};

/** Главная: новая сессия (Card) + вкладки «История» и «Прогресс». */
export default function HomePage() {
  const router = useRouter();
  const [vacancyText, setVacancyText] = useState("");
  const [seniority, setSeniority] = useState("middle");
  const [language, setLanguage] = useState("ru");
  const [style, setStyle] = useState("realistic");
  const [plannedQuestions, setPlannedQuestions] = useState("8");
  const [creating, setCreating] = useState(false);
  const [sessions, setSessions] = useState<SessionBrief[] | null>(null);

  const refreshSessions = useCallback(() => {
    listSessions()
      .then(setSessions)
      .catch(() => setSessions([]));
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  async function handleCreate() {
    if (!vacancyText.trim() || creating) return;
    setCreating(true);
    try {
      const { id } = await createSession({
        vacancy_text: vacancyText.trim(),
        seniority: seniority as "junior" | "middle" | "senior" | "lead",
        language: language as "ru" | "en",
        style: style as "friendly" | "strict" | "realistic",
        planned_questions: Number(plannedQuestions),
      });
      await startSession(id);
      router.push(`/session/${id}`);
    } catch (exc) {
      setCreating(false);
      toast.error(exc instanceof ApiError ? exc.message : "Не удалось создать сессию");
    }
  }

  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 flex flex-col gap-8">
      <h1 className="text-2xl font-semibold tracking-tight">Тренажёр собеседований</h1>

      <Card>
        <CardHeader>
          <CardTitle>Новая сессия</CardTitle>
          <CardDescription>Вставьте текст вакансии — ИИ составит план интервью</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Label htmlFor="vacancy">Текст вакансии</Label>
            <Textarea
              id="vacancy"
              placeholder="Вставьте описание вакансии…"
              className="min-h-40"
              value={vacancyText}
              onChange={(event) => setVacancyText(event.target.value)}
            />
          </div>
          <div className="grid gap-4 sm:grid-cols-2">
            <div className="flex flex-col gap-2">
              <Label>Грейд</Label>
              <Select value={seniority} onValueChange={setSeniority}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {Object.entries(SENIORITY_LABELS).map(([value, label]) => (
                    <SelectItem key={value} value={value}>
                      {label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Язык интервью</Label>
              <Select value={language} onValueChange={setLanguage}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="ru">Русский</SelectItem>
                  <SelectItem value="en">English</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="flex flex-col gap-2">
              <Label>Число вопросов</Label>
              <Select value={plannedQuestions} onValueChange={setPlannedQuestions}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["5", "8", "12"].map((value) => (
                    <SelectItem key={value} value={value}>
                      {value}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Стиль интервьюера</Label>
            <RadioGroup value={style} onValueChange={setStyle} className="flex flex-wrap gap-4">
              {Object.entries(STYLE_LABELS).map(([value, label]) => (
                <div key={value} className="flex items-center gap-2">
                  <RadioGroupItem value={value} id={`style-${value}`} />
                  <Label htmlFor={`style-${value}`} className="font-normal">
                    {label}
                  </Label>
                </div>
              ))}
            </RadioGroup>
          </div>
          <Button onClick={() => void handleCreate()} disabled={!vacancyText.trim() || creating}>
            {creating ? "Готовим интервью…" : "Начать собеседование"}
          </Button>
        </CardContent>
      </Card>

      <Tabs defaultValue="history">
        <TabsList>
          <TabsTrigger value="history">История</TabsTrigger>
          <TabsTrigger value="progress">Прогресс</TabsTrigger>
        </TabsList>
        <TabsContent value="history">
          {sessions === null ? (
            <p className="text-sm text-muted-foreground">Загрузка…</p>
          ) : sessions.length === 0 ? (
            <p className="text-sm text-muted-foreground">Сессий пока нет.</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Дата</TableHead>
                  <TableHead>Позиция</TableHead>
                  <TableHead>Статус</TableHead>
                  <TableHead className="text-right">Балл</TableHead>
                  <TableHead />
                </TableRow>
              </TableHeader>
              <TableBody>
                {sessions.map((session) => {
                  const status = STATUS_LABELS[session.status] ?? {
                    label: session.status,
                    variant: "outline" as const,
                  };
                  return (
                    <TableRow key={session.id}>
                      <TableCell>{new Date(session.created_at).toLocaleString("ru-RU")}</TableCell>
                      <TableCell>{session.position_title}</TableCell>
                      <TableCell>
                        <Badge variant={status.variant}>{status.label}</Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        {session.overall_score ?? "—"}
                      </TableCell>
                      <TableCell className="text-right">
                        {session.status === "completed" && (
                          <Button variant="link" className="h-auto p-0" asChild>
                            <a href={`/session/${session.id}/report`}>Отчёт</a>
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          )}
        </TabsContent>
        <TabsContent value="progress">
          <ProgressView />
        </TabsContent>
      </Tabs>
    </main>
  );
}
