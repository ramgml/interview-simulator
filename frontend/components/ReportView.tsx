"use client";

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { Report, TurnFeedback } from "@/lib/api";

const HIRE_LABELS: Record<string, { label: string; variant: "default" | "secondary" | "destructive" | "outline" }> = {
  strong_yes: { label: "Однозначно нанять", variant: "default" },
  yes: { label: "Нанять", variant: "default" },
  no: { label: "Не нанять", variant: "secondary" },
  strong_no: { label: "Точно не нанять", variant: "destructive" },
};

function FeedbackBlocks({ item }: { item: TurnFeedback }) {
  const rows = [
    { label: "Что хорошо", value: item.good },
    { label: "Что упущено", value: item.missed },
  ].filter((row) => row.value);
  if (rows.length === 0 && !item.strong_answer) return null;
  return (
    <div className="flex flex-col gap-2 mt-2">
      {rows.map((row) => (
        <p key={row.label} className="text-sm">
          <span className="font-medium">{row.label}: </span>
          {row.value}
        </p>
      ))}
      {item.strong_answer && (
        <Badge className="w-fit" variant="secondary">
          Сильный ответ
        </Badge>
      )}
    </div>
  );
}

/** Разбор отчёта: общий балл, компетенции, turn_feedback, сильные/слабые, план, вердикт. */
export default function ReportView({ report }: { report: Report }) {
  const hire = report.hire_recommendation ? HIRE_LABELS[report.hire_recommendation] : undefined;

  return (
    <div className="flex flex-col gap-6">
      {report.degraded && (
        <Alert>
          <AlertTitle>Отчёт сформирован в упрощённом виде</AlertTitle>
          <AlertDescription>
            Оценщик вернул неструктурированный ответ — баллы и разбор могут быть неполными.
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center justify-between gap-4">
            <span>Общий балл</span>
            <span className="text-3xl font-bold tabular-nums">
              {report.overall_score ?? "—"}/10
            </span>
          </CardTitle>
          {report.overall_score !== null && (
            <Progress value={report.overall_score * 10} className="h-3" />
          )}
        </CardHeader>
        <CardContent className="flex flex-wrap items-center gap-2">
          {hire && <Badge variant={hire.variant}>{hire.label}</Badge>}
          {report.verdict && <p className="text-sm">{report.verdict}</p>}
        </CardContent>
      </Card>

      {report.competencies.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Компетенции</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {report.competencies.map((comp) => (
              <div key={comp.name} className="flex flex-col gap-1">
                <div className="flex items-center gap-3">
                  <span className="w-48 shrink-0 text-sm font-medium">{comp.name}</span>
                  <Progress value={(comp.score ?? 0) * 10} className="flex-1" />
                  <span className="w-10 text-right text-sm tabular-nums">
                    {comp.score ?? "—"}
                  </span>
                </div>
                {comp.comment && <p className="text-sm text-muted-foreground">{comp.comment}</p>}
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {report.turn_feedback.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Разбор ответов</CardTitle>
            <CardDescription>Вопрос → ваш ответ → что хорошо и что упущено</CardDescription>
          </CardHeader>
          <CardContent>
            <Accordion type="single" collapsible>
              {report.turn_feedback.map((item, index) => (
                <AccordionItem key={item.turn_idx} value={`turn-${item.turn_idx}`}>
                  <AccordionTrigger>
                    <span className="text-left">
                      {index + 1}. {item.question ?? "Вопрос"}
                      {item.score !== null && ` — ${item.score}/10`}
                    </span>
                  </AccordionTrigger>
                  <AccordionContent className="flex flex-col gap-2">
                    {item.answer && (
                      <p className="text-sm whitespace-pre-wrap">
                        <span className="font-medium">Ответ: </span>
                        {item.answer}
                      </p>
                    )}
                    <FeedbackBlocks item={item} />
                  </AccordionContent>
                </AccordionItem>
              ))}
            </Accordion>
          </CardContent>
        </Card>
      )}

      {(report.strengths.length > 0 || report.weaknesses.length > 0) && (
        <div className="grid gap-4 sm:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Сильные стороны</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {report.strengths.length > 0 ? (
                report.strengths.map((item, index) => (
                  <Badge key={index} variant="secondary">
                    {item}
                  </Badge>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Зоны роста</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-wrap gap-2">
              {report.weaknesses.length > 0 ? (
                report.weaknesses.map((item, index) => (
                  <Badge key={index} variant="outline">
                    {item}
                  </Badge>
                ))
              ) : (
                <span className="text-sm text-muted-foreground">—</span>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {report.plan.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>План подготовки</CardTitle>
            <CardDescription>Тема → что сделать</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Тема</TableHead>
                  <TableHead>Действие</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {report.plan.map((item, index) => (
                  <TableRow key={index}>
                    <TableCell className="font-medium">{item.topic}</TableCell>
                    <TableCell>
                      {item.action}
                      {item.resources_hint && (
                        <span className="block text-sm text-muted-foreground">
                          {item.resources_hint}
                        </span>
                      )}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
