"use client";

import { useEffect, useState } from "react";
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Separator } from "@/components/ui/separator";
import { toast } from "sonner";
import { getSettings, updateSettings, testSettings, ApiError } from "@/lib/api";

const MODELS = ["glm/glm-5.3-flash", "glm/glm-5.3", "auto/best-coding"];
const WHISPER_MODELS = ["large-v3-turbo", "large-v3", "medium", "small"];
const TTS_VOICES = ["aidar", "baya", "kseniya", "xenia", "eugene", "random"];

/** Настройки провайдера LLM и голоса: GET/PUT /api/settings + GET /api/settings/test. */
export default function SettingsPage() {
  const [provider, setProvider] = useState<"local" | "cloud">("local");
  const [baseUrl, setBaseUrl] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [maskedKey, setMaskedKey] = useState(false);
  const [model, setModel] = useState("");
  const [whisperModel, setWhisperModel] = useState("");
  const [ttsVoice, setTtsVoice] = useState("");
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);

  useEffect(() => {
    getSettings()
      .then((settings) => {
        setProvider(settings.provider);
        setBaseUrl(settings.base_url);
        setModel(settings.model);
        setWhisperModel(settings.whisper_model);
        setTtsVoice(settings.tts_voice);
        setMaskedKey(settings.api_key !== null);
      })
      .catch((exc: unknown) =>
        toast.error(exc instanceof ApiError ? exc.message : "Не удалось загрузить настройки"),
      )
      .finally(() => setLoading(false));
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      // api_key: '***' (маска с GET) или пустая строка — бэкенд хранимый ключ не перетирает;
      // отправляем поле только если пользователь что-то ввёл.
      const stored = apiKey.trim();
      const saved = await updateSettings({
        provider,
        base_url: baseUrl.trim(),
        ...(stored && stored !== "***" ? { api_key: stored } : {}),
        model: model.trim(),
        whisper_model: whisperModel,
        tts_voice: ttsVoice,
      });
      setMaskedKey(saved.api_key !== null);
      setApiKey("");
      toast.success("Настройки сохранены");
    } catch (exc) {
      toast.error(exc instanceof ApiError ? exc.message : "Не удалось сохранить настройки");
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    try {
      await testSettings();
      toast.success("Соединение установлено");
    } catch (exc) {
      toast.error(exc instanceof ApiError ? exc.message : "Проверка не удалась");
    } finally {
      setTesting(false);
    }
  }

  if (loading) {
    return (
      <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-8">
        <p className="text-sm text-muted-foreground">Загрузка настроек…</p>
      </main>
    );
  }

  return (
    <main className="mx-auto w-full max-w-2xl flex-1 px-4 py-8 flex flex-col gap-6">
      <div className="flex items-center justify-between gap-4">
        <h1 className="text-2xl font-semibold tracking-tight">Настройки</h1>
        <Button variant="outline" asChild>
          <a href="/">На главную</a>
        </Button>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Провайдер LLM</CardTitle>
          <CardDescription>
            Локальный — без ключа; облачный — требует base_url, api_key и model
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <RadioGroup
            value={provider}
            onValueChange={(value) => setProvider(value as "local" | "cloud")}
            className="flex gap-6"
          >
            <div className="flex items-center gap-2">
              <RadioGroupItem value="local" id="provider-local" />
              <Label htmlFor="provider-local" className="font-normal">
                Локальный
              </Label>
            </div>
            <div className="flex items-center gap-2">
              <RadioGroupItem value="cloud" id="provider-cloud" />
              <Label htmlFor="provider-cloud" className="font-normal">
                Облачный
              </Label>
            </div>
          </RadioGroup>

          <div className="flex flex-col gap-2">
            <Label htmlFor="base-url">Base URL</Label>
            <Input
              id="base-url"
              placeholder="https://api.example.com/v1"
              value={baseUrl}
              onChange={(event) => setBaseUrl(event.target.value)}
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="api-key">API-ключ</Label>
            <Input
              id="api-key"
              type="password"
              placeholder={maskedKey ? "*** — не менять" : "Ключ не задан"}
              value={apiKey}
              onChange={(event) => setApiKey(event.target.value)}
            />
          </div>

          <div className="flex flex-col gap-2">
            <Label htmlFor="model">Модель</Label>
            <Input
              id="model"
              list="model-options"
              placeholder={MODELS[0]}
              value={model}
              onChange={(event) => setModel(event.target.value)}
            />
            <datalist id="model-options">
              {MODELS.map((option) => (
                <option key={option} value={option} />
              ))}
            </datalist>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Голос</CardTitle>
          <CardDescription>Распознавание речи и синтез голоса интервьюера</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <Label>Модель Whisper</Label>
            <Select value={whisperModel} onValueChange={setWhisperModel}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {WHISPER_MODELS.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-2">
            <Label>Голос озвучки</Label>
            <Select value={ttsVoice} onValueChange={setTtsVoice}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {TTS_VOICES.map((option) => (
                  <SelectItem key={option} value={option}>
                    {option}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      <Separator />

      <div className="flex flex-wrap gap-3">
        <Button onClick={() => void handleSave()} disabled={saving}>
          {saving ? "Сохраняем…" : "Сохранить"}
        </Button>
        <Button variant="outline" onClick={() => void handleTest()} disabled={testing}>
          {testing ? "Проверяем…" : "Проверить соединение"}
        </Button>
      </div>
    </main>
  );
}
