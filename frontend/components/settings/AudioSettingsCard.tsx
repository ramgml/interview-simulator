"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  applyAudioContextSink,
  getStoredDeviceId,
  inputConstraints,
  listAudioDevices,
  setStoredDeviceId,
} from "@/lib/audio";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

/** Опция «По умолчанию» в Select = пустой deviceId в localStorage. */
const DEFAULT_VALUE = "default";

/**
 * Карточка «Аудио» на /settings: выбор микрофона и устройства вывода
 * (enumerateDevices, сохранение в localStorage сразу при изменении),
 * проверка микрофона (getUserMedia → AnalyserNode → живой уровень + вердикт)
 * и проверка звука (осциллятор AudioContext → setSinkId выбранного вывода).
 */
export default function AudioSettingsCard() {
  const [inputs, setInputs] = useState<MediaDeviceInfo[]>([]);
  const [outputs, setOutputs] = useState<MediaDeviceInfo[]>([]);
  const [inputId, setInputId] = useState<string | null>(null);
  const [outputId, setOutputId] = useState<string | null>(null);

  const [micTesting, setMicTesting] = useState(false);
  const [micLevel, setMicLevel] = useState(0);
  const [micVerdict, setMicVerdict] = useState<string | null>(null);
  const [soundTesting, setSoundTesting] = useState(false);

  const testRafRef = useRef<number | null>(null);
  const testTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const testCtxRef = useRef<AudioContext | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      const devices = await listAudioDevices();
      setInputs(devices.inputs);
      setOutputs(devices.outputs);
    } catch {
      // enumerateDevices недоступен — оставляем пустые списки.
    }
  }, []);

  useEffect(() => {
    setInputId(getStoredDeviceId("input"));
    setOutputId(getStoredDeviceId("output"));
    void refreshDevices();
    return () => {
      if (testRafRef.current !== null) cancelAnimationFrame(testRafRef.current);
      if (testTimeoutRef.current) clearTimeout(testTimeoutRef.current);
      void testCtxRef.current?.close().catch(() => {});
    };
  }, [refreshDevices]);

  /**
   * Ручное обновление списка: listAudioDevices при пустых label сам
   * перезапросит getUserMedia (браузер покажет/не покажет запрос разрешения).
   */
  function handleRefreshDevices() {
    void refreshDevices();
  }

  /** Кнопка «Проверить микрофон»: уровень с выбранного входа, вердикт через ~4 сек. */
  async function handleTestMic() {
    setMicTesting(true);
    setMicVerdict(null);
    setMicLevel(0);
    let stream: MediaStream | null = null;
    let ctx: AudioContext | null = null;
    let peak = 0;
    const finish = () => {
      stopMicTest(stream, ctx);
      setMicTesting(false);
      setMicVerdict(
        peak > 0.05
          ? "Микрофон работает"
          : "Микрофон молчит — проверьте выбранное устройство",
      );
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: inputConstraints(inputId) });
      ctx = new AudioContext();
      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 1024;
      source.connect(analyser);
      const buffer = new Uint8Array(analyser.fftSize);
      const tick = () => {
        analyser.getByteTimeDomainData(buffer);
        let sumSquares = 0;
        for (let i = 0; i < buffer.length; i++) {
          const centered = buffer[i] - 128;
          sumSquares += centered * centered;
        }
        const rms = Math.sqrt(sumSquares / buffer.length) / 128; // 0..1
        peak = Math.max(peak, rms);
        setMicLevel(Math.round(Math.min(1, rms * 4) * 100));
        testRafRef.current = requestAnimationFrame(tick);
      };
      tick();
      testTimeoutRef.current = setTimeout(finish, 4000);
    } catch (err) {
      stopMicTest(stream, ctx);
      setMicTesting(false);
      setMicVerdict(describeMicError(err));
    }
  }

  /** Полная остановка теста: RAF, таймер, треки стрима, AudioContext. */
  function stopMicTest(stream: MediaStream | null, ctx: AudioContext | null) {
    if (testRafRef.current !== null) {
      cancelAnimationFrame(testRafRef.current);
      testRafRef.current = null;
    }
    if (testTimeoutRef.current) {
      clearTimeout(testTimeoutRef.current);
      testTimeoutRef.current = null;
    }
    stream?.getTracks().forEach((track) => track.stop());
    void ctx?.close().catch(() => {});
  }

  function describeMicError(err: unknown): string {
    if (err instanceof DOMException) {
      if (err.name === "NotAllowedError")
        return "Доступ к микрофону запрещён — разрешите доступ в настройках браузера";
      if (err.name === "NotFoundError" || err.name === "OverconstrainedError")
        return "Выбранное устройство недоступно";
    }
    return "Микрофон недоступен — проверьте выбранное устройство";
  }
  /** Кнопка «Проверить звук»: тон 1 кГц ~1 сек через выбранный вывод, без бэкенда. */
  async function handleTestSound() {
    setSoundTesting(true);
    try {
      const ctx = new AudioContext();
      testCtxRef.current = ctx;
      await applyAudioContextSink(ctx, outputId);
      const oscillator = ctx.createOscillator();
      const gain = ctx.createGain();
      oscillator.frequency.value = 1000;
      gain.gain.setValueAtTime(0.1, ctx.currentTime);
      gain.gain.setValueAtTime(0.1, ctx.currentTime + 0.9);
      gain.gain.linearRampToValueAtTime(0.0001, ctx.currentTime + 1);
      oscillator.connect(gain).connect(ctx.destination);
      oscillator.start();
      oscillator.stop(ctx.currentTime + 1);
      await new Promise((resolve) => setTimeout(resolve, 1100));
      void ctx.close().catch(() => {});
    } finally {
      testCtxRef.current = null;
      setSoundTesting(false);
    }
  }

  function handleInputChange(id: string) {
    const deviceId = id === DEFAULT_VALUE ? null : id;
    setInputId(deviceId);
    setStoredDeviceId("input", deviceId);
  }

  function handleOutputChange(id: string) {
    const deviceId = id === DEFAULT_VALUE ? null : id;
    setOutputId(deviceId);
    setStoredDeviceId("output", deviceId);
  }

  const labelsMissing =
    inputs.length > 0 && inputs.every((device) => !device.label);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Аудио</CardTitle>
        <CardDescription>
          Микрофон и устройство вывода для голосового цикла; выбор сохраняется сразу
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-col gap-2">
          <Label>Микрофон</Label>
          <Select value={inputId ?? DEFAULT_VALUE} onValueChange={handleInputChange}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_VALUE}>По умолчанию</SelectItem>
              {inputs.map((device, index) => (
                <SelectItem key={device.deviceId} value={device.deviceId}>
                  {device.label || `Устройство ${index + 1}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex flex-col gap-2">
          <Label>Устройство вывода</Label>
          <Select value={outputId ?? DEFAULT_VALUE} onValueChange={handleOutputChange}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={DEFAULT_VALUE}>По умолчанию</SelectItem>
              {outputs.map((device, index) => (
                <SelectItem key={device.deviceId} value={device.deviceId}>
                  {device.label || `Устройство ${index + 1}`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {labelsMissing && (
          <Button
            type="button"
            variant="link"
            className="self-start h-auto p-0"
            onClick={handleRefreshDevices}
          >
            Обновить список устройств
          </Button>
        )}

        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleTestMic()}
            disabled={micTesting}
          >
            {micTesting ? "Проверяем…" : "Проверить микрофон"}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => void handleTestSound()}
            disabled={soundTesting}
          >
            {soundTesting ? "Проверяем…" : "Проверить звук"}
          </Button>
        </div>

        {micTesting && (
          <div className="flex flex-col gap-2">
            <Progress value={micLevel} />
            <span className="text-xs text-muted-foreground">
              Скажите что-нибудь — идёт измерение уровня…
            </span>
          </div>
        )}
        {micVerdict && <p className="text-sm text-muted-foreground">{micVerdict}</p>}
      </CardContent>
    </Card>
  );
}
