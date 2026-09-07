/**
 * Общий модуль выбора аудио-устройств (микрофон, устройство вывода).
 *
 * Выбранные устройства хранятся в localStorage:
 *   - `audio-input-device` — deviceId микрофона (audioinput);
 *   - `audio-output-device` — deviceId устройства вывода (audiooutput).
 * Отсутствие ключа (или null) означает «устройство по умолчанию».
 *
 * Используется: карточка «Аудио» на /settings, Recorder (ввод), AudioQueue (вывод).
 */

export type AudioDeviceKind = "input" | "output";

const STORAGE_KEYS: Record<AudioDeviceKind, string> = {
  input: "audio-input-device",
  output: "audio-output-device",
};

export function getStoredDeviceId(kind: AudioDeviceKind): string | null {
  try {
    return localStorage.getItem(STORAGE_KEYS[kind]);
  } catch {
    // localStorage недоступен (приватный режим) — считаем выбор пустым.
    return null;
  }
}

/** Сохраняет выбор немедленно (без кнопки «Сохранить»); null = устройство по умолчанию. */
export function setStoredDeviceId(kind: AudioDeviceKind, id: string | null): void {
  try {
    if (id) localStorage.setItem(STORAGE_KEYS[kind], id);
    else localStorage.removeItem(STORAGE_KEYS[kind]);
  } catch {
    // localStorage недоступен — выбор просто не сохранится, работа не ломается.
  }
}

/**
 * Список устройств ввода/вывода. В Chromium до выдачи разрешения
 * enumerateDevices() возвращает пустые label, поэтому при необходимости
 * один раз запрашиваем getUserMedia({ audio: true }) (треки сразу stop)
 * и перечисляем повторно.
 */
export async function listAudioDevices(): Promise<{
  inputs: MediaDeviceInfo[];
  outputs: MediaDeviceInfo[];
}> {
  if (!navigator.mediaDevices?.enumerateDevices) {
    return { inputs: [], outputs: [] };
  }
  let devices = await navigator.mediaDevices.enumerateDevices();
  const hasUnlabeled = devices.some(
    (device) => (device.kind === "audioinput" || device.kind === "audiooutput") && !device.label,
  );
  if (hasUnlabeled) {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      stream.getTracks().forEach((track) => track.stop());
      devices = await navigator.mediaDevices.enumerateDevices();
    } catch {
      // Разрешение не выдано — вернём список как есть (с пустыми label).
    }
  }
  // В Chromium «устройство по умолчанию» приходит с deviceId="default" — это
  // алиас реального устройства, а в наших Select value="default" уже занят
  // опцией «По умолчанию». Дубликаты value ломают выбор, alias не возвращаем:
  // он дублирует физическое устройство из списка ниже.
  return {
    inputs: devices.filter(
      (device) => device.kind === "audioinput" && device.deviceId && device.deviceId !== "default",
    ),
    outputs: devices.filter(
      (device) => device.kind === "audiooutput" && device.deviceId && device.deviceId !== "default",
    ),
  };
}

/** Constraints для getUserMedia: точный deviceId выбранного микрофона или устройство по умолчанию. */
export function inputConstraints(deviceId: string | null): true | { deviceId: { exact: string } } {
  return deviceId ? { deviceId: { exact: deviceId } } : true;
}

/**
 * Безопасный вывод звука в выбранное устройство: setSinkId есть не везде
 * (Safari) — тогда тихо играем через default; падение setSinkId (устройство
 * исчезло/не поддерживается) не должно прерывать воспроизведение — глотаем
 * с warning в консоли.
 */
export async function applyOutputDevice(
  media: HTMLMediaElement,
  deviceId?: string | null,
): Promise<void> {
  const id = deviceId === undefined ? getStoredDeviceId("output") : deviceId;
  if (!id) return;
  if (typeof media.setSinkId !== "function") {
    console.info("setSinkId не поддерживается этим браузером — звук пойдёт в устройство по умолчанию");
    return;
  }
  try {
    await media.setSinkId(id);
  } catch {
    // Выбранный вывод недоступен — не роняем воспроизведение, играем через default.
    console.warn("setSinkId не удался — звук пойдёт в устройство по умолчанию");
  }
}

/**
 * То же для AudioContext (метод есть в Chromium, отсутствует в стандарте и
 * в DOM-lib TypeScript): карст в AudioContextWithSetSinkId, вызов под runtime-проверкой.
 */
interface AudioContextWithSetSinkId {
  setSinkId(deviceId: string): Promise<void>;
}

export async function applyAudioContextSink(
  ctx: AudioContext,
  deviceId?: string | null,
): Promise<void> {
  const id = deviceId === undefined ? getStoredDeviceId("output") : deviceId;
  if (!id || typeof (ctx as unknown as AudioContextWithSetSinkId).setSinkId !== "function") return;
  try {
    await (ctx as unknown as AudioContextWithSetSinkId).setSinkId(id);
  } catch {
    // Вывод недоступен — сигнал уйдёт в устройство по умолчанию.
    console.warn("setSinkId не удался для AudioContext — сигнал пойдёт в default");
  }
}
