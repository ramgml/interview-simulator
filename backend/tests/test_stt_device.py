"""VRAM fallback: при занятой GPU (resident-модель держит VRAM) и WHISPER_DEVICE=auto
быстрая проверка _resolve_device → cpu из-за исчерпания свободной памяти не тестируется
юнит-тестом (нужен реальный OOM) — fallback-ветка покрыта кодом + PM-прогоном.
"""

from app.stt import _resolve_device


def test_resolve_device_cpu_explicit():
    assert _resolve_device("cpu") == ("cpu", "int8")


def test_resolve_device_cuda_explicit():
    assert _resolve_device("cuda") == ("cuda", "int8_float16")


def test_resolve_device_auto_without_ct2_cuda(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_ctranslate2(name, *args, **kwargs):
        if name == "ctranslate2":
            raise ImportError("no cuda stack")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ctranslate2)
    assert _resolve_device("auto") == ("cpu", "int8")
