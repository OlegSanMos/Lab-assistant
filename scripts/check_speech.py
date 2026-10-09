from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmentor import speech
from labmentor.config import Settings, use_system_certificates

PHRASE = "Какова цель лабораторной работы номер один?"
WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json"]


def download_whisper(size: str) -> None:
    if speech.model_cached(size):
        print(f"Модель Whisper {size} уже скачана")
        return
    from faster_whisper.utils import _MODELS
    from huggingface_hub import snapshot_download

    print(f"Скачивается модель Whisper {size} ({speech._model_mb(size)}), это нужно один раз")
    snapshot_download(_MODELS.get(size, size), allow_patterns=[*WHISPER_FILES, "vocabulary.*"])


def versions() -> str:
    from importlib.metadata import PackageNotFoundError, version

    found = []
    for package in ("faster-whisper", "av", "ctranslate2"):
        try:
            found.append(f"{package} {version(package)}")
        except PackageNotFoundError:
            found.append(f"{package} не установлен")
    return ", ".join(found)


def check_tts(voice: str, out_dir: Path) -> Path | None:
    working = None
    for name, engine, ext in speech.tts_engines():
        path = out_dir / f"check_{name}.{ext}"
        started = time.monotonic()
        error = speech.try_engine(engine, PHRASE, voice, path)
        took = time.monotonic() - started
        status = "OK " if error is None else "ERR"
        print(f"  {status} {speech.TTS_NAMES[name]} ({took:.1f} с){': ' + error if error else ''}")
        if error is None and working is None:
            working = path
    return working


def main() -> None:
    use_system_certificates()
    settings = Settings.from_env()
    out_dir = Path(tempfile.mkdtemp(prefix="labmentor_speech_"))
    print("Озвучивание:")
    audio = check_tts(settings.tts_voice, out_dir)
    print(f"\nРаспознавание: {' -> '.join(speech.stt_engines(settings))} ({versions()})")
    ok = audio is not None
    if "local" in speech.stt_engines(settings):
        try:
            download_whisper(settings.whisper_size)
            speech.load_whisper(settings)
            print(f"  OK  модель Whisper {settings.whisper_size} загружена")
        except speech.SpeechError as exc:
            sys.exit(f"  ERR {exc}\n\nИсправьте ошибку и запустите проверку снова")
        except Exception as exc:
            message = speech._load_error(settings.whisper_size, exc)
            sys.exit(f"  ERR {message}\n\nИсправьте ошибку и запустите проверку снова")
    if audio is not None:
        try:
            text = speech.transcribe(audio, settings)
            print(f"  OK  распознано: «{text}» (озвучено: «{PHRASE}»)")
        except speech.SpeechError as exc:
            ok = False
            print(f"  ERR {exc}")
    print(f"\nФайлы проверки: {out_dir}")
    print("Речевой модуль готов к работе" if ok else "Исправьте ошибки выше и запустите снова")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
