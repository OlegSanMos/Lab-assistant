from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .formulas import latex_to_text

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".opus", ".webm", ".m4a", ".aac", ".flac"}
VOICES = {
    "Светлана (женский)": "ru-RU-SvetlanaNeural",
    "Дмитрий (мужской)": "ru-RU-DmitryNeural",
}
MALE_VOICES = {"ru-RU-DmitryNeural"}
MAX_TTS_CHARS = 700
SAMPLE_RATE = 16000
TTS_TIMEOUT = 25
RETRY_AFTER = 300
MODEL_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530, "large-v3": 3090}
NO_SPEECH = (
    "Речь не распознана: в записи не слышно слов. Проверьте, что браузеру разрешен доступ к "
    "микрофону и выбран нужный микрофон, говорите громче и ближе к нему."
)
TTS_NAMES = {
    "edge": "нейросетевой голос Edge TTS",
    "gtts": "Google TTS",
    "system": "системный голос компьютера (без интернета)",
}


class SpeechError(Exception):
    pass


_models: dict[str, object] = {}
_model_lock = threading.Lock()
_model_state: dict[str, str] = {}


def stt_engines(settings: Settings) -> list[str]:
    if settings.stt_engine in ("local", "openai"):
        return [settings.stt_engine]
    api = bool(settings.openai_api_key)
    if settings.provider == "openai" and api:
        return ["openai", "local"]
    return ["local", "openai"] if api else ["local"]


def transcribe(path: str | Path, settings: Settings) -> str:
    errors = []
    for engine in stt_engines(settings):
        try:
            if engine == "openai":
                text = _transcribe_api(path, settings)
            else:
                text = _transcribe_local(path, settings)
        except SpeechError as exc:
            errors.append(str(exc))
            continue
        if not text:
            raise SpeechError(NO_SPEECH)
        return text
    raise SpeechError(" ".join(errors))


def _transcribe_local(path: str | Path, settings: Settings) -> str:
    try:
        audio = decode_audio(path)
    except Exception as exc:
        raise SpeechError(
            f"Не удалось прочитать аудиозапись {Path(path).name}: {_short(exc)}."
        ) from exc
    model = load_whisper(settings)
    try:
        segments, _ = model.transcribe(audio, language="ru", vad_filter=True)
        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        raise SpeechError(f"Ошибка распознавания речи моделью Whisper: {_short(exc)}.") from exc


def decode_audio(path: str | Path):
    import av
    import numpy as np

    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
    chunks = []
    with av.open(str(path)) as container:
        frames = container.decode(audio=0)
        while True:
            try:
                frame = next(frames)
            except StopIteration:
                frame = None
            except av.error.InvalidDataError:
                continue
            chunks += [part.to_ndarray().reshape(-1) for part in resampler.resample(frame)]
            if frame is None:
                break
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def _transcribe_api(path: str | Path, settings: Settings) -> str:
    if not settings.openai_api_key:
        raise SpeechError("Whisper API: не задан ключ OPENAI_API_KEY.")
    from openai import OpenAI

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
        timeout=60,
        max_retries=1,
    )
    try:
        with open(path, "rb") as audio:
            result = client.audio.transcriptions.create(
                model=settings.stt_model, file=audio, language="ru"
            )
    except Exception as exc:
        detail = _short(exc)
        hint = ""
        if "unsupported_country" in detail or re.search(r"\b403\b", detail):
            hint = (
                " Доступ запрещен: OpenAI не работает в вашей стране или сеть закрывает доступ."
                " Распознавание выполняется на компьютере; чтобы не обращаться к API, задайте"
                " в .env STT_ENGINE=local."
            )
        raise SpeechError(
            f"Whisper API ({settings.stt_model}) недоступен: {detail}.{hint}"
        ) from exc
    return result.text.strip()


def _whisper(size: str):
    with _model_lock:
        if size not in _models:
            from faster_whisper import WhisperModel

            _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
        return _models[size]


def load_whisper(settings: Settings):
    size = settings.whisper_size
    if _model_state.get(size) != "готова":
        _model_state[size] = "загружается"
    try:
        model = _whisper(size)
    except ImportError as exc:
        _model_state[size] = "не установлен пакет faster-whisper"
        raise SpeechError(
            "Не установлен пакет faster-whisper для распознавания речи: выполните "
            "pip install -r requirements.txt."
        ) from exc
    except Exception as exc:
        _model_state[size] = f"не загружена ({_short(exc)[:120]})"
        raise SpeechError(_load_error(size, exc)) from exc
    _model_state[size] = "готова"
    return model


def _load_error(size: str, exc: Exception) -> str:
    detail = _short(exc)
    message = f"Не удалось загрузить модель распознавания речи Whisper «{size}»: {detail}."
    if "CERTIFICATE" in detail.upper():
        message += (
            " Сеть проверяет защищенные соединения своим сертификатом (корпоративная сеть):"
            " установите пакет truststore (pip install -r requirements.txt) и перезапустите."
        )
    return message + (
        " Модель скачивается один раз с huggingface.co. Проверьте интернет и выполните "
        "python scripts/check_speech.py. Если сайт недоступен, укажите в .env зеркало "
        "HF_ENDPOINT=https://hf-mirror.com или путь к скачанной модели в WHISPER_SIZE."
    )


def model_cached(size: str) -> bool:
    if Path(size).is_dir():
        return True
    try:
        from faster_whisper.utils import download_model

        download_model(size, local_files_only=True)
    except Exception:
        return False
    return True


def preload(settings: Settings, quiet: bool = False) -> threading.Thread | None:
    if stt_engines(settings)[0] != "local":
        return None
    size = settings.whisper_size

    def run() -> None:
        note = "" if model_cached(size) else f" (скачивается один раз, {_model_mb(size)})"
        if not quiet:
            print(f"Распознавание речи: загружается модель Whisper {size}{note}...")
        try:
            load_whisper(settings)
        except SpeechError as exc:
            if not quiet:
                print(f"Внимание: {exc}")
            return
        if not quiet:
            print(f"Распознавание речи: модель Whisper {size} готова")

    thread = threading.Thread(target=run, name="whisper-preload", daemon=True)
    thread.start()
    return thread


def _model_mb(size: str) -> str:
    return f"около {MODEL_MB[size]} МБ" if size in MODEL_MB else "несколько сотен МБ"


def stt_loading(settings: Settings) -> bool:
    local = stt_engines(settings)[0] == "local"
    return local and _model_state.get(settings.whisper_size) == "загружается"


def stt_status(settings: Settings) -> str:
    engines = stt_engines(settings)
    if engines[0] == "openai":
        status = f"Whisper API ({settings.stt_model})"
    else:
        size = settings.whisper_size
        state = _model_state.get(size)
        if state == "готова":
            status = f"Whisper {size} на компьютере ✅ готова"
        elif state == "загружается":
            status = f"Whisper {size} на компьютере ⏳ загружается ({_model_mb(size)})"
        elif state:
            status = f"Whisper {size} на компьютере ❌ {state}"
        else:
            status = f"Whisper {size} на компьютере, загрузится при первом вопросе"
    if len(engines) > 1:
        status += ", резерв: " + ("Whisper API" if engines[1] == "openai" else "Whisper локально")
    return status


def _short(exc: BaseException) -> str:
    for _ in range(6):
        cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        if cause is None or not str(cause):
            break
        exc = cause
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    text = re.sub(r",?\s*url='[^']*'?", "", text)
    text = f"{type(exc).__name__}: {text}" if text else type(exc).__name__
    return text[:300]


_skip_until: dict[str, float] = {}
_tts_state = {"engine": ""}


def text_for_speech(text: str, limit: int = MAX_TTS_CHARS) -> str:
    text = latex_to_text(re.sub(r"```.*?```", " ", text, flags=re.S))
    text = re.sub(r"^\s*\|.*\|\s*$", " ", text, flags=re.M)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[[^\]]*\.(md|pdf|docx|txt)\]", " ", text)
    text = re.sub(r"[#*_`>|]|[\U0001F300-\U0001FAFF☀-➿️]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    end = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
    cut = cut[: end + 1] if end > limit // 3 else cut.rsplit(" ", 1)[0] + "..."
    return f"{cut} Подробности в чате."


def synthesize(text: str, voice: str, out_dir: Path) -> str | None:
    clean = text_for_speech(text)
    if not clean:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, engine, ext in tts_engines():
        if time.monotonic() < _skip_until.get(name, 0):
            continue
        path = out_dir / f"answer_{uuid.uuid4().hex[:8]}.{ext}"
        if try_engine(engine, clean, voice, path) is None:
            _tts_state["engine"] = name
            return str(path)
        _skip_until[name] = time.monotonic() + RETRY_AFTER
    return None


def tts_engines() -> list[tuple]:
    return [
        ("edge", _edge_tts, "mp3"),
        ("gtts", _google_tts, "mp3"),
        ("system", _system_voice, "wav"),
    ]


def tts_status() -> str:
    engine = _tts_state["engine"]
    return TTS_NAMES[engine] if engine else "голос выбирается при первом ответе"


def try_engine(engine, text: str, voice: str, path: Path) -> str | None:
    errors = []

    def run() -> None:
        try:
            engine(text, voice, path)
        except Exception as exc:
            errors.append(_short(exc))

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(TTS_TIMEOUT)
    if worker.is_alive():
        return f"нет ответа за {TTS_TIMEOUT} с"
    if errors:
        return errors[0]
    if not path.exists() or path.stat().st_size == 0:
        return "пустой аудиофайл"
    return None


def _edge_tts(text: str, voice: str, path: Path) -> None:
    import edge_tts

    edge_tts.Communicate(text, voice, connect_timeout=8, receive_timeout=20).save_sync(str(path))


def _google_tts(text: str, voice: str, path: Path) -> None:
    from gtts import gTTS

    gTTS(text, lang="ru").save(str(path))


def _system_voice(text: str, voice: str, path: Path) -> None:
    male = voice in MALE_VOICES
    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as file:
        file.write(text)
    try:
        if sys.platform == "darwin":
            name = _mac_voice(male)
            command = ["say", "-v", name, "-o", str(path), "--file-format=WAVE"]
            command.append("--data-format=LEI16@22050")
            subprocess.run([*command, "-f", file.name], check=True, timeout=TTS_TIMEOUT)
        elif sys.platform == "win32":
            env = {**os.environ, "LM_TEXT": file.name, "LM_WAV": str(path)}
            command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_TTS]
            subprocess.run(command, check=True, timeout=TTS_TIMEOUT, env=env)
        else:
            espeak = shutil.which("espeak-ng") or shutil.which("espeak")
            if not espeak:
                raise RuntimeError("не найден espeak-ng")
            name = "ru" if male else "ru+f3"
            command = [espeak, "-v", name, "-s", "165", "-w", str(path), "-f", file.name]
            subprocess.run(command, check=True, timeout=TTS_TIMEOUT, capture_output=True)
    finally:
        os.unlink(file.name)


@lru_cache(maxsize=2)
def _mac_voice(male: bool) -> str:
    listing = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10)
    names = re.findall(r"^(.+?)\s+ru_RU\s", listing.stdout, flags=re.M)
    if not names:
        raise RuntimeError("в системе нет русского голоса")
    preferred = [n for n in names if (n.split()[0] in ("Yuri", "Юрий")) == male]
    return (preferred or names)[0]


WINDOWS_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ru-RU' } "
    "| Select-Object -First 1; if (-not $v) { exit 2 }; "
    "$s.SelectVoice($v.VoiceInfo.Name); $s.SetOutputToWaveFile($env:LM_WAV); "
    "$s.Speak([IO.File]::ReadAllText($env:LM_TEXT, [Text.Encoding]::UTF8)); $s.Dispose()"
)
