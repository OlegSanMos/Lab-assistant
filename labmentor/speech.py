from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import wave
from functools import lru_cache
from pathlib import Path

from .config import Settings
from .formulas import latex_to_text

AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".oga", ".opus", ".webm", ".m4a", ".aac", ".flac"}
VOICES = {
    "Женский": "ru-RU-SvetlanaNeural",
    "Мужской": "ru-RU-DmitryNeural",
}
MALE_VOICES = {"ru-RU-DmitryNeural"}
PIPER_VOICES = {True: "ru_RU-dmitri-medium", False: "ru_RU-irina-medium"}
PIPER_MD5 = {
    "ru_RU-dmitri-medium": "589ccc91745a1e2353508ff62c5941b7",
    "ru_RU-irina-medium": "21fbe77fdc68bdc35d7adb6bf4f52199",
}
PIPER_MB = 63
PIPER_HOSTS = ("https://huggingface.co", "https://hf-mirror.com")
PIPER_WAIT = 60
VOICES_DIR = Path.home() / ".cache" / "labmentor" / "voices"
MAX_TTS_CHARS = 700
SAMPLE_RATE = 16000
TTS_TIMEOUT = 25
RETRY_AFTER = 60
MAX_RETRY_AFTER = 1800
MALE_HINTS = {
    "darwin": "в macOS установите русский голос Yuri: Системные настройки, Универсальный доступ, "
    "раздел озвучивания текста (Spoken Content), Системный голос, Управлять голосами",
    "win32": "в Windows установите русский голос Pavel: Параметры, Время и язык, Речь",
}
MODEL_MB = {"tiny": 75, "base": 145, "small": 485, "medium": 1530, "large-v3": 3090}
NO_SPEECH = (
    "Речь не распознана: в записи не слышно слов. Проверьте, что браузеру разрешен доступ к "
    "микрофону и выбран нужный микрофон, говорите громче и ближе к нему."
)
TTS_NAMES = {
    "edge": "нейросетевой голос Edge TTS",
    "gtts": "Google TTS",
    "piper": "нейросетевой голос Piper (без интернета)",
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
_tts_failures: dict[str, int] = {}
_tts_errors: dict[str, str] = {}
_tts_state = {"engine": "", "voice": ""}
_piper: dict[str, object] = {}
_piper_state: dict[str, str] = {}
_piper_threads: dict[str, threading.Thread] = {}
_piper_lock = threading.Lock()
_piper_speak_lock = threading.Lock()


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
    _tts_state["voice"] = voice
    for name, engine, ext in tts_engines(voice):
        key = f"{name}:{voice}"
        if time.monotonic() < _skip_until.get(key, 0):
            continue
        loading = preload_piper(voice in MALE_VOICES) if name == "piper" else None
        if loading is not None:
            loading.join(PIPER_WAIT)
        path = out_dir / f"answer_{uuid.uuid4().hex[:8]}.{ext}"
        error = try_engine(engine, clean, voice, path)
        if error is None:
            _tts_state["engine"] = name
            _tts_errors.pop(key, None)
            _tts_failures.pop(key, None)
            return str(path)
        _tts_errors[key] = error
        _tts_failures[key] = _tts_failures.get(key, 0) + 1
        delay = RETRY_AFTER * 4 ** (_tts_failures[key] - 1)
        _skip_until[key] = time.monotonic() + min(delay, MAX_RETRY_AFTER)
    _tts_state["engine"] = ""
    return None


def tts_engines(voice: str = "") -> list[tuple]:
    edge = ("edge", _edge_tts, "mp3")
    google = ("gtts", _google_tts, "mp3")
    piper = ("piper", _piper_tts, "wav")
    system = ("system", _system_voice, "wav")
    if voice in MALE_VOICES:
        return [edge, piper, system, google]
    return [edge, google, piper, system]


def piper_files(name: str) -> tuple[Path, Path]:
    model = VOICES_DIR / f"{name}.onnx"
    return model, model.with_name(f"{name}.onnx.json")


def piper_urls(name: str, extension: str) -> list[str]:
    language, speaker, quality = name.split("-")
    folder = f"{language.split('_')[0]}/{language}/{speaker}/{quality}"
    hosts = [os.getenv("HF_ENDPOINT", ""), *PIPER_HOSTS]
    urls = [
        f"{host.rstrip('/')}/rhasspy/piper-voices/resolve/main/{folder}/{name}{extension}"
        for host in hosts
        if host
    ]
    return list(dict.fromkeys(urls))


def _intact(path: Path, digest: str | None = None) -> bool:
    if not path.exists() or not path.stat().st_size:
        return False
    if digest is None:
        return True
    md5 = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            md5.update(block)
    return md5.hexdigest() == digest


def download_piper(name: str) -> None:
    model, config = piper_files(name)
    for path, extension, digest in (
        (model, ".onnx", PIPER_MD5.get(name)),
        (config, ".onnx.json", None),
    ):
        if _intact(path, digest):
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(f"{path.name}.part")
        error = ""
        for url in piper_urls(name, extension):
            request = urllib.request.Request(url, headers={"User-Agent": "LabMentor"})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    with open(part, "wb") as file:
                        shutil.copyfileobj(response, file)
            except OSError as exc:
                error = _short(exc)
                continue
            if _intact(part, digest):
                part.replace(path)
                break
            error = f"файл {path.name} скачался не полностью или поврежден"
        else:
            part.unlink(missing_ok=True)
            raise RuntimeError(error)


def piper_cached(male: bool) -> bool:
    return all(_intact(path) for path in piper_files(PIPER_VOICES[male]))


def preload_piper(male: bool) -> threading.Thread | None:
    name = PIPER_VOICES[male]
    with _piper_lock:
        thread = _piper_threads.get(name)
        if name not in _piper and (thread is None or not thread.is_alive()):
            thread = threading.Thread(target=_load_piper, args=(name,), daemon=True)
            _piper_threads[name] = thread
            thread.start()
    return thread


def _load_piper(name: str) -> None:
    _piper_state[name] = "загружается"
    try:
        download_piper(name)
        from piper import PiperVoice

        model, config = piper_files(name)
        _piper[name] = PiperVoice.load(model, config, download_dir=VOICES_DIR)
    except ImportError:
        _piper_state[name] = "не установлен пакет piper-tts (pip install -r requirements.txt)"
        return
    except Exception as exc:
        _piper_state[name] = f"не загружен ({_short(exc)})"
        return
    _piper_state[name] = "готов"


def piper_loading() -> bool:
    return "загружается" in _piper_state.values()


def _piper_tts(text: str, voice: str, path: Path) -> None:
    name = PIPER_VOICES[voice in MALE_VOICES]
    if name not in _piper:
        raise RuntimeError(f"голос {_piper_state.get(name, 'не загружен')}")
    with _piper_speak_lock, wave.open(str(path), "wb") as wav:
        _piper[name].synthesize_wav(text, wav)


def male_voice_missing() -> bool:
    return _tts_state["voice"] in MALE_VOICES and _tts_state["engine"] == "gtts"


def _reasons(voice: str, names: tuple[str, ...]) -> str:
    return "; ".join(
        f"{TTS_NAMES[name]}: {_tts_errors[f'{name}:{voice}']}"
        for name in names
        if f"{name}:{voice}" in _tts_errors
    )


def voice_note() -> str:
    if not male_voice_missing():
        return ""
    if _piper_state.get(PIPER_VOICES[True]) == "загружается":
        return (
            f"Мужской голос Piper еще загружается (около {PIPER_MB} МБ, только при первом "
            "запуске), поэтому этот ответ озвучен женским голосом."
        )
    reasons = _reasons(_tts_state["voice"], ("edge", "piper", "system"))
    hint = MALE_HINTS.get(sys.platform, "установите espeak-ng")
    return (
        f"Мужской голос сейчас недоступен ({reasons}), ответ озвучен женским голосом. Мужской "
        f"голос Piper работает без интернета, но один раз скачивается с huggingface.co (около "
        f"{PIPER_MB} МБ): проверьте интернет или укажите в .env зеркало "
        f"HF_ENDPOINT=https://hf-mirror.com. Еще один вариант: {hint}."
    )


def tts_status() -> str:
    engine, voice = _tts_state["engine"], _tts_state["voice"]
    note = voice_note()
    if not voice:
        status = "голос выбирается при первом ответе"
    elif not engine:
        status = f"недоступно ({_reasons(voice, tuple(TTS_NAMES))})"
    else:
        male = voice in MALE_VOICES and not male_voice_missing()
        status = f"{TTS_NAMES[engine]}, {'мужской' if male else 'женский'} голос"
    if note:
        return f"{status}. {note}"
    if _piper_state.get(PIPER_VOICES[True]) == "загружается":
        status += f"; мужской голос Piper загружается (около {PIPER_MB} МБ)"
    return status


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


def _proxy() -> str | None:
    from urllib.request import getproxies

    proxies = getproxies()
    return proxies.get("https") or proxies.get("http") or None


def _edge_tts(text: str, voice: str, path: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(
        text, voice, proxy=_proxy(), connect_timeout=8, receive_timeout=20
    )
    communicate.save_sync(str(path))


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
            gender = "Male" if male else "Female"
            env = {**os.environ, "LM_TEXT": file.name, "LM_WAV": str(path), "LM_GENDER": gender}
            command = ["powershell", "-NoProfile", "-NonInteractive", "-Command", WINDOWS_TTS]
            result = subprocess.run(command, timeout=TTS_TIMEOUT, env=env)
            if result.returncode == 2:
                kind = "мужского" if male else "женского"
                raise RuntimeError(f"в Windows нет {kind} русского голоса")
            if result.returncode:
                raise RuntimeError(f"PowerShell завершился с кодом {result.returncode}")
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
    fitting = [n for n in names if (n.split()[0] in ("Yuri", "Юрий")) == male]
    if not fitting:
        raise RuntimeError(
            "в macOS нет русского голоса Yuri" if male else "в macOS нет русского голоса"
        )
    return sorted(fitting, key=lambda n: "(" not in n)[0]


WINDOWS_TTS = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "$v = $s.GetInstalledVoices() | Where-Object { $_.VoiceInfo.Culture.Name -eq 'ru-RU' "
    "-and [string]$_.VoiceInfo.Gender -eq $env:LM_GENDER } "
    "| Select-Object -First 1; if (-not $v) { exit 2 }; "
    "$s.SelectVoice($v.VoiceInfo.Name); $s.SetOutputToWaveFile($env:LM_WAV); "
    "$s.Speak([IO.File]::ReadAllText($env:LM_TEXT, [Text.Encoding]::UTF8)); $s.Dispose()"
)
