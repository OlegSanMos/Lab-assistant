import shutil
import sys
import threading
import time
import wave
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from conftest import make_wav
from labmentor import speech, tools
from labmentor.config import Settings, use_system_certificates
from labmentor.llm import (
    CachedEmbeddings,
    LLMUnavailableError,
    create_chat_model,
    create_embeddings,
)


@pytest.fixture(autouse=True)
def fresh_speech_state():
    speech._model_state.clear()
    speech._skip_until.clear()
    speech._tts_state["engine"] = ""
    yield
    speech._model_state.clear()
    speech._skip_until.clear()


class FakeWhisper:
    def __init__(self, text=" Какова цель ", tail="работы? "):
        self.parts = [text, tail]

    def transcribe(self, audio, language, vad_filter):
        assert language == "ru" and vad_filter
        assert audio.dtype == np.float32 and audio.ndim == 1
        return [SimpleNamespace(text=part) for part in self.parts], None


def fake_openai(text=" Закон Ома ", error=None, calls=None):
    class FakeOpenAI:
        def __init__(self, api_key, base_url, **kwargs):
            if calls is not None:
                calls.update(key=api_key, **kwargs)

            def create(model, file, language):
                if error:
                    raise error
                return SimpleNamespace(text=text)

            self.audio = SimpleNamespace(transcriptions=SimpleNamespace(create=create))

    return FakeOpenAI


def test_text_for_speech_removes_markup():
    text = "## Ответ\n**R = 101 Ом** [lr1_ohm_law.md], см. https://example.com и `код`"
    assert speech.text_for_speech(text) == "Ответ R = 101 Ом , см. и код"


def test_text_for_speech_shortens_long_answers():
    table = "| U, В | I, мА |\n|---|---|\n| 1 | 10 |\n"
    text = "⚠️ Первое предложение.\n" + table + "Второе предложение! " * 60
    spoken = speech.text_for_speech(text, limit=200)
    assert spoken.startswith("Первое предложение. Второе предложение!")
    assert spoken.endswith("! Подробности в чате.") and len(spoken) < 230
    assert "|" not in spoken and "⚠" not in spoken


def test_stt_engine_order(settings):
    assert speech.stt_engines(settings) == ["openai", "local"]
    settings.provider = "ollama"
    assert speech.stt_engines(settings) == ["local", "openai"]
    settings.openai_api_key = ""
    assert speech.stt_engines(settings) == ["local"]
    settings.stt_engine = "openai"
    assert speech.stt_engines(settings) == ["openai"]


def test_local_transcription_uses_whisper(monkeypatch, settings, tmp_path):
    monkeypatch.setattr(speech, "_whisper", lambda size: FakeWhisper())
    settings.stt_engine = "local"
    audio = make_wav(tmp_path / "question.wav")
    assert speech.transcribe(audio, settings) == "Какова цель работы?"
    assert "✅ готова" in speech.stt_status(settings)


def test_api_transcription_uses_openai_client(monkeypatch, settings, tmp_path):
    calls = {}
    monkeypatch.setattr("openai.OpenAI", fake_openai(calls=calls))
    audio = tmp_path / "question.webm"
    audio.write_bytes(b"data")
    assert speech.transcribe(audio, settings) == "Закон Ома"
    assert calls["key"] == "test-key" and calls["timeout"] == 60


def test_transcription_falls_back_and_explains_errors(monkeypatch, settings, tmp_path):
    audio = make_wav(tmp_path / "question.wav")
    settings.provider = "ollama"

    def no_model(size):
        try:
            raise ConnectionError("[SSL: CERTIFICATE_VERIFY_FAILED] self-signed certificate")
        except ConnectionError as cause:
            raise OSError("We couldn't connect to huggingface.co") from cause

    monkeypatch.setattr(speech, "_whisper", no_model)
    monkeypatch.setattr("openai.OpenAI", fake_openai())
    assert speech.transcribe(audio, settings) == "Закон Ома"
    assert "❌ не загружена (ConnectionError" in speech.stt_status(settings)

    region = RuntimeError("Error code: 403 - unsupported_country_region_territory")
    monkeypatch.setattr("openai.OpenAI", fake_openai(error=region))
    with pytest.raises(speech.SpeechError) as failure:
        speech.transcribe(audio, settings)
    message = str(failure.value)
    assert "Whisper «small»" in message and "CERTIFICATE_VERIFY_FAILED" in message
    assert "truststore" in message and "HF_ENDPOINT" in message
    assert "не работает в вашей стране" in message and "STT_ENGINE=local" in message

    monkeypatch.setattr(speech, "_whisper", lambda size: FakeWhisper("", ""))
    with pytest.raises(speech.SpeechError, match="Речь не распознана"):
        speech.transcribe(audio, settings)

    def no_package(size):
        raise ImportError("No module named 'faster_whisper'")

    monkeypatch.setattr(speech, "_whisper", no_package)
    settings.stt_engine = "local"
    with pytest.raises(speech.SpeechError, match="Не установлен пакет faster-whisper"):
        speech.transcribe(audio, settings)


def test_broken_recording_and_model_on_disk(monkeypatch, settings, tmp_path):
    class BrokenModel:
        def transcribe(self, audio, language, vad_filter):
            raise RuntimeError("out of memory")

    monkeypatch.setattr(speech, "_whisper", lambda size: BrokenModel())
    settings.stt_engine = "local"
    broken = tmp_path / "rec.wav"
    broken.write_bytes(b"RIFF not really audio")
    with pytest.raises(speech.SpeechError, match="Не удалось прочитать аудиозапись rec.wav"):
        speech.transcribe(broken, settings)
    with pytest.raises(speech.SpeechError, match="Ошибка распознавания речи моделью Whisper"):
        speech.transcribe(make_wav(tmp_path / "ok.wav"), settings)
    assert speech.model_cached(str(tmp_path))


def test_decode_audio_without_metadata_errors(monkeypatch, tmp_path):
    import av

    real_open = av.open

    def pyav19_open(file, *args, **kwargs):
        if "metadata_errors" in kwargs:
            raise TypeError("open() got an unexpected keyword argument 'metadata_errors'")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(av, "open", pyav19_open)
    path = tmp_path / "stereo.wav"
    rate, seconds = 44100, 2
    tone = (8000 * np.sin(2 * np.pi * 440 * np.arange(rate * seconds) / rate)).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(np.repeat(tone, 2).tobytes())
    audio = speech.decode_audio(path)
    assert audio.dtype == np.float32 and audio.ndim == 1
    assert abs(len(audio) - 16000 * seconds) < 400 and 0.2 < np.abs(audio).max() < 0.3


def test_short_error_keeps_explicit_message():
    try:
        try:
            raise KeyError("внутренняя ошибка")
        except KeyError:
            raise ValueError("Error code: 403 - unsupported_country_region_territory") from None
    except ValueError as exc:
        assert (
            speech._short(exc)
            == "ValueError: Error code: 403 - unsupported_country_region_territory"
        )


def test_system_certificates(monkeypatch):
    calls = []
    monkeypatch.setitem(
        sys.modules, "truststore", SimpleNamespace(inject_into_ssl=lambda: calls.append(1))
    )
    assert use_system_certificates() and calls == [1]
    monkeypatch.setitem(sys.modules, "truststore", None)
    assert not use_system_certificates()


def test_preload_loads_model_in_background(monkeypatch, settings):
    settings.provider = "ollama"
    assert speech.preload(settings.__class__(provider="openai", openai_api_key="k")) is None
    started = threading.Event()
    release = threading.Event()

    def slow_model(size):
        started.set()
        release.wait(5)
        return FakeWhisper()

    monkeypatch.setattr(speech, "_whisper", slow_model)
    monkeypatch.setattr(speech, "model_cached", lambda size: True)
    thread = speech.preload(settings, quiet=True)
    assert started.wait(5) and speech.stt_loading(settings)
    assert "⏳ загружается" in speech.stt_status(settings)
    release.set()
    thread.join(5)
    assert not speech.stt_loading(settings) and "✅ готова" in speech.stt_status(settings)


def test_synthesis_falls_back_and_skips_failed_engines(monkeypatch, tmp_path):
    calls = []

    def engine(name, works=True):
        def run(text, voice, path):
            calls.append(name)
            if not works:
                raise ConnectionError("нет сети")
            path.write_bytes(name.encode())

        return run

    monkeypatch.setattr(speech, "_edge_tts", engine("edge"))
    path = speech.synthesize("**Ответ:** R = 101 Ом", "ru-RU-SvetlanaNeural", tmp_path)
    assert open(path, "rb").read() == b"edge" and speech.tts_status() == speech.TTS_NAMES["edge"]

    monkeypatch.setattr(speech, "_edge_tts", engine("edge", works=False))
    monkeypatch.setattr(speech, "_google_tts", engine("gtts", works=False))
    monkeypatch.setattr(speech, "_system_voice", engine("system"))
    calls.clear()
    path = speech.synthesize("Ответ", "ru-RU-SvetlanaNeural", tmp_path)
    assert path.endswith(".wav") and calls == ["edge", "gtts", "system"]
    calls.clear()
    speech.synthesize("Второй ответ", "ru-RU-SvetlanaNeural", tmp_path)
    assert calls == ["system"]

    monkeypatch.setattr(speech, "_system_voice", engine("system", works=False))
    speech._skip_until.clear()
    assert speech.synthesize("Ответ", "ru-RU-SvetlanaNeural", tmp_path) is None
    assert speech.synthesize("```код```", "ru-RU-SvetlanaNeural", tmp_path) is None


def test_hanging_tts_engine_times_out(monkeypatch, tmp_path):
    monkeypatch.setattr(speech, "TTS_TIMEOUT", 0.2)
    error = speech.try_engine(lambda text, voice, path: time.sleep(2), "Ответ", "", tmp_path / "a")
    assert error == "нет ответа за 0.2 с"


def test_edge_tts_gets_timeouts(monkeypatch, tmp_path):
    seen = {}

    class Communicate:
        def __init__(self, text, voice, **kwargs):
            seen.update(kwargs, voice=voice)

        def save_sync(self, path):
            open(path, "wb").write(b"mp3")

    monkeypatch.setattr("edge_tts.Communicate", Communicate)
    speech._edge_tts("Ответ", "ru-RU-DmitryNeural", tmp_path / "a.mp3")
    assert seen == {"connect_timeout": 8, "receive_timeout": 20, "voice": "ru-RU-DmitryNeural"}


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng не установлен")
def test_system_voice_works_offline(tmp_path):
    path = tmp_path / "answer.wav"
    speech._system_voice("Сопротивление сто один Ом", "ru-RU-SvetlanaNeural", path)
    assert path.read_bytes()[:4] == b"RIFF" and path.stat().st_size > 10_000


def test_system_voice_commands_for_macos_and_windows(monkeypatch, tmp_path):
    commands = []

    def run(command, **kwargs):
        commands.append((command, kwargs))
        out = "Milena              ru_RU    # Здравствуйте!\nYuri ru_RU # Привет\nAlex en_US # Hi\n"
        return SimpleNamespace(stdout=out)

    monkeypatch.setattr(speech.subprocess, "run", run)
    monkeypatch.setattr(speech.sys, "platform", "darwin")
    speech._mac_voice.cache_clear()
    speech._system_voice("Ответ", "ru-RU-DmitryNeural", tmp_path / "a.wav")
    say = commands[-1][0]
    assert say[:3] == ["say", "-v", "Yuri"] and "--file-format=WAVE" in say and "-f" in say
    speech._system_voice("Ответ", "ru-RU-SvetlanaNeural", tmp_path / "a.wav")
    assert commands[-1][0][2] == "Milena"

    monkeypatch.setattr(speech.sys, "platform", "win32")
    speech._system_voice("Ответ", "ru-RU-SvetlanaNeural", tmp_path / "w.wav")
    command, kwargs = commands[-1]
    assert command[0] == "powershell" and "System.Speech" in command[-1]
    assert kwargs["env"]["LM_WAV"].endswith("w.wav") and kwargs["env"]["LM_TEXT"]


def test_web_search_formats_results(monkeypatch):
    class FakeDDGS:
        def text(self, query, region, max_results):
            assert region == "ru-ru"
            return [{"title": "Сталь 45", "href": "https://example.com/45", "body": "E = 200 ГПа"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    result = tools.web_search.invoke({"query": "модуль Юнга стали 45"})
    assert "1. Сталь 45" in result and "E = 200 ГПа" in result


def test_web_search_reports_network_error(monkeypatch):
    class OfflineDDGS:
        def text(self, *args, **kwargs):
            raise ConnectionError("нет сети")

    monkeypatch.setattr("ddgs.DDGS", OfflineDDGS)
    assert "недоступен" in tools.web_search.invoke({"query": "плотность меди"})


def test_wikipedia_search(monkeypatch):
    page = {
        "title": "Закон Гука",
        "index": 1,
        "fullurl": "https://ru.wikipedia.org/wiki/Закон_Гука",
        "extract": "Утверждение о линейной связи деформации и напряжения.",
    }
    response = SimpleNamespace(json=lambda: {"query": {"pages": {"1": page}}})
    monkeypatch.setattr(tools.requests, "get", lambda *args, **kwargs: response)
    result = tools.wikipedia_search.invoke({"query": "закон Гука"})
    assert result.startswith("Закон Гука (https://ru.wikipedia.org/wiki/Закон_Гука)")


def test_calculator_tool():
    assert tools.calculator.invoke({"expression": "sqrt(0.5 * 100)"}) == "sqrt(0.5 * 100) = 7.07107"
    assert tools.calculator.invoke({"expression": "import os"}).startswith("Ошибка")


def test_chat_model_factory(settings):
    openai_model = create_chat_model(settings, model="gpt-4.1-mini", temperature=0)
    assert isinstance(openai_model, ChatOpenAI) and openai_model.max_tokens == 1500
    ollama_model = create_chat_model(settings, provider="ollama", model="qwen2.5:7b", num_ctx=4096)
    assert isinstance(ollama_model, ChatOllama) and ollama_model.num_ctx == 4096
    settings.openai_api_key = ""
    with pytest.raises(LLMUnavailableError):
        create_chat_model(settings)


def test_embeddings_factory_and_cache(settings, tmp_path):
    assert create_embeddings(settings) is None
    settings.embeddings_provider = "ollama"
    assert isinstance(create_embeddings(settings), CachedEmbeddings)

    class Counting(Embeddings):
        calls = 0

        def embed_documents(self, texts):
            Counting.calls += 1
            return [[float(len(t)), 1.0] for t in texts]

        def embed_query(self, text):
            return [float(len(text)), 1.0]

    cache = tmp_path / "cache.json"
    first = CachedEmbeddings(Counting(), cache)
    assert first.embed_documents(["закон Ома", "ВАХ"]) == [[9.0, 1.0], [3.0, 1.0]]
    first.embed_documents(["ВАХ"])
    second = CachedEmbeddings(Counting(), cache)
    second.embed_documents(["закон Ома"])
    assert Counting.calls == 1


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("NUM_CTX", "4096")
    monkeypatch.setenv("HISTORY_TOKENS", "1500")
    monkeypatch.delenv("CHAT_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_PROVIDER", raising=False)
    settings = Settings.from_env()
    assert (settings.chat_model, settings.vision_model, settings.embedding_model) == (
        "qwen2.5:7b",
        "qwen2.5vl:7b",
        "bge-m3",
    )
    assert settings.num_ctx == 4096 and settings.history_tokens == 1500
    assert settings.embeddings_provider == "ollama"


def test_check_setup_gives_hints(monkeypatch, settings):
    from labmentor.llm import check_setup

    assert check_setup(settings) == []
    settings.openai_api_key = ""
    assert "OPENAI_API_KEY" in check_setup(settings)[0]

    ollama = Settings(provider="ollama", storage_dir=settings.storage_dir)
    tags = {"models": [{"name": "qwen2.5:7b"}, {"name": "bge-m3:latest"}]}
    monkeypatch.setattr(
        "labmentor.llm.requests.get", lambda *args, **kwargs: SimpleNamespace(json=lambda: tags)
    )
    assert check_setup(ollama) == [
        "Модель qwen2.5vl:7b не загружена: выполните ollama pull qwen2.5vl:7b"
    ]

    def offline(*args, **kwargs):
        raise ConnectionError("connection refused")

    monkeypatch.setattr("labmentor.llm.requests.get", offline)
    assert "запустите Ollama" in check_setup(ollama)[0]
