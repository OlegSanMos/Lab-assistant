import socket

import pytest
from gradio_client import Client, handle_file

from conftest import SAMPLES, ScriptedChatModel, lab_responder, make_wav
from labmentor.assistant import LabAssistant
from labmentor.config import Settings
from labmentor.speech import SpeechError
from labmentor.ui import build_ui, launch_options

OPTIONS = ["auto", "openai", "gpt-4.1-mini", 0.3, 8192, 2000, True, 4, True, 3, None]
LAB_1 = "Лабораторная работа № 1. Проверка закона Ома для участка цепи"


def chat_text(chat: list) -> str:
    parts = []
    for message in chat:
        parts.append(str((message.get("metadata") or {}).get("title", "")))
        content = message["content"]
        for item in content if isinstance(content, list) else [content]:
            parts.append(item.get("text", "") if isinstance(item, dict) else str(item))
    return "\n".join(parts)


def has_image(chat: list) -> bool:
    return any(
        isinstance(item, dict) and item.get("type") == "file"
        for message in chat
        for item in (message["content"] if isinstance(message["content"], list) else [])
    )


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def start(assistant: LabAssistant):
    demo = build_ui(assistant)
    port = free_port()
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
        **launch_options(assistant.settings),
    )
    return demo, Client(f"http://127.0.0.1:{port}/", verbose=False)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    settings = Settings(
        provider="openai",
        openai_api_key="test-key",
        embeddings_provider="none",
        storage_dir=tmp_path_factory.mktemp("ui"),
    )
    llm = ScriptedChatModel(responder=lab_responder)
    demo, client = start(LabAssistant(settings, llm_factory=lambda **kwargs: llm))
    yield client
    demo.close()


def send(client, text, files=(), history=(), options=OPTIONS):
    message = {"text": text, "files": [handle_file(str(f)) for f in files]}
    job = client.submit(
        message, list(history), False, "ru-RU-SvetlanaNeural", *options, api_name="/on_message"
    )
    job.result()
    return job.outputs()


def test_rag_question_through_server(client):
    outputs = send(client, "Какова цель лабораторной работы № 1?")
    chat, _, audio, context, *_ = outputs[-1]
    assert len(outputs) > 2
    assert audio is None
    text = chat_text(chat)
    assert "Цель работы: проверить закон Ома" in text
    assert "🧭 Диспетчер" in text and "📚 Использованные фрагменты методичек" in text
    assert "Сообщений в окне:** 2 из 2" in context


def test_table_attachment_and_data_agent(client):
    outputs = send(
        client, "Построй график I(U) и найди сопротивление", files=[SAMPLES / "ohm_law.csv"]
    )
    chat = outputs[-1][0]
    text = chat_text(chat)
    assert "Загружена таблица ohm_law.csv: 14 строк" in text
    assert "🔧 Инструмент fit_dependency" in text and has_image(chat)
    table, x_col, y_col = next(o[5:8] for o in outputs if "headers" in (o[5] or {}))
    assert table["headers"] == ["U_В", "I_мА"] and len(table["data"]) == 14
    assert (x_col["value"], y_col["value"]) == ("U_В", "I_мА")


def test_image_attachment_goes_to_vision(client):
    chat = send(client, "Определи постоянную времени", files=[SAMPLES / "rc_oscillogram.png"])[-1][
        0
    ]
    assert "tau около 1 с" in chat_text(chat) and has_image(chat)


def test_exam_dialog(client):
    history = []
    for answer in ("Проверь меня по работе 1", "Ток пропорционален напряжению", "Последовательно"):
        history = send(client, answer, history=history)[-1][0]
    text = chat_text(history)
    assert "Вопрос 1 из 2" in text and "8/10" in text and "Итог тренировочной защиты" in text


def test_data_tab_endpoints(client):
    table, x_col, y_col, stat, note = client.predict(
        handle_file(str(SAMPLES / "ohm_law.csv")), api_name="/on_table"
    )
    assert note.startswith("Загружена таблица") and stat["value"] is None
    table, *_, note = client.predict(
        "R, Ом = U_В / I_мА * 1000", x_col["value"], y_col["value"], api_name="/on_add_column"
    )
    assert note == "Добавлен столбец R_Ом" and "R_Ом" in table["headers"]
    plot, results = client.predict("U_В", "I_мА", 1, 0.95, "R_Ом", api_name="/on_analyze")
    assert plot.endswith(".png")
    assert "R² = 0.9999" in results and "Результат: R_Ом = 101.34 ± 0.32, P = 0.95" in results
    report, markdown = client.predict("Опыт при 22 °C", *OPTIONS[:-1], LAB_1, api_name="/on_report")
    assert report.endswith(".docx") and "## Выводы" in markdown


def test_context_window_controls(client):
    options = OPTIONS[:4] + [4096, 8000] + OPTIONS[6:]
    info, _ = client.predict(*options, api_name="/on_context_settings")
    assert "Окно модели 4096" in info and "история 596" in info and "из 596 токенов" in info
    chat, _, context, summary = client.predict(api_name="/on_clear")
    assert chat == [] and "Сообщений в окне:** 0 из 0" in context and summary == ""


def test_voice_question_gets_spoken_answer(client, monkeypatch, tmp_path):
    spoken = []

    def fake_synthesize(text, voice, out_dir):
        spoken.append(text)
        return str(make_wav(tmp_path / "answer.wav"))

    question = "Какова цель лабораторной работы № 1?"
    monkeypatch.setattr("labmentor.assistant.transcribe", lambda path, settings: question)
    monkeypatch.setattr("labmentor.assistant.synthesize", fake_synthesize)
    recording = handle_file(str(make_wav(tmp_path / "question.wav")))
    job = client.submit(recording, [], "ru-RU-DmitryNeural", *OPTIONS, api_name="/on_voice")
    job.result()
    outputs = job.outputs()
    assert "🎤 Распознаю речь..." in chat_text(outputs[0][0])
    chat, _, audio = outputs[-1][:3]
    assert f"🎤 {question}" in chat_text(chat)
    assert "Цель работы: проверить закон Ома" in chat_text(chat)
    assert audio and audio.endswith(".wav")
    assert spoken[0].startswith("Цель работы")
    status, _ = client.predict(api_name="/on_speech_status")
    assert status.startswith("Запишите вопрос") and "Распознавание речи:" in status
    assert "Озвучивание:" in status


def test_voice_recognition_error_is_explained(client, monkeypatch, tmp_path):
    def not_recognized(path, settings):
        raise SpeechError("Речь не распознана: в записи не слышно слов.")

    monkeypatch.setattr("labmentor.assistant.transcribe", not_recognized)
    monkeypatch.setattr("labmentor.assistant.synthesize", lambda text, voice, out: None)
    recording = handle_file(str(make_wav(tmp_path / "silence.wav")))
    chat = client.predict(recording, [], "ru-RU-SvetlanaNeural", *OPTIONS, api_name="/on_voice")[0]
    assert "⚠️ Речь не распознана" in chat_text(chat) and "Цель работы" not in chat_text(chat)


def test_check_voice_button(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "labmentor.assistant.synthesize", lambda text, voice, out: str(make_wav(tmp_path / "a.wav"))
    )
    audio, status = client.predict("ru-RU-SvetlanaNeural", api_name="/on_check_voice")
    assert audio.endswith(".wav") and "Озвучивание:" in status


@pytest.mark.parametrize("provider", ["ollama", "openai"])
def test_app_starts_with_real_model_client(provider, tmp_path):
    url = f"http://127.0.0.1:{free_port()}"
    settings = Settings(
        provider=provider,
        openai_api_key="test-key",
        openai_base_url=f"{url}/v1",
        ollama_base_url=url,
        storage_dir=tmp_path,
    )
    llm = ScriptedChatModel(responder=lab_responder)
    assistant = LabAssistant(settings, llm_factory=lambda **kwargs: llm)
    assert assistant.kb.embeddings is not None
    demo, client = start(assistant)
    try:
        options = ["auto", provider, settings.chat_model, *OPTIONS[3:]]
        chat = send(client, "Какова цель лабораторной работы № 1?", options=options)[-1][0]
    finally:
        demo.close()
    assert "Цель работы: проверить закон Ома" in chat_text(chat)
