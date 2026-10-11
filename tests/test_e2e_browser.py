import socket
import time

import pytest

from conftest import DEMO_MANUALS, SAMPLES, ScriptedChatModel, lab_responder, make_wav
from labmentor.assistant import LabAssistant
from labmentor.config import Settings
from labmentor.ui import INPUT_HINT, build_ui, launch_options

sync_api = pytest.importorskip("playwright.sync_api")
INPUT = INPUT_HINT
TRACK_AUDIO = """
window.__audio = {elements: [], starts: [], ended: 0};
const create = document.createElement.bind(document);
document.createElement = (tag, ...rest) => {
  const element = create(tag, ...rest);
  if (String(tag).toLowerCase() === "audio") {
    window.__audio.elements.push(element);
    element.addEventListener("playing", () => {
      const src = element.currentSrc;
      if (!window.__audio.starts.some((start) => start.src === src)) {
        window.__audio.starts.push({src, time: element.currentTime});
      }
    });
    element.addEventListener("ended", () => (window.__audio.ended += 1));
  }
  return element;
};
"""


@pytest.fixture(scope="module")
def app_url(tmp_path_factory):
    settings = Settings(
        provider="openai",
        openai_api_key="test-key",
        embeddings_provider="none",
        manuals_dir=DEMO_MANUALS,
        storage_dir=tmp_path_factory.mktemp("e2e"),
    )
    llm = ScriptedChatModel(responder=lab_responder)
    demo = build_ui(LabAssistant(settings, llm_factory=lambda **kwargs: llm))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
        **launch_options(settings),
    )
    yield f"http://127.0.0.1:{port}/"
    demo.close()


@pytest.fixture(scope="module")
def browser(tmp_path_factory):
    microphone = make_wav(tmp_path_factory.mktemp("mic") / "mic.wav", seconds=3)
    media = [
        "--autoplay-policy=no-user-gesture-required",
        "--use-fake-ui-for-media-stream",
        "--use-fake-device-for-media-stream",
        f"--use-file-for-fake-audio-capture={microphone}",
    ]
    with sync_api.sync_playwright() as playwright:
        try:
            browser = playwright.chromium.launch(args=media)
        except Exception as exc:
            pytest.skip(f"Chromium недоступен: {exc}")
        yield browser
        browser.close()


@pytest.fixture
def page(browser, app_url):
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.goto(app_url, wait_until="domcontentloaded")
    page.get_by_placeholder(INPUT).wait_for(timeout=30_000)
    yield page
    assert errors == [], f"Ошибки JavaScript на странице: {errors}"
    page.close()


def ask(page, text, files=()):
    if files:
        page.locator("input[type=file]").first.set_input_files([str(f) for f in files])
        page.wait_for_timeout(800)
    box = page.get_by_placeholder(INPUT)
    box.fill(text)
    box.press("Enter")


def visible(page, text, timeout=30_000):
    page.locator(f"text={text} >> visible=true").first.wait_for(timeout=timeout)


def test_page_has_controls(page):
    for label in (
        "Режим работы",
        "Бюджет токенов истории",
        "Размер контекстного окна модели, токенов",
    ):
        assert page.get_by_text(label).first.is_visible()
    assert page.get_by_role("tab", name="📊 Обработка данных").is_visible()
    visible(page, "Здравствуйте! Я ЛабМентор")


def test_chat_rag_answer_with_sources(page):
    ask(page, "Какова цель лабораторной работы № 1?")
    visible(page, "Цель работы: проверить закон Ома")
    visible(page, "Использованные фрагменты методичек")
    visible(page, "Сообщений в окне:")


def test_table_upload_runs_data_agent_and_plots(page):
    ask(page, "Построй график I(U) и найди сопротивление", files=[SAMPLES / "ohm_law.csv"])
    visible(page, "Сопротивление R = 1/k")
    visible(page, "Инструмент fit_dependency")
    assert page.locator("img[src*='plot_']").count() >= 1


def test_image_and_exam_dialog(page):
    ask(page, "Определи постоянную времени", files=[SAMPLES / "rc_oscillogram.png"])
    visible(page, "tau около 1 с")
    ask(page, "Проверь меня по работе 1")
    visible(page, "Вопрос 1 из 2")
    ask(page, "Ток прямо пропорционален напряжению")
    visible(page, "Оценка за ответ:")
    ask(page, "стоп")
    visible(page, "Итог тренировочной защиты")


def test_any_file_is_read_and_remembered(page, tmp_path):
    netlist = tmp_path / "filter.cir"
    netlist.write_text("* RC-фильтр\nR1 in out 1k\nC1 out 0 1u\n.end\n", encoding="utf-8")
    ask(page, "", files=[netlist])
    visible(page, "📎 filter.cir")
    visible(page, "Цель работы: проверить закон Ома")
    visible(page, "Файлы в памяти диалога: filter.cir")


def test_data_processing_tab(page):
    page.get_by_role("tab", name="📊 Обработка данных").click()
    page.locator('input[type=file][accept=".csv, .tsv, .xlsx"]').set_input_files(
        str(SAMPLES / "ohm_law.csv")
    )
    visible(page, "Загружена таблица ohm_law.csv")
    page.get_by_placeholder("R, Ом = U_В / I_мА * 1000").fill("R, Ом = U_В / I_мА * 1000")
    page.get_by_role("button", name="➕ Добавить столбец").click()
    visible(page, "Добавлен столбец R_Ом")
    page.get_by_role("button", name="📈 Обработать").click()
    visible(page, "Коэффициент детерминации R² = 0.99998")
    visible(page, "Результат: R_Ом = 101.34 ± 0.32, P = 0.95")


def test_clear_dialog(page):
    ask(page, "Привет!", files=())
    visible(page, "Сообщений в окне: 2 из 2")
    sync_api.expect(page.get_by_placeholder(INPUT)).to_be_editable(timeout=30_000)
    page.get_by_role("button", name="🧹 Очистить диалог").click()
    visible(page, "Здравствуйте! Я ЛабМентор")
    visible(page, "Сообщений в окне: 0 из 0")


def test_formulas_rendered_with_katex(page):
    ask(page, "Какие формулы нужны в работе 1?")
    visible(page, "Закон Ома для участка цепи")
    answer = page.locator(".message", has_text="Закон Ома для участка цепи").last
    sync_api.expect(answer.locator(".katex")).to_have_count(5)
    assert answer.locator(".katex-display").count() == 2
    assert answer.locator(".katex-error").count() == 0
    text = answer.inner_text()
    assert "$" not in text and "\\(" not in text and "\\[" not in text


def test_voice_question_by_microphone(page, monkeypatch, tmp_path):
    question = "Какова цель лабораторной работы № 1?"
    answer = make_wav(tmp_path / "answer.wav")
    monkeypatch.setattr("labmentor.assistant.transcribe", lambda path, settings: question)
    monkeypatch.setattr("labmentor.assistant.synthesize", lambda text, voice, out: str(answer))
    visible(page, "Запишите вопрос: после остановки записи он отправится сам")
    recorder = page.locator("#voice-input")
    recorder.get_by_role("button", name="Record").first.click()
    page.wait_for_timeout(1500)
    recorder.get_by_role("button", name="Stop").first.click()
    visible(page, f"🎤 {question}")
    visible(page, "Цель работы: проверить закон Ома")
    spoken = page.locator("#voice-answer a[download]")
    sync_api.expect(spoken).to_have_attribute("download", "answer.wav")
    sync_api.expect(recorder.get_by_role("button", name="Record").first).to_be_visible()


def ask_by_voice(page):
    recorder = page.locator("#voice-input")
    record = recorder.get_by_role("button", name="Record").first
    record.wait_for(timeout=30_000)
    record.click()
    page.wait_for_timeout(1200)
    recorder.get_by_role("button", name="Stop").first.click()


def start_time(page, number):
    page.wait_for_function("n => window.__audio.starts.length >= n", arg=number, timeout=30_000)
    return page.evaluate("n => window.__audio.starts[n - 1].time", number)


def test_new_spoken_answer_starts_from_beginning(browser, app_url, monkeypatch, tmp_path):
    answers = []

    def speak(text, voice, out_dir):
        if len(answers) == 1:
            time.sleep(3)
        answers.append(make_wav(tmp_path / f"answer_{len(answers) + 1}.wav", seconds=6))
        return str(answers[-1])

    question = "Какова цель лабораторной работы № 1?"
    monkeypatch.setattr("labmentor.assistant.transcribe", lambda path, settings: question)
    monkeypatch.setattr("labmentor.assistant.synthesize", speak)
    page = browser.new_page(viewport={"width": 1440, "height": 1000})
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.add_init_script(TRACK_AUDIO)
    page.goto(app_url, wait_until="domcontentloaded")
    page.get_by_text("Голос, интернет, защита").first.click()
    page.get_by_label("Озвучивать и ответы на вопросы, заданные текстом").check()
    ask(page, question)
    assert start_time(page, 1) < 0.5
    page.wait_for_timeout(1500)
    sync_api.expect(page.get_by_placeholder(INPUT)).to_be_editable(timeout=30_000)
    ask(page, "А какое оборудование нужно?")
    page.wait_for_function("() => window.__audio.elements.every(a => a.paused)", timeout=1500)
    assert start_time(page, 2) < 0.5
    page.wait_for_function("() => window.__audio.ended >= 1", timeout=15_000)
    ask_by_voice(page)
    assert start_time(page, 3) < 0.5
    assert len(answers) == 3 and errors == [], f"Ошибки JavaScript на странице: {errors}"
    page.close()
