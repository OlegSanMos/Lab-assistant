import socket
import time

import pytest
from PIL import Image

from conftest import (
    DEMO_MANUALS,
    SAMPLES,
    ScriptedChatModel,
    lab_responder,
    make_video,
    make_wav,
)
from labmentor.assistant import LabAssistant
from labmentor.config import Settings
from labmentor.ui import INPUT_HINT, LAB_LABEL, build_ui, launch_options

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
    assert page.get_by_role("tab", name="Обработка данных").is_visible()
    visible(page, "Здравствуйте! Я ЛабМентор")


def test_dialog_fills_the_window(page):
    sizes = """() => ({
        scroll: document.documentElement.scrollHeight,
        inner: window.innerHeight,
        chat: document.querySelector("#chat").getBoundingClientRect().width,
        sidebar: document.querySelector(".sidebar").getBoundingClientRect().width,
    })"""
    for width, height in ((1440, 1000), (1920, 1080), (1280, 800)):
        page.set_viewport_size({"width": width, "height": height})
        page.wait_for_function(
            "() => document.documentElement.scrollHeight <= window.innerHeight + 2",
            timeout=10_000,
        )
        found = page.evaluate(sizes)
        assert found["chat"] > width - found["sidebar"] - 120
        assert 300 <= found["sidebar"] <= 420


def test_dialog_expands_to_full_screen(page):
    state = """() => {
        const chat = document.querySelector("#chat").getBoundingClientRect();
        const shown = (selector) => Boolean(document.querySelector(selector).offsetParent);
        return {
            focus: document.body.classList.contains("lm-focus"),
            sidebar: document.querySelector(".sidebar").classList.contains("open"),
            menus: [shown("#app-title"), shown(".tab-wrapper")],
            height: Math.round(chat.height),
            width: Math.round(chat.width),
            fits: document.documentElement.scrollHeight <= window.innerHeight + 2,
        };
    }"""
    grown = """normal => {
        const chat = document.querySelector("#chat").getBoundingClientRect();
        return chat.width > normal.width + 250 && chat.height > normal.height + 100;
    }"""
    restored = """normal => {
        const chat = document.querySelector("#chat").getBoundingClientRect();
        return Math.round(chat.width) === normal.width && Math.round(chat.height) === normal.height;
    }"""
    page.wait_for_function(
        "() => document.documentElement.scrollHeight <= window.innerHeight + 2", timeout=10_000
    )
    normal = page.evaluate(state)
    assert normal["sidebar"] and normal["menus"] == [True, True] and not normal["focus"]
    expand = page.get_by_role("button", name="Развернуть диалог")
    collapse = page.get_by_role("button", name="Обычный вид")
    sync_api.expect(collapse).to_be_hidden()
    for close in (collapse.click, lambda: page.keyboard.press("Escape")):
        expand.click()
        page.wait_for_function(grown, arg=normal, timeout=10_000)
        full = page.evaluate(state)
        assert full["focus"] and not full["sidebar"] and full["menus"] == [False, False]
        assert full["fits"]
        sync_api.expect(expand).to_be_hidden()
        sync_api.expect(page.get_by_placeholder(INPUT)).to_be_visible()
        close()
        page.wait_for_function(restored, arg=normal, timeout=10_000)
        assert page.evaluate(state) == normal
        sync_api.expect(collapse).to_be_hidden()
    apart = """() => {
        const button = document.querySelector("#btn-focus-on").getBoundingClientRect();
        const tabs = ".tab-container:not(.visually-hidden) button, .overflow-menu button";
        return [...document.querySelectorAll(tabs)]
            .filter(tab => tab.offsetParent)
            .map(tab => tab.getBoundingClientRect())
            .every(tab => tab.right <= button.left || tab.left >= button.right);
    }"""
    for width in (1024, 820, 1440):
        page.set_viewport_size({"width": width, "height": 1000})
        page.wait_for_function(apart, timeout=10_000)
        sync_api.expect(expand).to_be_visible()


def test_main_blocks_have_icons(page):
    selectors = [
        "#app-title h2",
        "#side-title h3",
        "#acc-model > .label-wrap > span:first-child",
        "#acc-context > .label-wrap > span:first-child",
        "#tab-dialog-button",
        "#tab-data-button",
        "#tab-report-button",
        "#tab-about-button",
        "#btn-clear",
        "#btn-focus-on",
    ]
    icons = page.evaluate(
        """selectors => selectors.map(selector => {
            const style = getComputedStyle(document.querySelector(selector), "::before");
            return [style.maskImage || style.webkitMaskImage, parseFloat(style.width)];
        })""",
        selectors,
    )
    assert all(mask.startswith('url("data:image/svg+xml') and width > 10 for mask, width in icons)
    page.get_by_role("tab", name="Обработка данных").click()
    sync_api.expect(page.locator("#btn-explain")).to_be_visible()
    mask = page.evaluate(
        "() => getComputedStyle(document.querySelector('#btn-explain'), '::before').maskImage"
    )
    assert mask.startswith('url("data:image/svg+xml')


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
    visible(page, "Файл: filter.cir")
    visible(page, "Цель работы: проверить закон Ома")
    visible(page, "Файлы в памяти диалога: filter.cir")


def test_video_is_watched_in_chat(page, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "labmentor.assistant.transcribe", lambda path, settings: "Включаю генератор"
    )
    ask(page, "", files=[make_video(tmp_path / "opyt.mp4")])
    visible(page, "Видео: opyt.mp4")
    visible(page, "tau около 1 с")
    sheet = page.locator("img[src*='video_']").first
    sync_api.expect(sheet).to_be_visible(timeout=30_000)
    page.wait_for_function(
        "image => image.complete && image.naturalWidth > 400", arg=sheet.element_handle()
    )
    visible(page, "Файлы в памяти диалога: opyt.mp4")


def test_report_has_real_pictures(page, tmp_path):
    photo = tmp_path / "stand.jpg"
    Image.new("RGB", (640, 480), (40, 90, 160)).save(photo)
    page.get_by_label(LAB_LABEL).click()
    page.get_by_role("option", name="Лабораторная работа № 1.", exact=False).first.click()
    page.get_by_role("tab", name="Отчет").click()
    page.locator('input[type=file][accept="image/*"]').set_input_files(str(photo))
    page.wait_for_timeout(800)
    captions = page.get_by_label("Подписи к рисункам по одной на строку, иначе их даст модель")
    captions.fill("Фото лабораторного стенда")
    page.get_by_role("button", name="Сформировать черновик отчета").click()
    visible(page, "Рисунок 1. Фото лабораторного стенда")
    visible(page, "Материалы, приложенные к работе, приведены на рисунке 1.")
    picture = page.locator("img[alt='Рисунок 1. Фото лабораторного стенда']").first
    page.wait_for_function(
        "image => image.complete && image.naturalWidth == 640", arg=picture.element_handle()
    )
    assert page.locator("a[download$='.docx']").count() >= 1


def test_data_processing_tab(page):
    page.get_by_role("tab", name="Обработка данных").click()
    page.locator('input[type=file][accept=".csv, .tsv, .xlsx"]').set_input_files(
        str(SAMPLES / "ohm_law.csv")
    )
    visible(page, "Загружена таблица ohm_law.csv")
    page.get_by_placeholder("R, Ом = U_В / I_мА * 1000").fill("R, Ом = U_В / I_мА * 1000")
    page.get_by_role("button", name="Добавить столбец").click()
    visible(page, "Добавлен столбец R_Ом")
    page.get_by_role("button", name="Обработать").click()
    visible(page, "Коэффициент детерминации R² = 0.99998")
    visible(page, "Результат: R_Ом = 101.34 ± 0.32, P = 0.95")


def test_conclusions_for_the_lab_chosen_on_data_tab(page, monkeypatch):
    explain_results = LabAssistant.explain_results

    def slow(self, session, options):
        time.sleep(2)
        return explain_results(self, session, options)

    monkeypatch.setattr(LabAssistant, "explain_results", slow)
    page.get_by_role("tab", name="Обработка данных").click()
    page.locator('input[type=file][accept=".csv, .tsv, .xlsx"]').set_input_files(
        str(SAMPLES / "ohm_law.csv")
    )
    visible(page, "Загружена таблица ohm_law.csv")
    page.get_by_role("button", name="Обработать").click()
    visible(page, "Коэффициент детерминации")
    explain = page.get_by_role("button", name="Сформулировать выводы (LLM)")
    explain.click()
    visible(page, "Выберите лабораторную работу, по которой нужны выводы")
    choice = page.get_by_label("Лабораторная работа для выводов")
    choice.click()
    page.get_by_role("option", name="Лабораторная работа № 1.", exact=False).first.click()
    first = choice.input_value()
    sync_api.expect(page.get_by_label(LAB_LABEL)).to_have_value(first)
    explain.click()
    visible(page, "Готовлю выводы по лабораторной работе 1:")
    button = page.locator("#btn-explain")
    sync_api.expect(button).to_be_disabled()
    sync_api.expect(button).to_have_text("Выводы готовятся...")
    spinning = page.evaluate(
        """() => [document.querySelector("#btn-explain"), document.querySelector(".lm-loading")]
            .map(element => getComputedStyle(element, "::before").animationName)"""
    )
    assert spinning == ["lm-spin", "lm-spin"]
    visible(page, f"Выводы по работе «{first}»")
    sync_api.expect(button).to_be_enabled()
    sync_api.expect(button).to_have_text("Сформулировать выводы (LLM)")
    assert page.locator(".lm-loading").count() == 0
    visible(page, "Зависимость подтверждена")
    page.get_by_label(LAB_LABEL).click()
    page.get_by_role("option", name="Лабораторная работа № 2.", exact=False).first.click()
    second = page.get_by_label(LAB_LABEL).input_value()
    assert second != first
    sync_api.expect(choice).to_have_value(second)


def test_clear_dialog(page):
    ask(page, "Привет!", files=())
    visible(page, "Сообщений в окне: 2 из 2")
    sync_api.expect(page.get_by_placeholder(INPUT)).to_be_editable(timeout=30_000)
    page.get_by_role("button", name="Очистить диалог").click()
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
    visible(page, f"Голосом: {question}")
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
