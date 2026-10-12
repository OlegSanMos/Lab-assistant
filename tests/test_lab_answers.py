import pytest

from conftest import ScriptedChatModel, course_responder
from labmentor.assistant import LabAssistant, Options, Reply, cut_loop
from labmentor.config import Settings
from labmentor.knowledge import overview
from labmentor.ui import render

QUESTION = "Расскажи. о чем лабораторная работа 1.4 и что в ней нужно делать по шагам"
FIGURE_9 = "Примеры графиков статических характеристик приведены в таблице 3 и рисунке 9"
LOOP = (
    "Требования к отчёту перечислены в методичке.\n\nПримеры измерений\n"
    "Примеры измеренных характеристик приведены в таблице 3 [lab_04_cmos.md].\n"
    + f"Примеры графиков\n{FIGURE_9} [lab_04_cmos.md].\n" * 6
    + "Примеры графиков\nПримеры графиков статических характеристик приведены в таб"
)


def answering(tmp_path, text: str | None = None):
    def responder(messages):
        if text and "Отвечай на основе фрагментов" in messages[0].text:
            return text
        return course_responder(messages)

    settings = Settings(
        provider="openai", openai_api_key="k", embeddings_provider="none", storage_dir=tmp_path
    )
    llm = ScriptedChatModel(responder=responder)
    return LabAssistant(settings, llm_factory=lambda **kwargs: llm), llm


def test_question_about_a_lab_uses_only_its_manual(course_assistant):
    sources = {doc.metadata["source"] for doc in course_assistant.kb.search(QUESTION, 6)}
    assert sources == {"lab_04_cmos.md"}
    found = course_assistant.kb.search("Как оформить таблицы в отчете по работе 1.2?", 8)
    assert {doc.metadata["source"] for doc in found} <= {"lab_02_bjt.md", "report_requirements.md"}


def test_overview_keeps_goals_tasks_and_report_requirements(course_assistant):
    text = course_assistant.kb.texts["lab_04_cmos.md"]
    full = overview(text, 12000)
    for part in (
        "1.4.1. Цели работы",
        "1.4.3. Описание лабораторного стенда",
        "1.4.4. Задание для работы в лаборатории",
        "1.4.5. Задание для работы в программе SPICE-моделирования",
        "1.4.6. Требования к отчёту",
    ):
        assert part in full
    assert "Контрольные вопросы" not in full and "Рекомендуемая литература" not in full
    short = overview(text, 3000)
    assert len(short) < 3100 and "1.4.1. Цели работы" in short
    assert "1.4.4. Задание для работы в лаборатории" in short


def test_overview_question_gets_the_whole_lab_task(tmp_path):
    assistant, llm = answering(tmp_path)
    reply = list(assistant.respond(assistant.new_session(), QUESTION, Options()))[-1]
    system = llm.calls[-1][0].content
    assert reply.route == "manual" and "Не повторяй одни и те же предложения" in system
    context = system.split("Фрагменты методических указаний:", 1)[1]
    for part in ("1.4.1. Цели работы", "1.4.4. Задание для работы в лаборатории", "отчёту"):
        assert part in context
    assert "lab_01_diod-stab.md" not in context and "lab_02_bjt.md" not in context
    assert any("(фрагмент основные разделы)" in source for source in reply.sources)


def test_figures_named_in_the_answer_are_shown_in_chat(tmp_path):
    assistant, _ = answering(tmp_path, f"{FIGURE_9} [lab_04_cmos.md].")
    reply = list(assistant.respond(assistant.new_session(), QUESTION, Options()))[-1]
    assert reply.text.startswith(FIGURE_9)
    assert reply.text.endswith("](images/lab_04_cmos/fig09.png)")
    assert "![Рис. 9. Примеры измеренных статических характеристик" in reply.text
    content = render(reply, assistant.find_image)[-2]["content"]
    assert "/gradio_api/file=" in content and "*Рис. 9. Примеры измеренных" in content
    shown = "Рисунок: ![Рис. 9. Схема](images/lab_04_cmos/fig09.png), см. рис. 9."
    assert assistant.answer_figures(shown, [], assistant.overview_lab(QUESTION, Options())) == []


def test_repeating_answer_is_cut(tmp_path):
    text, looped = cut_loop(LOOP)
    assert looped and text.count(FIGURE_9) == 1 and not text.endswith("Примеры графиков")
    normal = "| U, В | I, мА |\n|---|---|\n| 1 | 2 |\n|---|---|\n" + "Готово.\n" * 5
    assert cut_loop(normal) == (normal, False)
    assistant, _ = answering(tmp_path, LOOP)
    reply = list(assistant.respond(assistant.new_session(), QUESTION, Options()))[-1]
    assert reply.text.count(FIGURE_9) == 1 and reply.text.count("fig09.png") == 1


@pytest.mark.parametrize(
    ("question", "number"),
    [
        (QUESTION, "1.4"),
        ("Что нужно сделать в работе 1.2 по шагам?", "1.2"),
        ("Опиши лабораторную работу 3", "1.3"),
        ("Какова цель работы 1.1?", ""),
    ],
)
def test_overview_questions_are_recognized(course_assistant, question, number):
    title = course_assistant.overview_lab(question, Options())
    assert (f"работа {number}." in title) if number else title == ""


def test_lab_selected_in_sidebar_is_used_without_number(course_assistant):
    lab = next(t for t in course_assistant.lab_titles() if "работа 1.5." in t)
    assert course_assistant.overview_lab("О чем эта работа?", Options(lab=lab)) == lab
    assert course_assistant.overview_lab("О чем эта работа?", Options()) == ""
    assert render(Reply(text=""))[-1]["content"] == "Обрабатываю запрос..."
