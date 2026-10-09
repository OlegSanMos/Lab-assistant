from pathlib import Path

from docx import Document

from labmentor.assistant import Options
from labmentor.ui import build_ui, context_outputs, render


def run(assistant, session, question, image=None, **options):
    replies = list(assistant.respond(session, question, Options(**options), image))
    return replies[-1]


def test_rag_answer_with_sources_updates_memory(assistant):
    session = assistant.new_session()
    reply = run(assistant, session, "Какова цель лабораторной работы № 1?")
    assert reply.route == "manual" and not reply.error
    assert "закон Ома" in reply.text
    assert any("№ 1" in source for source in reply.sources)
    assert len(session.memory.messages) == 2


def test_router_selects_flowchart_and_draws_image(assistant):
    session = assistant.new_session()
    reply = run(assistant, session, "Нарисуй схему порядка выполнения работы 1")
    assert reply.route == "flowchart"
    assert Path(reply.images[0]).exists() and "Собрать цепь" in reply.text


def test_data_agent_calls_tools_and_plots(assistant, sample):
    session = assistant.new_session()
    got = assistant.ingest(session, [sample("ohm_law.csv")])
    assert got.notes[0].startswith("Загружена таблица") and got.image is None and got.speech == ""
    reply = run(assistant, session, "Построй график I(U) и найди сопротивление")
    assert reply.route == "data"
    assert [step["title"] for step in reply.steps[1:]] == [
        "🔧 Инструмент describe_data",
        "🔧 Инструмент fit_dependency",
    ]
    assert "Результат: МНК-аппроксимация" in reply.steps[2]["content"]
    assert len(reply.images) == 1 and Path(reply.images[0]).exists()
    assert "Сопротивление R = 1/k" in reply.text


def test_data_agent_requires_table(assistant):
    reply = run(assistant, assistant.new_session(), "Найди погрешность", mode="data")
    assert "загрузите таблицу" in reply.text


def test_research_agent_uses_manual_search(assistant):
    reply = run(
        assistant, assistant.new_session(), "Найди в интернете модуль Юнга", allow_web=False
    )
    assert reply.route == "research"
    assert reply.steps[1]["title"] == "🔧 Инструмент search_lab_manuals"
    assert "Лабораторная работа № 2" in reply.steps[1]["content"]


def test_exam_flow(assistant):
    session = assistant.new_session()
    reply = run(assistant, session, "Проверь меня по работе 1", exam_questions=2)
    assert reply.route == "exam" and "Вопрос 1 из 2" in reply.text
    reply = run(assistant, session, "Ток равен напряжению, деленному на сопротивление")
    assert "8/10" in reply.text and "Вопрос 2 из 2" in reply.text
    reply = run(assistant, session, "Последовательно")
    assert "Итог тренировочной защиты" in reply.text and session.exam is None


def test_exam_can_be_stopped(assistant):
    session = assistant.new_session()
    run(assistant, session, "Задай вопросы для защиты", mode="exam")
    reply = run(assistant, session, "стоп")
    assert "завершена до первого ответа" in reply.text and session.exam is None


def test_vision_sends_image_to_model(assistant, scripted_llm, sample):
    session = assistant.new_session()
    assert "Прикрепите изображение" in run(assistant, session, "Что на фото?", mode="vision").text
    reply = run(
        assistant, session, "Определи постоянную времени", image=sample("rc_oscillogram.png")
    )
    assert reply.route == "vision" and "tau" in reply.text
    content = scripted_llm.calls[-1][-1].content
    assert content[1]["type"] == "image_url" and content[1]["image_url"]["url"].startswith(
        "data:image/jpeg"
    )


def test_summary_after_context_overflow(assistant):
    session = assistant.new_session()
    question = "Расскажи подробно, какова цель лабораторной работы № 1 и что нужно сделать? " * 4
    for _ in range(3):
        run(assistant, session, question, history_tokens=256)
    assert session.memory.max_tokens == 256 and session.context_plan["num_ctx"] == 8192
    assert assistant.update_memory(session, Options()) is True
    assert "закону Ома" in session.memory.summary


def test_context_plan_respects_model_window(assistant):
    session = assistant.new_session()
    plan = assistant.apply_context(session, Options(num_ctx=4096, history_tokens=8000, top_k=4))
    assert plan["history"] == 4096 - plan["system"] - 300 - plan["rag"] - plan["answer"]
    assert session.memory.max_tokens == plan["history"]


def test_data_tab_and_report(assistant, sample):
    session = assistant.new_session()
    assistant.load_table(session, sample("ohm_law.csv"))
    plot, results = assistant.analyze(session, "U_В", "I_мА", 1, 0.95, "I_мА")
    assert Path(plot).exists() and "МНК-аппроксимация" in results and "Статистика I_мА" in results
    options = Options(lab=assistant.lab_titles()[0])
    assert assistant.explain_results(session, options)
    markdown, path = assistant.make_report(session, options, "Опыт при 22 °C")
    text = "\n".join(p.text for p in Document(path).paragraphs)
    assert "## Выводы" in markdown and "Закон Ома подтвержден" in text and "Таблица 1" in text


def test_ingest_document_and_error(assistant, tmp_path):
    manual = tmp_path / "lr4.md"
    manual.write_text(
        "# Лабораторная работа № 4. Изучение трансформатора\nКоэффициент трансформации.",
        encoding="utf-8",
    )
    broken = tmp_path / "broken.xlsx"
    broken.write_text("not an excel file")
    notes = assistant.ingest(assistant.new_session(), [str(manual), str(broken)]).notes
    assert any("ошибка обработки" in n for n in notes) and any(
        "База знаний пополнена" in n for n in notes
    )
    assert "Лабораторная работа № 4. Изучение трансформатора" in assistant.lab_titles()


def test_llm_error_is_reported(settings):
    from labmentor.assistant import LabAssistant

    settings.openai_api_key = ""
    reply = run(LabAssistant(settings), LabAssistant(settings).new_session(), "Привет", mode="chat")
    assert reply.error and "OPENAI_API_KEY" in reply.text


def test_ui_builds_and_renders(assistant):
    demo = build_ui(assistant)
    assert demo is not None
    session = assistant.new_session()
    reply = run(assistant, session, "Какова цель работы № 1?")
    messages = render(reply)
    assert messages[0]["metadata"]["title"] == "🧭 Диспетчер"
    assert messages[-1]["metadata"]["title"].startswith("📚")
    assert "из 2000 токенов" in context_outputs(session)[0]
