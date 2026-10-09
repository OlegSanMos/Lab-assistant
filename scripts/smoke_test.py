from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmentor.assistant import LabAssistant, Options, Reply
from labmentor.config import Settings, use_system_certificates
from labmentor.llm import check_setup
from labmentor.speech import synthesize, transcribe

SAMPLES = ROOT / "data" / "samples"
LAB_1 = "Лабораторная работа № 1. Проверка закона Ома для участка цепи"


@dataclass
class Result:
    name: str
    ok: bool
    seconds: float
    details: str


def ask(assistant: LabAssistant, session, question: str, image: str | None = None, **opts) -> Reply:
    reply = list(assistant.respond(session, question, Options(**opts), image))[-1]
    if reply.error:
        raise RuntimeError(reply.text)
    return reply


def scenario_chat(assistant):
    reply = ask(assistant, assistant.new_session(), "Ответь одним словом: ты готов?", mode="chat")
    assert reply.text.strip(), "пустой ответ"
    return reply.text


def scenario_router(assistant):
    cases = {
        "Какое оборудование нужно для работы № 2?": "manual",
        "Найди в интернете плотность меди": "research",
        "Задай мне контрольные вопросы для защиты": "exam",
        "Нарисуй блок-схему порядка выполнения работы": "flowchart",
    }
    hits = [
        assistant.choose_route(assistant.new_session(), q, Options(), None) == r
        for q, r in cases.items()
    ]
    assert sum(hits) >= 3, f"верно {sum(hits)} из {len(hits)}"
    return f"верно {sum(hits)} из {len(hits)} маршрутов"


def scenario_rag(assistant):
    reply = ask(
        assistant, assistant.new_session(), "Какова цель лабораторной работы № 1?", mode="manual"
    )
    assert any("№ 1" in s for s in reply.sources), "нет источника из ЛР 1"
    return reply.text


def scenario_data_agent(assistant):
    session = assistant.new_session()
    assistant.load_table(session, SAMPLES / "ohm_law.csv")
    reply = ask(
        assistant,
        session,
        "Построй график I(U) и найди сопротивление резистора с погрешностью",
        mode="data",
    )
    tools = [step["title"] for step in reply.steps]
    assert any("fit_dependency" in t for t in tools), f"агент не вызвал МНК: {tools}"
    assert reply.images, "график не построен"
    return f"инструменты: {', '.join(t.split()[-1] for t in tools[1:])}; график: {reply.images[0]}"


def scenario_research_agent(assistant):
    reply = ask(
        assistant,
        assistant.new_session(),
        "Найди в интернете модуль Юнга стали 45 и сравни с методичкой",
        mode="research",
    )
    tools = [step["title"].split()[-1] for step in reply.steps[1:]]
    assert tools, "агент не вызвал инструменты"
    return f"инструменты: {', '.join(tools)}; ответ: {reply.text}"


def scenario_flowchart(assistant):
    reply = ask(
        assistant,
        assistant.new_session(),
        "Нарисуй блок-схему порядка выполнения работы № 3",
        mode="flowchart",
    )
    assert reply.images, "схема не построена"
    return f"{reply.images[0]}\n{reply.text}"


def scenario_vision(assistant):
    reply = ask(
        assistant,
        assistant.new_session(),
        "Определи по осциллограмме постоянную времени",
        image=str(SAMPLES / "rc_oscillogram.png"),
    )
    return reply.text


def scenario_exam(assistant):
    session = assistant.new_session()
    first = ask(assistant, session, "Проверь меня по работе 1", mode="exam", exam_questions=3)
    assert session.exam is not None, "вопросы не сгенерированы"
    second = ask(
        assistant,
        session,
        "Ток прямо пропорционален напряжению и обратно пропорционален сопротивлению",
        mode="exam",
    )
    assert "/10" in second.text, "нет оценки ответа"
    return f"{first.text}\n\n{second.text}"


def scenario_context(assistant):
    session = assistant.new_session()
    question = "Расскажи подробно о порядке выполнения лабораторной работы № 2. " * 3
    for _ in range(3):
        ask(assistant, session, question, mode="manual", history_tokens=256)
    assert assistant.update_memory(session, Options()), "резюме не построено"
    return session.memory.summary


def scenario_report(assistant):
    session = assistant.new_session()
    assistant.load_table(session, SAMPLES / "ohm_law.csv")
    assistant.analyze(session, "U_В", "I_мА", 1, 0.95, None)
    markdown, path = assistant.make_report(session, Options(lab=LAB_1), "Опыт при 22 °C")
    assert "Вывод" in markdown, "в отчете нет выводов"
    return path


def scenario_speech(assistant):
    out = assistant.settings.output_dir
    audio = synthesize(
        "Какова цель лабораторной работы номер один?", assistant.settings.tts_voice, out
    )
    assert audio, "синтез речи недоступен (нужен интернет)"
    text = transcribe(audio, assistant.settings)
    assert "цель" in text.lower(), f"распознано: {text}"
    return f"синтез: {audio}; распознано: {text}"


SCENARIOS = [
    ("Модель отвечает", scenario_chat, False),
    ("Диспетчер (router_chain)", scenario_router, False),
    ("RAG по методичке", scenario_rag, False),
    ("Агент-аналитик и график", scenario_data_agent, False),
    ("Агент-исследователь (интернет)", scenario_research_agent, True),
    ("Блок-схема", scenario_flowchart, False),
    ("Анализ изображения", scenario_vision, False),
    ("Тренировочная защита", scenario_exam, False),
    ("Контекстное окно и резюме", scenario_context, False),
    ("Отчет DOCX", scenario_report, False),
    ("Синтез и распознавание речи", scenario_speech, True),
]


def run_smoke(assistant: LabAssistant, internet: bool = True) -> list[Result]:
    results = []
    for name, scenario, needs_internet in SCENARIOS:
        if needs_internet and not internet:
            continue
        start = time.perf_counter()
        try:
            details, ok = str(scenario(assistant)), True
        except Exception as exc:
            details, ok = f"{type(exc).__name__}: {exc}", False
        results.append(Result(name, ok, time.perf_counter() - start, details))
        print(f"{'OK ' if ok else 'ERR'} {name:34} {results[-1].seconds:6.1f} с", flush=True)
    return results


def save_log(results: list[Result], settings: Settings) -> Path:
    log_dir = settings.storage_dir / "smoke"
    log_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Smoke-тест: {settings.provider}, модель {settings.chat_model}", ""]
    for r in results:
        lines += [
            f"## {'OK' if r.ok else 'ОШИБКА'}: {r.name} ({r.seconds:.1f} с)",
            "",
            r.details,
            "",
        ]
    path = log_dir / "smoke_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка ассистента на настоящей модели")
    parser.add_argument(
        "--no-internet", action="store_true", help="пропустить поиск в интернете и синтез речи"
    )
    args = parser.parse_args()
    use_system_certificates()
    settings = Settings.from_env()
    print(
        f"Провайдер: {settings.provider}, модель: {settings.chat_model}, "
        f"зрение: {settings.vision_model}, эмбеддинги: {settings.embeddings_provider}\n"
    )
    problems = check_setup(settings)
    if problems:
        print("Исправьте настройки и запустите проверку снова:")
        print("\n".join(f"- {problem}" for problem in problems))
        sys.exit(2)
    assistant = LabAssistant(settings)
    print(f"База знаний: {assistant.kb.describe().splitlines()[0]}\n")
    results = run_smoke(assistant, internet=not args.no_internet)
    passed = sum(r.ok for r in results)
    print(
        f"\nИтого: {passed} из {len(results)} сценариев. Подробности: {save_log(results, settings)}"
    )
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
