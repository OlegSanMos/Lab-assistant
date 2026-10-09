from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from labmentor.assistant import LabAssistant, Options, Reply
from labmentor.config import Settings, use_system_certificates
from labmentor.formulas import latex_to_text
from labmentor.llm import check_setup

GOLDEN = ROOT / "data" / "eval" / "golden_set.json"
SAMPLES = ROOT / "data" / "samples"


def ask(assistant: LabAssistant, session, question: str, image: str | None = None, **opts) -> Reply:
    reply = list(assistant.respond(session, question, Options(**opts), image))[-1]
    if reply.error:
        raise RuntimeError(reply.text)
    return reply


def fact_found(fact: str, text: str) -> bool:
    text = latex_to_text(text).lower().replace("ё", "е")
    return any(variant.strip().lower().replace("ё", "е") in text for variant in fact.split("|"))


def numbers(text: str, allow_integer: bool = False) -> list[float]:
    pattern = r"\d+(?:[.,]\d+)?" if allow_integer else r"\d+[.,]\d+"
    return [float(v.replace(",", ".")) for v in re.findall(pattern, latex_to_text(text))]


def eval_retrieval(assistant: LabAssistant, questions: list[dict], k: int) -> list[dict]:
    rows = []
    for item in questions:
        sources = [doc.metadata["source"] for doc in assistant.kb.search(item["question"], k)]
        rows.append(
            {
                "id": item["id"],
                "ok": item["source"] in sources,
                "top1": sources[:1] == [item["source"]],
                "found": sources,
            }
        )
    return rows


def eval_routes(assistant: LabAssistant, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        session = assistant.new_session()
        if case.get("table"):
            assistant.load_table(session, SAMPLES / "ohm_law.csv")
        route = assistant.choose_route(session, case["question"], Options(), None)
        rows.append(
            {
                "question": case["question"],
                "ok": route == case["route"],
                "expected": case["route"],
                "got": route,
            }
        )
    return rows


def eval_answers(assistant: LabAssistant, questions: list[dict], k: int) -> list[dict]:
    rows = []
    for item in questions:
        try:
            text = ask(
                assistant, assistant.new_session(), item["question"], mode="manual", top_k=k
            ).text
        except Exception as exc:
            text = f"ОШИБКА: {exc}"
        found = [fact for fact in item["facts"] if fact_found(fact, text)]
        rows.append(
            {
                "id": item["id"],
                "recall": len(found) / len(item["facts"]),
                "cited": item["source"] in text,
                "missing": [f for f in item["facts"] if f not in found],
                "answer": text,
            }
        )
    return rows


def eval_calculations(assistant: LabAssistant, cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        session = assistant.new_session()
        if case.get("table"):
            assistant.load_table(session, SAMPLES / case["table"])
        image = str(SAMPLES / case["image"]) if case.get("image") else None
        try:
            text = ask(assistant, session, case["question"], image=image, mode=case["mode"]).text
        except Exception as exc:
            text = f"ОШИБКА: {exc}"
        low, high = case["range"]
        values = numbers(text, case.get("allow_integer", False))
        hits = [v for v in values if low <= v <= high]
        rows.append(
            {
                "name": case["name"],
                "ok": bool(hits),
                "values": hits,
                "expected": f"{low}...{high} {case['unit']}",
                "answer": text,
            }
        )
    return rows


def summary(report: dict) -> list[str]:
    lines = []
    if "retrieval" in report:
        rows = report["retrieval"]
        lines.append(
            f"Поиск: нужная методичка первой (hit@1) {sum(r['top1'] for r in rows)} "
            f"из {len(rows)}, среди найденных (hit@k) {sum(r['ok'] for r in rows)} из {len(rows)}"
        )
    if "routes" in report:
        rows = report["routes"]
        lines.append(f"Точность диспетчера: {sum(r['ok'] for r in rows)} из {len(rows)}")
    if "answers" in report:
        rows = report["answers"]
        recall = sum(r["recall"] for r in rows) / len(rows) * 100
        complete = sum(r["recall"] == 1 for r in rows)
        lines.append(
            f"Ключевые факты в ответах: {recall:.0f} % (все факты в {complete} из {len(rows)})"
        )
        lines.append(
            f"Ссылка на методичку в ответе: {sum(r['cited'] for r in rows)} из {len(rows)}"
        )
    if "calculations" in report:
        rows = report["calculations"]
        lines.append(f"Расчеты в допуске: {sum(r['ok'] for r in rows)} из {len(rows)}")
    return lines


def save_report(report: dict, settings: Settings) -> Path:
    lines = [f"# Оценка ответов: {settings.provider}, модель {settings.chat_model}", ""]
    lines += [f"- {line}" for line in summary(report)] + [""]
    for row in report.get("retrieval", []):
        if not row["ok"]:
            lines.append(f"Поиск не нашел методичку для {row['id']}: найдено {row['found']}")
    for row in report.get("routes", []):
        if not row["ok"]:
            lines.append(
                f"Диспетчер: «{row['question']}» -> {row['got']} (ожидалось {row['expected']})"
            )
    for row in report.get("answers", []):
        lines += [
            "",
            f"## {row['id']}: факты {row['recall']:.0%}, источник "
            f"{'указан' if row['cited'] else 'не указан'}",
        ]
        if row["missing"]:
            lines.append(f"Не найдены факты: {', '.join(row['missing'])}")
        lines += ["", row["answer"]]
    for row in report.get("calculations", []):
        lines += [
            "",
            f"## {row['name']}: {'в допуске' if row['ok'] else 'НЕ в допуске'} "
            f"(ожидалось {row['expected']}, найдено {row['values']})",
            "",
            row["answer"],
        ]
    out = settings.storage_dir / "eval"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def evaluate(
    assistant: LabAssistant, golden: dict, k: int = 4, retrieval_only: bool = False
) -> dict:
    report = {"retrieval": eval_retrieval(assistant, golden["questions"], k)}
    if not retrieval_only:
        report["routes"] = eval_routes(assistant, golden["routes"])
        report["answers"] = eval_answers(assistant, golden["questions"], k)
        report["calculations"] = eval_calculations(assistant, golden["calculations"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка ответов ассистента на эталонном наборе")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="проверить только поиск по методичкам (без модели)",
    )
    parser.add_argument("--top-k", type=int, default=4, help="число фрагментов в контексте")
    args = parser.parse_args()
    use_system_certificates()
    settings = Settings.from_env()
    if not args.retrieval_only and (problems := check_setup(settings)):
        print("Исправьте настройки:\n" + "\n".join(f"- {problem}" for problem in problems))
        sys.exit(2)
    assistant = LabAssistant(settings)
    print(
        f"Модель: {settings.provider} / {settings.chat_model}; "
        f"{assistant.kb.describe().splitlines()[0]}\n"
    )
    report = evaluate(
        assistant, json.loads(GOLDEN.read_text(encoding="utf-8")), args.top_k, args.retrieval_only
    )
    print("\n".join(summary(report)))
    print(f"\nПодробный отчет с ответами модели: {save_report(report, settings)}")


if __name__ == "__main__":
    main()
