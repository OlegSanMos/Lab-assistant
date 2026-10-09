import importlib.util
import json
import sys

from conftest import PROJECT_DIR

spec = importlib.util.spec_from_file_location("evaluate", PROJECT_DIR / "scripts" / "evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
sys.modules["evaluate"] = evaluate
spec.loader.exec_module(evaluate)
GOLDEN = json.loads(evaluate.GOLDEN.read_text(encoding="utf-8"))


def test_fact_and_number_matching():
    assert evaluate.fact_found("вольт-ампер|ВАХ", "Строим ВАХ резистора")
    assert evaluate.fact_found("Последовательн", "амперметр включают ПОСЛЕДОВАТЕЛЬНО")
    assert not evaluate.fact_found("параллельн", "амперметр включают последовательно")
    assert evaluate.numbers("R = 101,5 Ом, k = 9.85, n = 14") == [101.5, 9.85]
    assert evaluate.numbers("τ ≈ 1 с", allow_integer=True) == [1.0]
    latex = r"$R = \frac{1}{k} = 101{,}5\ \text{Ом}$, $P \le 0{,}5\ \text{Вт}$"
    assert evaluate.fact_found("1/k|1 / k", latex) and evaluate.fact_found("0,5 Вт", latex)
    assert evaluate.numbers(latex) == [101.5, 0.5]


def test_golden_set_is_consistent():
    manuals = {path.name for path in (PROJECT_DIR / "data" / "manuals").iterdir()}
    assert len(GOLDEN["questions"]) == 15 and len(GOLDEN["routes"]) == 12
    assert all(item["source"] in manuals and item["facts"] for item in GOLDEN["questions"])


def test_retrieval_finds_expected_manual_first(assistant):
    rows = evaluate.eval_retrieval(assistant, GOLDEN["questions"], k=4)
    assert all(row["top1"] for row in rows), [row for row in rows if not row["top1"]]


def test_full_evaluation_runs_and_saves_report(assistant):
    report = evaluate.evaluate(assistant, GOLDEN, k=4)
    assert set(report) == {"retrieval", "routes", "answers", "calculations"}
    assert len(report["answers"]) == 15 and len(report["calculations"]) == 3
    goal = next(row for row in report["answers"] if row["id"] == "lr1-goal")
    assert goal["recall"] > 0 and goal["cited"]
    lines = evaluate.summary(report)
    assert lines[0].startswith("Поиск: нужная методичка первой (hit@1) 15 из 15")
    text = evaluate.save_report(report, assistant.settings).read_text(encoding="utf-8")
    assert "Точность диспетчера" in text and "## lr1-goal" in text
