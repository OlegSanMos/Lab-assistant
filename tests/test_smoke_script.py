import importlib.util
import sys

from conftest import PROJECT_DIR

spec = importlib.util.spec_from_file_location(
    "smoke_test", PROJECT_DIR / "scripts" / "smoke_test.py"
)
smoke = importlib.util.module_from_spec(spec)
sys.modules["smoke_test"] = smoke
spec.loader.exec_module(smoke)


def test_smoke_scenarios_pass_with_scripted_model(assistant, tmp_path, capsys):
    results = smoke.run_smoke(assistant, internet=False)
    failed = [(r.name, r.details) for r in results if not r.ok]
    assert failed == []
    assert len(results) == 9
    log = smoke.save_log(results, assistant.settings)
    assert "OK: RAG по методичке" in log.read_text(encoding="utf-8")
    assert "Агент-аналитик и график" in capsys.readouterr().out
