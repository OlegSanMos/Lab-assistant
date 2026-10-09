import importlib.util
import os
import re
import socket
import subprocess
import sys
import threading

import pytest

from conftest import PROJECT_DIR

spec = importlib.util.spec_from_file_location("app", PROJECT_DIR / "app.py")
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


@pytest.fixture
def busy_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen()
        yield sock.getsockname()[1]


def test_busy_port_is_skipped(busy_port, monkeypatch, capsys):
    monkeypatch.setenv("GRADIO_SERVER_PORT", str(busy_port))
    port = app.choose_port("127.0.0.1")
    assert busy_port < port < busy_port + app.PORT_ATTEMPTS
    out = capsys.readouterr().out
    assert f"Порт {busy_port} занят" in out and f"http://127.0.0.1:{port}" in out


def test_busy_explicit_port_gives_hint(busy_port):
    with pytest.raises(SystemExit) as stop:
        app.choose_port("127.0.0.1", busy_port)
    hint = str(stop.value)
    assert f"Порт {busy_port} занят другой программой, поэтому ЛабМентор не запущен" in hint
    assert f"-iTCP:{busy_port}" in hint or f"findstr :{busy_port}" in hint
    assert "запустите без --port" in hint
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        free = sock.getsockname()[1]
    assert app.choose_port("127.0.0.1", free) == free


@pytest.fixture
def default_port_busy():
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", app.DEFAULT_PORT))
            sock.listen()
        except OSError:
            pass
        yield app.DEFAULT_PORT


def test_app_starts_on_next_port_when_port_is_busy(default_port_busy):
    env = {
        **{k: v for k, v in os.environ.items() if k != "GRADIO_SERVER_PORT"},
        "LLM_PROVIDER": "openai",
        "OPENAI_API_KEY": "test-key",
        "EMBEDDINGS_PROVIDER": "none",
        "GRADIO_ANALYTICS_ENABLED": "False",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
    }
    lines = []

    def read_until_started(process):
        for line in process.stdout:
            lines.append(line)
            if "Running on local URL" in line:
                return

    with subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=PROJECT_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        reader = threading.Thread(target=read_until_started, args=(process,), daemon=True)
        reader.start()
        reader.join(timeout=120)
        process.terminate()
        reader.join(timeout=30)
    output = "".join(lines)
    started = re.search(r"Running on local URL:\s+http://127\.0\.0\.1:(\d+)", output)
    assert started, output
    assert int(started.group(1)) != default_port_busy
    assert f"Порт {default_port_busy} занят" in output and "Traceback" not in output
