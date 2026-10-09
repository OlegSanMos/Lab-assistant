import argparse
import os
import socket
import sys

from labmentor.assistant import LabAssistant
from labmentor.config import Settings, use_system_certificates
from labmentor.llm import check_setup
from labmentor.speech import preload
from labmentor.ui import build_ui, launch_options

DEFAULT_PORT = 7860
PORT_ATTEMPTS = 20


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET) as sock:
        if os.name == "posix":
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except socket.gaierror:
            raise
        except OSError:
            return False
    return True


def choose_port(host: str, port: int | None = None) -> int:
    start = port if port is not None else int(os.getenv("GRADIO_SERVER_PORT", DEFAULT_PORT))
    candidates = [start] if port is not None else range(start, start + PORT_ATTEMPTS)
    for candidate in candidates:
        if port_is_free(host, candidate):
            if candidate != start:
                url_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
                print(
                    f"Порт {start} занят другой программой (возможно, ЛабМентор уже запущен "
                    f"в другом окне терминала).\nЛабМентор будет запущен на свободном порту "
                    f"{candidate}: http://{url_host}:{candidate}"
                )
            return candidate
    if port is not None:
        busy = f"Порт {start} занят другой программой"
        advice = "запустите без --port: свободный порт будет выбран автоматически"
    else:
        busy = f"Порты {start}...{candidates[-1]} заняты другими программами"
        advice = f"укажите другой порт: python app.py --port {candidates[-1] + 1}"
    find = (
        f"netstat -ano | findstr :{start}"
        if sys.platform == "win32"
        else f"lsof -nP -iTCP:{start} -sTCP:LISTEN"
    )
    raise SystemExit(
        f"{busy}, поэтому ЛабМентор не запущен.\n"
        f"Какая программа занимает порт {start}: {find}\n"
        f"Остановите ее (Ctrl+C в ее окне терминала) или {advice}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ЛабМентор: виртуальный ассистент лабораторного практикума"
    )
    parser.add_argument("--host", default="127.0.0.1", help="адрес веб-сервера")
    parser.add_argument(
        "--port", type=int, help="порт веб-сервера (по умолчанию первый свободный от 7860)"
    )
    parser.add_argument("--share", action="store_true", help="создать публичную ссылку Gradio")
    args = parser.parse_args()

    use_system_certificates()
    settings = Settings.from_env()
    port = choose_port(args.host, args.port)
    for problem in check_setup(settings):
        print(f"Внимание: {problem}")
    assistant = LabAssistant(settings)
    print(f"Провайдер LLM: {settings.provider}, модель: {settings.chat_model}")
    print(f"База знаний: {assistant.kb_status}")
    preload(settings)
    demo = build_ui(assistant)
    demo.queue(default_concurrency_limit=4).launch(
        server_name=args.host, server_port=port, share=args.share, **launch_options(settings)
    )


if __name__ == "__main__":
    main()
