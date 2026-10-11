import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from docx import Document as WordDocument
from langchain_core.runnables import RunnableLambda
from PIL import Image

import labmentor.assistant as assistant_module
from conftest import SCAN_TEXT, ScriptedChatModel, course_responder
from labmentor import knowledge
from labmentor.assistant import LabAssistant, Options
from labmentor.config import Settings
from labmentor.knowledge import is_text_file, load_file
from labmentor.ui import context_outputs

SLIDE = (
    '<p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" '
    'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"><p:cSld><p:spTree><p:sp>'
    "<p:txBody><a:p><a:r><a:t>Слайд {0}: </a:t></a:r><a:r><a:t>текст</a:t></a:r></a:p>"
    "</p:txBody></p:sp></p:spTree></p:cSld></p:sld>"
)
CONTENT = (
    '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
    'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"><office:body><office:text>'
    "<text:h>Лабораторная работа 8</text:h><text:p>Цель: <text:span>изучить</text:span> триггер"
    "</text:p></office:text></office:body></office:document-content>"
)


def text_of(path) -> str:
    return "\n".join(doc.page_content for doc in load_file(path))


def make_zip(path: Path, files: dict[str, str]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


def make_docx(path: Path, *paragraphs: str) -> Path:
    document = WordDocument()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
    return path


def picture(path: Path, color: str = "red") -> str:
    Image.new("RGB", (40, 30), color).save(path)
    return str(path)


def test_text_files_in_any_encoding(tmp_path):
    phrase = "Цель работы: измерить коэффициент усиления"
    for encoding in ("utf-8", "utf-8-sig", "cp1251", "utf-16"):
        path = tmp_path / f"notes_{encoding}.txt"
        path.write_bytes(phrase.encode(encoding))
        assert text_of(path) == phrase
    netlist = tmp_path / "amp.cir"
    netlist.write_text("* Усилитель\nR1 in out 10k\n.ac dec 10 1 1meg\n.end\n", encoding="cp1251")
    assert is_text_file(netlist) and "R1 in out 10k" in text_of(netlist)
    assert "Усилитель" in text_of(netlist)
    binary = tmp_path / "data.bin"
    binary.write_bytes(bytes(range(256)) * 4)
    assert not is_text_file(binary)
    with pytest.raises(ValueError, match="формат .bin не поддерживается"):
        load_file(binary)


def test_slides_open_document_and_web_page(tmp_path):
    pptx = make_zip(
        tmp_path / "lecture.pptx",
        {f"ppt/slides/slide{n}.xml": SLIDE.format(n) for n in (2, 10, 1)},
    )
    assert text_of(pptx) == (
        "Слайд 1\nСлайд 1: текст\n\nСлайд 2\nСлайд 2: текст\n\nСлайд 3\nСлайд 10: текст"
    )
    odt = make_zip(tmp_path / "lab8.odt", {"content.xml": CONTENT})
    assert text_of(odt) == "Лабораторная работа 8\nЦель: изучить триггер"
    page = tmp_path / "article.html"
    page.write_text(
        "<html><head><style>p {color: red}</style><script>var x = 1</script></head><body>"
        "<h1>Стабилитрон</h1><p>Напряжение &laquo;стабилизации&raquo;</p></body></html>",
        encoding="utf-8",
    )
    assert text_of(page) == "Стабилитрон\n\nНапряжение «стабилизации»"


def test_rtf_and_doc_read_with_textutil_on_mac(monkeypatch, tmp_path):
    calls = []

    def textutil(command, capture_output, timeout):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="Текст из RTF".encode())

    monkeypatch.setattr(
        knowledge.shutil, "which", lambda name: "/usr/bin/textutil" if name == "textutil" else None
    )
    monkeypatch.setattr(knowledge.subprocess, "run", textutil)
    rtf = tmp_path / "notes.rtf"
    rtf.write_text("{\\rtf1 text}")
    assert text_of(rtf) == "Текст из RTF"
    assert calls == [["textutil", "-convert", "txt", "-stdout", str(rtf)]]
    monkeypatch.setattr(knowledge.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="сохраните файл как DOCX или PDF"):
        load_file(tmp_path / "old.doc")


@pytest.mark.skipif(not shutil.which("soffice"), reason="LibreOffice не установлен")
def test_rtf_and_odt_made_by_libreoffice(tmp_path):
    source = make_docx(tmp_path / "source.docx", "Цель работы: изучить RS-триггер")
    for target in ("rtf", "odt"):
        command = ["soffice", f"-env:UserInstallation={(tmp_path / 'profile').as_uri()}"]
        command += ["--headless", "--convert-to", target, "--outdir", str(tmp_path), str(source)]
        subprocess.run(command, capture_output=True, timeout=180, check=True)
        assert "Цель работы: изучить RS-триггер" in text_of(tmp_path / f"source.{target}")


def test_heic_photo_is_converted_or_explained(monkeypatch, assistant, tmp_path):
    heic = tmp_path / "IMG_0001.HEIC"
    heic.write_bytes(b"heic")

    def sips(command, capture_output, timeout, check):
        Image.new("RGB", (10, 10)).save(command[-1], "JPEG")

    monkeypatch.setattr(
        assistant_module.shutil, "which", lambda name: "/usr/bin/sips" if name == "sips" else None
    )
    monkeypatch.setattr(assistant_module.subprocess, "run", sips)
    got = assistant.ingest(assistant.new_session(), [str(heic)])
    assert got.images[0].endswith(".jpg") and Path(got.images[0]).exists()
    monkeypatch.setattr(assistant_module.shutil, "which", lambda name: None)
    monkeypatch.setitem(sys.modules, "pillow_heif", None)
    got = assistant.ingest(assistant.new_session(), [str(heic)])
    assert got.images == [] and "сохраните его как JPEG" in got.notes[0]


def test_several_photos_are_analyzed_and_remembered(assistant, scripted_llm, tmp_path):
    session = assistant.new_session()
    got = assistant.ingest(
        session, [picture(tmp_path / "a.png"), picture(tmp_path / "b.jpg", "blue")]
    )
    assert len(got.images) == 2 and [f.kind for f in session.files] == ["изображение"] * 2
    reply = list(assistant.respond(session, "Сравни схемы", Options(), got.images))[-1]
    parts = scripted_llm.calls[-1][-1].content
    assert reply.route == "vision" and [p["type"] for p in parts].count("image_url") == 2
    reply = list(assistant.respond(session, "А что на фото справа?", Options()))[-1]
    parts = scripted_llm.calls[-1][-1].content
    assert reply.route == "vision" and [p["type"] for p in parts].count("image_url") == 2
    assert session.memory.messages[0].content == "[Загружены файлы: a.png, b.jpg] Сравни схемы"


def test_scanned_pdf_is_read_by_vision_model_once(settings, scripted_llm, tmp_path):
    scan = tmp_path / "scan.pdf"
    Image.new("RGB", (200, 280), "white").save(scan)
    assistant = LabAssistant(settings, llm_factory=lambda **kwargs: scripted_llm)
    session = assistant.new_session()
    got = assistant.ingest(session, [str(scan)], Options())
    assert "scan.pdf: 1 фрагм., скан распознан моделью" in got.notes[0]
    assert got.documents == ["scan.pdf"]
    docs = assistant.kb.search("емкость конденсатора", 2, only={"scan.pdf"})
    assert SCAN_TEXT in docs[0].page_content
    calls = len(scripted_llm.calls)
    restarted = LabAssistant(settings, llm_factory=lambda **kwargs: scripted_llm)
    assert "scan.pdf: 1 фрагм." in restarted.kb_status and len(scripted_llm.calls) == calls


def test_new_document_is_read_and_kept_in_dialog_memory(tmp_path):
    settings = Settings(
        provider="openai", openai_api_key="k", embeddings_provider="none", storage_dir=tmp_path
    )
    llm = ScriptedChatModel(responder=course_responder)
    assistant = LabAssistant(settings, llm_factory=lambda **kwargs: llm)
    lab6 = make_docx(
        tmp_path / "lab6_opamp.docx",
        "Лабораторная работа 6. Исследование инвертирующего усилителя на ОУ",
        "Цель работы: измерить коэффициент усиления инвертирующего усилителя.",
        "Оборудование: генератор сигналов, осциллограф, источник питания ±15 В.",
    )
    session = assistant.new_session()
    got = assistant.ingest(session, [str(lab6)], Options())
    assert got.documents == ["lab6_opamp.docx"]

    def first_source(question):
        reply = list(assistant.respond(session, question, Options()))[-1]
        return reply.route, reply.sources[0]

    assert first_source("О чем этот файл?") == (
        "manual",
        "Лабораторная работа 6. Исследование инвертирующего усилителя на ОУ (фрагмент 1)",
    )
    assert first_source("Какое оборудование нужно?")[1].startswith("Лабораторная работа 6.")
    assert first_source("Какова цель работы 1.2?")[1].startswith("Лабораторная работа 1.2.")
    assert session.memory.messages[0].content == (
        "[Загружены файлы: lab6_opamp.docx] О чем этот файл?"
    )
    assert "Файлы в памяти диалога:** lab6_opamp.docx" in context_outputs(session)[0]
    assert "lab6_opamp.docx (документ): 1 фрагм. в базе знаний" in session.files_text()


def test_router_never_answers_without_the_new_document(assistant, monkeypatch, tmp_path):
    monkeypatch.setattr(
        assistant_module.chains,
        "build_router_chain",
        lambda llm: RunnableLambda(lambda inputs: "chat"),
    )
    session = assistant.new_session()
    note = tmp_path / "notes.txt"
    note.write_text("Заметки к работе 3", encoding="utf-8")
    assistant.ingest(session, [str(note)])
    assert assistant.choose_route(session, "Перескажи", Options()) == "manual"
    session.new_files = []
    assert assistant.choose_route(session, "Привет!", Options()) == "chat"
    assert assistant.choose_route(session, "Что в загруженном файле?", Options()) == "manual"


def test_uploads_survive_restart_and_can_be_forgotten(settings, scripted_llm, tmp_path):
    assistant = LabAssistant(settings, llm_factory=lambda **kwargs: scripted_llm)
    original = assistant.kb.titles["lr1_ohm_law.md"]
    changed = tmp_path / "lr1_ohm_law.md"
    changed.write_text("# Закон Ома, новая редакция\nТекст", encoding="utf-8")
    lab6 = make_docx(tmp_path / "lab6.docx", "Лабораторная работа 6. Операционный усилитель")
    assert assistant.add_uploads([changed, lab6]) == [
        "lr1_ohm_law.md: 1 фрагм.",
        "lab6.docx: 1 фрагм.",
    ]
    restarted = LabAssistant(settings, llm_factory=lambda **kwargs: scripted_llm)
    assert restarted.kb.titles["lr1_ohm_law.md"] == "Закон Ома, новая редакция"
    assert "Лабораторная работа 6. Операционный усилитель" in restarted.lab_titles()
    assert restarted.forget_uploads() == 2 and not any(settings.uploads_dir.iterdir())
    assert (
        restarted.kb.titles["lr1_ohm_law.md"] == original and "lab6.docx" not in restarted.kb.titles
    )


def test_unreadable_files_are_explained(assistant, tmp_path):
    old_table = tmp_path / "old.xls"
    old_table.write_bytes(b"\xd0\xcf\x11\xe0")
    archive = tmp_path / "photos.zip"
    archive.write_bytes(b"PK\x03\x04\x00\x00")
    notes = assistant.ingest(assistant.new_session(), [str(old_table), str(archive)]).notes
    assert "сохраните ее как XLSX или CSV" in notes[0]
    assert "photos.zip: не удалось прочитать (формат .zip не поддерживается)" in notes[1]
