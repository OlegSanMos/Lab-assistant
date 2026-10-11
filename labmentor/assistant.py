from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from PIL import Image

from . import chains
from .agents import build_data_agent, build_research_agent
from .config import DEFAULT_MODELS, Settings
from .data_analysis import (
    TABLE_EXTENSIONS,
    DataWorkspace,
    column_stats,
    fit_polynomial,
    format_stats,
    load_table,
)
from .graphics import draw_flowchart, plot_fit
from .knowledge import KnowledgeBase, format_docs, load_file
from .llm import create_chat_model, create_embeddings
from .memory import ConversationMemory, plan_context
from .report import export_report_docx, report_inserts
from .speech import AUDIO_EXTENSIONS, SpeechError, synthesize, transcribe

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".heic",
    ".heif",
}
DOCUMENT, TABLE, IMAGE = "документ", "таблица", "изображение"
FILE_WORDS = re.compile(r"файл|документ|загруж|прикреп|вложен", re.IGNORECASE)
OCR_PAGES = 10
SUMMARY_REQUEST = (
    "Прочитай загруженные файлы ({names}) и кратко перескажи их содержание: о чем они и что в них "
    "главное."
)
MODES = {
    "auto": "Авто (диспетчер)",
    "manual": "Консультант по методичкам (RAG)",
    "research": "Агент-исследователь (интернет)",
    "data": "Агент-аналитик данных",
    "vision": "Анализ изображения",
    "flowchart": "Блок-схема хода работы",
    "exam": "Экзаменатор (защита работы)",
    "chat": "Свободный диалог",
}
STOP_WORDS = {"стоп", "хватит", "закончить", "завершить", "stop"}


@dataclass
class Options:
    mode: str = "auto"
    provider: str = ""
    model: str = ""
    temperature: float = 0.3
    num_ctx: int = 8192
    history_tokens: int = 2000
    use_summary: bool = True
    top_k: int = 4
    allow_web: bool = True
    exam_questions: int = 5
    lab: str = ""


@dataclass
class ExamState:
    questions: list[dict]
    index: int = 0
    scores: list[int] = field(default_factory=list)


@dataclass
class Attachments:
    notes: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    documents: list[str] = field(default_factory=list)
    speech: str = ""
    speech_error: str = ""


@dataclass
class FileRecord:
    name: str
    kind: str
    info: str
    turn: int = 0


@dataclass
class Session:
    memory: ConversationMemory
    data: DataWorkspace = field(default_factory=DataWorkspace)
    exam: ExamState | None = None
    context_plan: dict = field(default_factory=dict)
    files: list[FileRecord] = field(default_factory=list)
    new_files: list[FileRecord] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    turn: int = 0

    def files_text(self) -> str:
        if not self.files:
            return "нет"
        return "\n".join(f"- {f.name} ({f.kind}): {f.info}" for f in self.files[-12:])

    def documents(self) -> list[FileRecord]:
        return [f for f in self.files if f.kind == DOCUMENT]

    def forget_files(self) -> None:
        self.files, self.new_files, self.images = [], [], []


@dataclass
class Reply:
    route: str = ""
    text: str = ""
    steps: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: bool = False


def image_to_data_url(source: str | Image.Image, max_side: int = 1600) -> str:
    image = source if isinstance(source, Image.Image) else Image.open(source)
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def file_digest(path: Path) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            md5.update(block)
    return md5.hexdigest()


def open_image(path: Path, out_dir: Path) -> str:
    if path.suffix.lower() in (".heic", ".heif"):
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{path.stem}_{file_digest(path)[:8]}.jpg"
        if shutil.which("sips"):
            command = ["sips", "-s", "format", "jpeg", str(path), "--out", str(target)]
            subprocess.run(command, capture_output=True, timeout=60, check=True)
        else:
            try:
                from pillow_heif import register_heif_opener
            except ImportError:
                raise ValueError(
                    "фото HEIC открывается в macOS, на другом компьютере сохраните его как JPEG"
                ) from None
            register_heif_opener()
            Image.open(path).convert("RGB").save(target, "JPEG", quality=92)
        path = target
    with Image.open(path) as image:
        image.verify()
    return str(path)


def source_list(docs: list[Document]) -> list[str]:
    names = [
        f"{d.metadata.get('title', d.metadata.get('source'))} (фрагмент "
        f"{d.metadata.get('chunk', '?')})"
        for d in docs
    ]
    return list(dict.fromkeys(names))


class LabAssistant:
    def __init__(
        self, settings: Settings, llm_factory: Callable[..., BaseChatModel] | None = None
    ) -> None:
        self.settings = settings
        self._llm_factory = llm_factory or (lambda **kw: create_chat_model(settings, **kw))
        self._llms: dict[tuple, BaseChatModel] = {}
        self.kb = KnowledgeBase(
            create_embeddings(settings), settings.chunk_size, settings.chunk_overlap
        )
        self.kb_status = self.kb.add_directory(settings.manuals_dir)
        uploads = self.load_uploads()
        if uploads:
            self.kb_status += f"; загруженные файлы: {uploads}"

    def new_session(self) -> Session:
        return Session(memory=ConversationMemory(self.settings.history_tokens))

    def llm(
        self, options: Options, vision: bool = False, temperature: float | None = None
    ) -> BaseChatModel:
        provider = options.provider or self.settings.provider
        own = provider == self.settings.provider
        if vision:
            model = self.settings.vision_model if own else DEFAULT_MODELS[provider]["vision"]
        else:
            model = options.model or (
                self.settings.chat_model if own else DEFAULT_MODELS[provider]["chat"]
            )
        temperature = options.temperature if temperature is None else temperature
        key = (provider, model, temperature, options.num_ctx)
        if key not in self._llms:
            self._llms[key] = self._llm_factory(
                provider=provider, model=model, temperature=temperature, num_ctx=options.num_ctx
            )
        return self._llms[key]

    def apply_context(self, session: Session, options: Options) -> dict:
        plan = plan_context(
            options.num_ctx,
            options.history_tokens,
            options.top_k,
            self.settings.chunk_size,
            self.settings.max_answer_tokens,
        )
        session.memory.max_tokens, session.memory.use_summary = plan["history"], options.use_summary
        session.context_plan = plan
        return plan

    def lab_titles(self) -> list[str]:
        return sorted(self.kb.titles.values())

    def ingest(
        self, session: Session, paths: list[str], options: Options | None = None
    ) -> Attachments:
        result, documents = Attachments(), []
        session.turn += 1
        session.new_files = []
        for path in map(Path, paths or []):
            ext = path.suffix.lower()
            try:
                if ext in AUDIO_EXTENSIONS:
                    result.speech = f"{result.speech} {transcribe(path, self.settings)}".strip()
                elif ext in IMAGE_EXTENSIONS:
                    result.images.append(open_image(path, self.settings.output_dir))
                    self._remember(session, FileRecord(path.name, IMAGE, "отправлено модели"))
                elif ext in TABLE_EXTENSIONS:
                    note = self.load_table(session, path)
                    result.notes.append(note)
                    info = note.split(":", 1)[1].strip()
                    self._remember(session, FileRecord(path.name, TABLE, info))
                else:
                    documents.append(path)
            except SpeechError as exc:
                result.speech_error = str(exc)
            except Exception as exc:
                result.notes.append(f"Файл {path.name}: ошибка обработки ({exc})")
        if documents:
            lines = self.add_uploads(documents, options or Options())
            result.notes.append("База знаний пополнена: " + "; ".join(lines))
            for path in documents:
                count = self.kb.count(path.name)
                if count:
                    result.documents.append(path.name)
                    info = f"{count} фрагм. в базе знаний, «{self.kb.titles[path.name]}»"
                    self._remember(session, FileRecord(path.name, DOCUMENT, info))
        if result.images:
            session.images = result.images
        return result

    def _remember(self, session: Session, record: FileRecord) -> None:
        record.turn = session.turn
        session.files = [f for f in session.files if f.name != record.name] + [record]
        session.new_files.append(record)

    def read_file(self, path: Path, options: Options | None = None) -> list[Document]:
        docs = load_file(path)
        if docs or path.suffix.lower() != ".pdf":
            return docs
        return self.read_scan(path, options)

    def read_scan(self, path: Path, options: Options | None) -> list[Document]:
        cache = self.settings.cache_dir / "ocr" / f"{file_digest(path)}.json"
        if cache.exists():
            pages = json.loads(cache.read_text(encoding="utf-8"))
        elif options is None:
            return []
        else:
            from pypdf import PdfReader

            chain = chains.build_ocr_chain(self.llm(options, vision=True, temperature=0))
            pages = []
            for page in PdfReader(str(path)).pages[:OCR_PAGES]:
                pictures = list(page.images)
                if not pictures:
                    pages.append("")
                    continue
                picture = max(pictures, key=lambda item: len(item.data))
                url = image_to_data_url(picture.image, 2000)
                pages.append(chain.invoke({"image_url": url}).strip())
            if any(pages):
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
        return [
            Document(page_content=text, metadata={"source": path.name, "page": number, "ocr": 1})
            for number, text in enumerate(pages, start=1)
            if text.strip()
        ]

    def add_uploads(self, paths: list[str | Path], options: Options | None = None) -> list[str]:
        lines = []
        for path in map(Path, paths):
            try:
                docs = self.read_file(path, options)
            except Exception as exc:
                lines.append(f"{path.name}: не удалось прочитать ({exc})")
                continue
            if not docs and path.suffix.lower() == ".pdf":
                lines.append(
                    f"{path.name}: в PDF нет текста, а модель для распознавания недоступна"
                )
                continue
            line = self.kb.add_documents({path.name: docs})
            if docs:
                self.save_upload(path)
                if docs[0].metadata.get("ocr"):
                    line += f", скан распознан моделью (не больше {OCR_PAGES} страниц)"
            lines.append(line)
        return lines

    def save_upload(self, path: Path) -> Path:
        folder = self.settings.uploads_dir
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / path.name
        if path.resolve() != target.resolve():
            shutil.copy2(path, target)
        return target

    def load_uploads(self) -> str:
        folder = self.settings.uploads_dir
        if not folder.is_dir():
            return ""
        documents, errors = {}, []
        for path in sorted(p for p in folder.iterdir() if p.is_file()):
            try:
                documents[path.name] = self.read_file(path)
            except Exception as exc:
                errors.append(f"{path.name}: ошибка чтения ({exc})")
        report = self.kb.add_documents(documents) if documents else ""
        return "; ".join(part for part in [report, *errors] if part)

    def forget_uploads(self) -> int:
        folder = self.settings.uploads_dir
        names = {p.name for p in folder.iterdir() if p.is_file()} if folder.is_dir() else set()
        for name in names:
            (folder / name).unlink()
        if names:
            self.kb.remove(names)
            manuals = [self.settings.manuals_dir / name for name in sorted(names)]
            restored = [path for path in manuals if path.is_file()]
            if restored:
                self.kb.add_files(restored)
        return len(names)

    def load_table(self, session: Session, path: str | Path) -> str:
        df = load_table(path)
        session.data = DataWorkspace(df=df, name=Path(path).name)
        return (
            f"Загружена таблица {Path(path).name}: {len(df)} строк, столбцы {', '.join(df.columns)}"
        )

    def respond(
        self,
        session: Session,
        question: str,
        options: Options,
        images: str | list[str] | None = None,
    ) -> Iterator[Reply]:
        images = [images] if isinstance(images, str) else list(images or [])
        reply = Reply()
        self.apply_context(session, options)
        try:
            reply.route = self.choose_route(session, question, options, images)
            reply.steps.append(
                {"title": "🧭 Диспетчер", "content": f"Исполнитель: {MODES[reply.route]}"}
            )
            yield reply
            handler = getattr(self, f"_handle_{reply.route}")
            yield from handler(session, question, options, reply, images)
        except Exception as exc:
            reply.error = True
            reply.text = f"⚠️ Не удалось получить ответ: {exc}"
        if not reply.error and reply.text:
            asked = question or ("[изображение]" if images else "")
            if session.new_files:
                names = ", ".join(f.name for f in session.new_files)
                asked = f"[Загружены файлы: {names}] {asked}".strip()
            session.memory.add_turn(asked, reply.text)
        session.new_files = []
        yield reply

    def choose_route(
        self,
        session: Session,
        question: str,
        options: Options,
        images: str | list[str] | None = None,
    ) -> str:
        if session.exam is not None and options.mode in ("auto", "exam"):
            return "exam"
        if options.mode != "auto":
            return options.mode
        if images:
            return "vision"
        router = chains.build_router_chain(self.llm(options, temperature=0))
        route = router.invoke(
            {
                "question": question,
                "has_table": "да" if session.data.df is not None else "нет",
                "files": session.files_text(),
            }
        )
        if route == "vision" and not session.images:
            return "manual"
        fresh = any(f.kind == DOCUMENT for f in session.new_files)
        if route == "chat" and (fresh or FILE_WORDS.search(question) and session.documents()):
            return "manual"
        return route

    def _focus(self, session: Session, question: str) -> dict:
        fresh = {f.name for f in session.new_files if f.kind == DOCUMENT}
        if fresh:
            return {"only": fresh}
        documents = session.documents()
        if not documents:
            return {}
        if FILE_WORDS.search(question):
            last = max(f.turn for f in documents)
            return {"only": {f.name for f in documents if f.turn == last}}
        return {"prefer": {f.name for f in documents}}

    def _context(self, session: Session) -> dict:
        return {
            "history": session.memory.window(),
            "summary": session.memory.summary_text(),
            "files": session.files_text(),
        }

    def _handle_manual(self, session, question, options, reply, images):
        retriever = self.kb.as_retriever(options.top_k, **self._focus(session, question))
        chain = chains.build_rag_chain(self.llm(options), retriever)
        for chunk in chain.stream({**self._context(session), "question": question}):
            if "docs" in chunk:
                reply.sources = source_list(chunk["docs"])
            if "answer" in chunk:
                reply.text += chunk["answer"]
                yield reply

    def _handle_chat(self, session, question, options, reply, images):
        for token in chains.build_chat_chain(self.llm(options)).stream(
            {**self._context(session), "question": question}
        ):
            reply.text += token
            yield reply

    def _handle_vision(self, session, question, options, reply, images):
        images = images or session.images
        if not images:
            reply.text = "Прикрепите изображение: фото установки, схему, осциллограмму или график."
            yield reply
            return
        text = question or "Что изображено? Сними значения и проверь, нет ли ошибок."
        content = [{"type": "text", "text": text}]
        content += [
            {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            for path in images[:4]
        ]
        inputs = {**self._context(session), "request": [HumanMessage(content=content)]}
        for token in chains.build_vision_chain(self.llm(options, vision=True)).stream(inputs):
            reply.text += token
            yield reply

    def _handle_research(self, session, question, options, reply, images):
        agent = build_research_agent(
            self.llm(options),
            self.kb,
            options.top_k,
            options.allow_web,
            session.memory.summary_text(),
            session.files_text(),
            self._focus(session, question),
        )
        yield from self._run_agent(agent, session, question, reply)

    def _handle_data(self, session, question, options, reply, images):
        if session.data.df is None:
            reply.text = (
                "Сначала загрузите таблицу измерений (CSV или XLSX): прикрепите файл к сообщению "
                "или откройте вкладку «Обработка данных»."
            )
            yield reply
            return
        agent = build_data_agent(
            self.llm(options),
            session.data,
            self.settings.output_dir,
            session.memory.summary_text(),
            session.files_text(),
        )
        yield from self._run_agent(agent, session, question, reply)

    def _run_agent(self, agent, session: Session, question: str, reply: Reply) -> Iterator[Reply]:
        messages = [*session.memory.window(), HumanMessage(content=question)]
        first_image, steps = len(session.data.images), {}
        for update in agent.stream(
            {"messages": messages}, {"recursion_limit": 20}, stream_mode="updates"
        ):
            for output in update.values():
                new_messages = output.get("messages", []) if isinstance(output, dict) else []
                for message in new_messages:
                    if isinstance(message, AIMessage):
                        for call in message.tool_calls:
                            steps[call["id"]] = {
                                "title": f"🔧 Инструмент {call['name']}",
                                "content": "Аргументы: "
                                + json.dumps(call["args"], ensure_ascii=False),
                            }
                            reply.steps.append(steps[call["id"]])
                        if message.text and not message.tool_calls:
                            reply.text = message.text
                    elif isinstance(message, ToolMessage) and message.tool_call_id in steps:
                        result = message.text
                        steps[message.tool_call_id]["content"] += "\n\nРезультат: " + (
                            result[:700] + "..." if len(result) > 700 else result
                        )
            reply.images = session.data.images[first_image:]
            yield reply

    def _handle_flowchart(self, session, question, options, reply, images):
        focus = self._focus(session, question)
        docs = self.kb.search(f"{options.lab} {question}", max(options.top_k, 5), **focus)
        plan = chains.build_flowchart_chain(self.llm(options, temperature=0)).invoke(
            {"question": question, "context": format_docs(docs)}
        )
        reply.images.append(draw_flowchart(plan["title"], plan["steps"], self.settings.output_dir))
        reply.text = f"**{plan['title']}**\n\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(plan["steps"], 1)
        )
        reply.sources = source_list(docs)
        yield reply

    def _handle_exam(self, session, question, options, reply, images):
        if session.exam is None:
            topic = f"{options.lab} {question}".strip()
            focus = self._focus(session, question)
            docs = self.kb.search(f"{topic} контрольные вопросы", max(options.top_k, 6), **focus)
            questions = chains.build_quiz_chain(self.llm(options)).invoke(
                {"n": options.exam_questions, "topic": topic, "context": format_docs(docs)}
            )
            session.exam = ExamState(questions[: options.exam_questions])
            reply.sources = source_list(docs)
            reply.text = (
                f"Начинаем подготовку к защите: {len(session.exam.questions)} вопросов. "
                "Отвечайте своими словами, для досрочного завершения напишите «стоп».\n\n"
                + self._exam_question(session.exam)
            )
            yield reply
            return
        exam = session.exam
        if question.strip().lower().strip(".!") in STOP_WORDS:
            reply.text = self._exam_finish(session)
            yield reply
            return
        current = exam.questions[exam.index]
        grade = chains.build_grade_chain(self.llm(options, temperature=0)).invoke(
            {
                "question": current["question"],
                "answer": question,
                "reference": current["answer"] or "эталон не задан, оцени ответ по существу",
            }
        )
        exam.scores.append(grade["score"])
        exam.index += 1
        parts = [f"Оценка за ответ: **{grade['score']}/10**. {grade['feedback']}"]
        if current["answer"]:
            parts.append(f"Эталонный ответ: {current['answer']}")
        parts.append(
            self._exam_question(exam)
            if exam.index < len(exam.questions)
            else self._exam_finish(session)
        )
        reply.text = "\n\n".join(parts)
        yield reply

    @staticmethod
    def _exam_question(exam: ExamState) -> str:
        question = exam.questions[exam.index]["question"]
        return f"**Вопрос {exam.index + 1} из {len(exam.questions)}.** {question}"

    @staticmethod
    def _exam_finish(session: Session) -> str:
        exam, session.exam = session.exam, None
        if not exam.scores:
            return "Тренировочная защита завершена до первого ответа."
        total, maximum = sum(exam.scores), 10 * len(exam.scores)
        percent = total / maximum * 100
        mark = (
            "отлично"
            if percent >= 85
            else "хорошо"
            if percent >= 70
            else "удовлетворительно"
            if percent >= 50
            else "неудовлетворительно"
        )
        text = (
            f"**Итог тренировочной защиты:** {total} из {maximum} баллов ({percent:.0f} %), "
            f"оценка «{mark}»."
        )
        weak = [exam.questions[i]["question"] for i, score in enumerate(exam.scores) if score < 6]
        if weak:
            text += "\n\nРекомендуется повторить:\n" + "\n".join(f"- {q}" for q in weak)
        return text

    def update_memory(self, session: Session, options: Options) -> bool:
        try:
            return session.memory.update_summary(
                chains.build_summary_chain(self.llm(options, temperature=0))
            )
        except Exception:
            return False

    def speak(self, text: str, voice: str) -> str | None:
        return synthesize(text, voice, self.settings.output_dir)

    def analyze(
        self, session: Session, x: str, y: str, degree: int, p: float, stat_column: str | None
    ) -> tuple[str, str]:
        df = session.data.df
        if df is None:
            raise ValueError("Таблица не загружена")
        fit = fit_polynomial(df, x, y, degree, p)
        plot = plot_fit(df, fit, self.settings.output_dir)
        results = [fit.describe()]
        if stat_column:
            results.append(format_stats(column_stats(df, stat_column, p)))
        session.data.last_plot, session.data.last_results = plot, "\n\n".join(results)
        return plot, session.data.last_results

    def explain_results(self, session: Session, options: Options) -> str:
        if not session.data.last_results:
            raise ValueError("Сначала выполните обработку данных")
        docs = self.kb.search(f"{options.lab} обработка результатов", options.top_k)
        return chains.build_insight_chain(self.llm(options)).invoke(
            {
                "lab": options.lab or "не указана",
                "table": session.data.name,
                "results": session.data.last_results,
                "context": format_docs(docs),
            }
        )

    def manual_text(self, title: str) -> str:
        chunks = [c.page_content for c in self.kb.chunks if c.metadata.get("title") == title]
        return "\n".join(chunks)[:12000] if chunks else format_docs(self.kb.search(title, 6))

    def make_report(self, session: Session, options: Options, notes: str) -> tuple[str, str]:
        markdown = chains.build_report_chain(self.llm(options)).invoke(
            {
                "lab": options.lab,
                "manual": self.manual_text(options.lab),
                "notes": notes or "нет",
                "results": session.data.last_results or "обработка данных не выполнялась",
                "inserts": report_inserts(session.data.df, session.data.last_plot),
            }
        )
        path = export_report_docx(
            markdown, options.lab, self.settings.output_dir, session.data.df, session.data.last_plot
        )
        return markdown, path
