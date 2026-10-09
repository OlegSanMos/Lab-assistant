from __future__ import annotations

import base64
import io
import json
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
from .knowledge import DOC_EXTENSIONS, KnowledgeBase, format_docs
from .llm import create_chat_model, create_embeddings
from .memory import ConversationMemory, plan_context
from .report import export_report_docx
from .speech import AUDIO_EXTENSIONS, SpeechError, synthesize, transcribe

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
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
    image: str | None = None
    speech: str = ""
    speech_error: str = ""


@dataclass
class Session:
    memory: ConversationMemory
    data: DataWorkspace = field(default_factory=DataWorkspace)
    exam: ExamState | None = None
    context_plan: dict = field(default_factory=dict)


@dataclass
class Reply:
    route: str = ""
    text: str = ""
    steps: list[dict] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    error: bool = False


def image_to_data_url(path: str, max_side: int = 1600) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


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

    def ingest(self, session: Session, paths: list[str]) -> Attachments:
        result, documents = Attachments(), []
        for path in map(Path, paths or []):
            ext = path.suffix.lower()
            try:
                if ext in AUDIO_EXTENSIONS:
                    result.speech = f"{result.speech} {transcribe(path, self.settings)}".strip()
                elif ext in IMAGE_EXTENSIONS:
                    result.image = str(path)
                elif ext in TABLE_EXTENSIONS:
                    result.notes.append(self.load_table(session, path))
                elif ext in DOC_EXTENSIONS:
                    documents.append(path)
                else:
                    result.notes.append(f"Файл {path.name}: формат не поддерживается")
            except SpeechError as exc:
                result.speech_error = str(exc)
            except Exception as exc:
                result.notes.append(f"Файл {path.name}: ошибка обработки ({exc})")
        if documents:
            result.notes.append("База знаний пополнена: " + self.kb.add_files(documents))
        return result

    def load_table(self, session: Session, path: str | Path) -> str:
        df = load_table(path)
        session.data = DataWorkspace(df=df, name=Path(path).name)
        return (
            f"Загружена таблица {Path(path).name}: {len(df)} строк, столбцы {', '.join(df.columns)}"
        )

    def respond(
        self, session: Session, question: str, options: Options, image: str | None = None
    ) -> Iterator[Reply]:
        reply = Reply()
        self.apply_context(session, options)
        try:
            reply.route = self.choose_route(session, question, options, image)
            reply.steps.append(
                {"title": "🧭 Диспетчер", "content": f"Исполнитель: {MODES[reply.route]}"}
            )
            yield reply
            handler = getattr(self, f"_handle_{reply.route}")
            yield from handler(session, question, options, reply, image)
        except Exception as exc:
            reply.error = True
            reply.text = f"⚠️ Не удалось получить ответ: {exc}"
        if not reply.error and reply.text:
            session.memory.add_turn(question or "[изображение]", reply.text)
        yield reply

    def choose_route(
        self, session: Session, question: str, options: Options, image: str | None
    ) -> str:
        if session.exam is not None and options.mode in ("auto", "exam"):
            return "exam"
        if options.mode != "auto":
            return options.mode
        if image:
            return "vision"
        router = chains.build_router_chain(self.llm(options, temperature=0))
        return router.invoke(
            {"question": question, "has_table": "да" if session.data.df is not None else "нет"}
        )

    def _context(self, session: Session) -> dict:
        return {"history": session.memory.window(), "summary": session.memory.summary_text()}

    def _handle_manual(self, session, question, options, reply, image):
        chain = chains.build_rag_chain(self.llm(options), self.kb.as_retriever(options.top_k))
        for chunk in chain.stream({**self._context(session), "question": question}):
            if "docs" in chunk:
                reply.sources = source_list(chunk["docs"])
            if "answer" in chunk:
                reply.text += chunk["answer"]
                yield reply

    def _handle_chat(self, session, question, options, reply, image):
        for token in chains.build_chat_chain(self.llm(options)).stream(
            {**self._context(session), "question": question}
        ):
            reply.text += token
            yield reply

    def _handle_vision(self, session, question, options, reply, image):
        if not image:
            reply.text = "Прикрепите изображение: фото установки, схему, осциллограмму или график."
            yield reply
            return
        inputs = {
            **self._context(session),
            "image_url": image_to_data_url(image),
            "question": question or "Что изображено? Сними значения и проверь, нет ли ошибок.",
        }
        for token in chains.build_vision_chain(self.llm(options, vision=True)).stream(inputs):
            reply.text += token
            yield reply

    def _handle_research(self, session, question, options, reply, image):
        agent = build_research_agent(
            self.llm(options),
            self.kb,
            options.top_k,
            options.allow_web,
            session.memory.summary_text(),
        )
        yield from self._run_agent(agent, session, question, reply)

    def _handle_data(self, session, question, options, reply, image):
        if session.data.df is None:
            reply.text = (
                "Сначала загрузите таблицу измерений (CSV или XLSX): прикрепите файл к сообщению "
                "или откройте вкладку «Обработка данных»."
            )
            yield reply
            return
        agent = build_data_agent(
            self.llm(options), session.data, self.settings.output_dir, session.memory.summary_text()
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

    def _handle_flowchart(self, session, question, options, reply, image):
        docs = self.kb.search(f"{options.lab} {question}", max(options.top_k, 5))
        plan = chains.build_flowchart_chain(self.llm(options, temperature=0)).invoke(
            {"question": question, "context": format_docs(docs)}
        )
        reply.images.append(draw_flowchart(plan["title"], plan["steps"], self.settings.output_dir))
        reply.text = f"**{plan['title']}**\n\n" + "\n".join(
            f"{i}. {s}" for i, s in enumerate(plan["steps"], 1)
        )
        reply.sources = source_list(docs)
        yield reply

    def _handle_exam(self, session, question, options, reply, image):
        if session.exam is None:
            topic = f"{options.lab} {question}".strip()
            docs = self.kb.search(f"{topic} контрольные вопросы", max(options.top_k, 6))
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
            }
        )
        path = export_report_docx(
            markdown, options.lab, self.settings.output_dir, session.data.df, session.data.last_plot
        )
        return markdown, path
