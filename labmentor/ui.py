from __future__ import annotations

import base64
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import gradio as gr
import pandas as pd

from .assistant import (
    IMAGE_EXTENSIONS,
    MODES,
    SUMMARY_REQUEST,
    VIDEO_REQUEST,
    LabAssistant,
    Options,
    Reply,
    Session,
)
from .config import DEFAULT_MODELS, PROJECT_DIR, Settings
from .data_analysis import add_column, parse_column_spec
from .formulas import LATEX_DELIMITERS, prettify_math
from .knowledge import DOC_EXTENSIONS
from .llm import PROVIDERS
from .report import IMAGE
from .speech import (
    MALE_VOICES,
    VOICES,
    piper_loading,
    preload_piper,
    stt_loading,
    stt_status,
    tts_status,
    voice_note,
)
from .video import is_voice

STATIC = Path(__file__).resolve().parent / "static"
ICONS = {
    "#app-title h2": "flask-conical",
    "#side-title h3": "sliders-horizontal",
    "#acc-model > .label-wrap > span:first-child": "brain",
    "#acc-context > .label-wrap > span:first-child": "app-window",
    "#acc-kb > .label-wrap > span:first-child": "library",
    "#acc-voice > .label-wrap > span:first-child": "audio-lines",
    "#tab-dialog-button": "messages-square",
    "#tab-data-button": "chart-line",
    "#tab-report-button": "file-text",
    "#tab-about-button": "info",
    "#btn-clear": "brush-cleaning",
    "#btn-forget": "trash-2",
    "#btn-check-voice": "volume-2",
    "#btn-add-column": "plus",
    "#btn-analyze": "calculator",
    "#btn-explain": "sparkles",
    "#btn-report": "file-pen",
}
ICON_STYLE = (
    "content:'';display:inline-block;flex-shrink:0;width:1.15em;height:1.15em;"
    "margin-right:.45em;vertical-align:-.2em;background-color:currentColor;"
    "-webkit-mask-position:center;mask-position:center;-webkit-mask-size:contain;"
    "mask-size:contain;-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat"
)
TITLE = "ЛабМентор"
LAB_LABEL = "Текущая лабораторная работа (для выводов, отчета, защиты и схем)"
SUBTITLE = "виртуальный ассистент лабораторного практикума по инженерным дисциплинам"
PLACEHOLDER = (
    "### Здравствуйте! Я ЛабМентор\n"
    "Помогу разобраться в методичке, обработать измерения, построить график, оформить отчет и "
    "подготовиться к защите лабораторной работы.\n\n"
    "Пишите вопрос, прикрепляйте фото установки или осциллограммы, видео опыта, таблицу "
    "измерений (CSV, XLSX), методичку (PDF, DOCX) или задайте вопрос голосом в блоке «Вопрос "
    "голосом» под полем ввода: ответ прозвучит автоматически."
)
EXAMPLES = [
    {"text": "Какова цель работы 1.1 и какие параметры модели диода в ней определяют?"},
    {"text": "Нарисуй блок-схему порядка выполнения работы 1.3"},
    {"text": "Найди в интернете типичное напряжение Эрли транзистора и сравни с методичкой"},
    {"text": "Проверь меня: задай вопросы для защиты работы 1.2"},
]
INPUT_HINT = "Задайте вопрос или прикрепите файл: методичку, таблицу, фото, видео, аудио"
VOICE_HINT = (
    "Запишите вопрос: после остановки записи он отправится сам, а ответ прозвучит автоматически."
)
NO_VOICE = (
    "Озвучивание недоступно: нет связи с Edge TTS и Google TTS, голос Piper не загружен, "
    "системный голос не найден"
)
ABOUT = """
### Назначение
ЛабМентор сопровождает студента на всех этапах лабораторной работы: подготовка по методическим
указаниям, получение справочных данных, анализ фото и видео опыта, обработка измерений
(статистика, коэффициент Стьюдента, МНК, графики), оформление отчета и тренировочная защита.

### Как устроен ассистент
- **Диспетчер** (цепочка router_chain) выбирает исполнителя запроса в режиме «Авто».
- **Консультант по методичкам (RAG)**: гибридный поиск BM25 + векторный, объединение рангов RRF.
- **Агент-исследователь**: методички, интернет (DuckDuckGo), Википедия, калькулятор.
- **Агент-аналитик данных**: описание таблицы, статистика и погрешности, МНК, графики,
  вычисляемые столбцы.
- **Экзаменатор**: генерирует контрольные вопросы и оценивает ответы по шкале 0...10.
- **Контекстное окно**: последние сообщения в пределах бюджета токенов и резюме вытесненной истории.
- **Модальности**: текст, изображения (анализ фото, схем и осциллограмм, генерация графиков
  и блок-схем), видео (раскадровка и распознавание речи из ролика), речь (Whisper и синтез
  голоса).
- **Отчет**: черновик по методичке и требованиям кафедры с таблицей измерений, графиком,
  рисунками из методички и фотографиями студента, экспорт в DOCX.

### Соответствие заданию
| Требование | Как выполнено |
|---|---|
| Веб-приложение или чат-бот | веб-интерфейс Gradio: чат и вкладки обработки данных и отчета |
| Элементы управления | режим работы, модель, температура, контекстное окно, top-k, голос |
| Ввод и генерация текста | диалог с моделью, ответы с формулами и ссылками на источники |
| Еще одна модальность | изображения, речь и видео |
| Внешние данные | интернет, методички и файлы (PDF, DOCX, PPTX и др.), таблицы CSV и XLSX |
| Контекстное окно (LangChain) | последние сообщения по бюджету токенов и резюме истории |
| Не менее двух цепочек | 12 цепочек LCEL |
| RAG, два агента или открытая LLM | все три: RAG по методичкам, два агента, модели Ollama |

### Модели
OpenAI-совместимый API или открытые LLM через Ollama (Qwen2.5, Llama 3.1, Gemma 3).
"""


def icon_css() -> str:
    rules = [",".join(f"{selector}::before" for selector in ICONS) + "{" + ICON_STYLE + "}"]
    for selector, name in ICONS.items():
        data = base64.b64encode((STATIC / "icons" / f"{name}.svg").read_bytes()).decode()
        image = f'url("data:image/svg+xml;base64,{data}")'
        rules.append(f"{selector}::before{{-webkit-mask-image:{image};mask-image:{image}}}")
    return "\n".join(rules)


def fresh_audio(path: str | None):
    return gr.Audio(value=path, playback_position=0)


def make_options(
    mode,
    provider,
    model,
    temperature,
    num_ctx,
    history_tokens,
    use_summary,
    top_k,
    allow_web,
    exam_n,
    lab,
) -> Options:
    return Options(
        mode=mode or "auto",
        provider=provider,
        model=(model or "").strip(),
        temperature=float(temperature),
        num_ctx=int(num_ctx),
        history_tokens=int(history_tokens),
        use_summary=bool(use_summary),
        top_k=int(top_k),
        allow_web=bool(allow_web),
        exam_questions=int(exam_n),
        lab=lab or "",
    )


FILE_ROUTE = "/gradio_api/file="


def served(text: str, find_image: Callable[[str], Path | None], captions: bool = False) -> str:

    def link(match) -> str:
        ref = match.group(2) or match.group(3)
        if ref.startswith(("http://", "https://", "data:")):
            return match.group(0)
        path = find_image(ref)
        if path is None:
            return f"*{match.group(1)}*" if match.group(1).strip() else ""
        image = f"![{match.group(1)}]({FILE_ROUTE}{quote(str(path))})"
        return f"{image}\n\n*{match.group(1)}*" if captions and match.group(1) else image

    return IMAGE.sub(link, text)


def render(reply: Reply, find_image: Callable[[str], Path | None] | None = None) -> list[dict]:
    messages = [
        {
            "role": "assistant",
            "content": prettify_math(step["content"]),
            "metadata": {"title": step["title"], "status": "done"},
        }
        for step in reply.steps
    ]
    text = prettify_math(reply.text) or "Обрабатываю запрос..."
    if find_image is not None:
        text = served(text, find_image)
    messages.append({"role": "assistant", "content": text})
    messages += [{"role": "assistant", "content": {"path": path}} for path in reply.images]
    if reply.sources:
        messages.append(
            {
                "role": "assistant",
                "content": "\n".join(f"- {s}" for s in reply.sources),
                "metadata": {"title": "Использованные фрагменты методичек", "status": "done"},
            }
        )
    return messages


def context_outputs(session: Session) -> tuple[str, str]:
    st, plan = session.memory.stats(), session.context_plan
    percent = min(100, round(st["tokens"] / max(st["limit"], 1) * 100))
    names = ", ".join(f.name for f in session.files) or "нет"
    info = (
        f"**История:** {st['tokens']} из {st['limit']} токенов ({percent} %)  \n"
        f"**Сообщений в окне:** {st['in_window']} из {st['total']}  \n"
        f"**Резюме:** {st['summary_words']} слов  \n"
        f"**Файлы в памяти диалога:** {names}"
    )
    if plan:
        info += (
            f"  \n**Окно модели {plan['num_ctx']}:** промпт ~{plan['system']}, RAG ~{plan['rag']}, "
            f"история {plan['history']}, ответ {plan['answer']}"
        )
    return info, session.memory.summary


def table_updates(session: Session, x=None, y=None, stat=None) -> list:
    df = session.data.df
    if df is None:
        return [gr.skip()] * 4
    columns = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not columns:
        return [df, gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), gr.Dropdown(choices=[])]
    x = x if x in columns else columns[0]
    y = y if y in columns else columns[min(1, len(columns) - 1)]
    stat = stat if stat in columns else None
    return [
        df.round(6),
        gr.Dropdown(choices=columns, value=x),
        gr.Dropdown(choices=columns, value=y),
        gr.Dropdown(choices=columns, value=stat),
    ]


def build_ui(assistant: LabAssistant) -> gr.Blocks:
    settings = assistant.settings
    labs = assistant.lab_titles()

    def make_session() -> Session:
        return assistant.new_session()

    with gr.Blocks(title=f"{TITLE}: {SUBTITLE}", fill_height=True) as demo:
        session = gr.State(make_session)

        with gr.Sidebar(width=360):
            gr.Markdown("### Управление ассистентом", elem_id="side-title")
            mode = gr.Radio(
                [(label, key) for key, label in MODES.items()], value="auto", label="Режим работы"
            )
            lab = gr.Dropdown(
                labs,
                value=None,
                label=LAB_LABEL,
                allow_custom_value=True,
            )
            with gr.Accordion("Языковая модель", open=False, elem_id="acc-model"):
                provider = gr.Dropdown(
                    list(PROVIDERS),
                    value=settings.provider,
                    label="Провайдер (ollama: открытые LLM)",
                )
                model = gr.Textbox(settings.chat_model, label="Модель")
                temperature = gr.Slider(0, 1, settings.temperature, step=0.05, label="Температура")
            with gr.Accordion("Контекстное окно", open=True, elem_id="acc-context"):
                num_ctx = gr.Slider(
                    2048,
                    32768,
                    settings.num_ctx,
                    step=1024,
                    label="Размер контекстного окна модели, токенов",
                )
                history_tokens = gr.Slider(
                    256, 8192, settings.history_tokens, step=128, label="Бюджет токенов истории"
                )
                use_summary = gr.Checkbox(True, label="Сжимать вытесненную историю в резюме")
                context_info = gr.Markdown()
                summary_box = gr.Textbox(
                    label="Резюме вытесненной истории", lines=3, interactive=False
                )
                clear_btn = gr.Button("Очистить диалог", elem_id="btn-clear")
            with gr.Accordion("База знаний (RAG)", open=False, elem_id="acc-kb"):
                top_k = gr.Slider(
                    1, 10, settings.top_k, step=1, label="Фрагментов в контексте (top-k)"
                )
                kb_upload = gr.File(
                    file_count="multiple",
                    file_types=sorted(DOC_EXTENSIONS),
                    label="Добавить методические указания и другие документы",
                )
                kb_info = gr.Markdown(assistant.kb.describe())
                forget_btn = gr.Button("Забыть загруженные файлы", size="sm", elem_id="btn-forget")
            with gr.Accordion("Голос, интернет, защита", open=False, elem_id="acc-voice"):
                speak = gr.Checkbox(False, label="Озвучивать и ответы на вопросы, заданные текстом")
                voices = list(VOICES.items())
                if settings.tts_voice not in VOICES.values():
                    voices.append((settings.tts_voice, settings.tts_voice))
                voice = gr.Radio(voices, value=settings.tts_voice, label="Голос озвучки")
                check_voice = gr.Button("Проверить голос", size="sm", elem_id="btn-check-voice")
                allow_web = gr.Checkbox(True, label="Разрешить поиск в интернете")
                exam_n = gr.Slider(3, 10, 5, step=1, label="Число вопросов на защите")

        gr.Markdown(f"## {TITLE}: {SUBTITLE}", elem_id="app-title")
        with gr.Tabs():
            with gr.Tab("Диалог", elem_id="tab-dialog"):
                chatbot = gr.Chatbot(
                    height=540,
                    placeholder=PLACEHOLDER,
                    examples=EXAMPLES,
                    label="Диалог с ассистентом",
                    feedback_options=None,
                    latex_delimiters=LATEX_DELIMITERS,
                )
                chat_input = gr.MultimodalTextbox(
                    placeholder=INPUT_HINT,
                    sources=["upload"],
                    file_types=None,
                    file_count="multiple",
                    show_label=False,
                )
                with gr.Row(equal_height=True):
                    voice_in = gr.Audio(
                        sources=["microphone"],
                        type="filepath",
                        label="Вопрос голосом",
                        buttons=[],
                        editable=False,
                        elem_id="voice-input",
                        scale=3,
                    )
                    answer_audio = gr.Audio(
                        label="Ответ голосом",
                        autoplay=True,
                        interactive=False,
                        elem_id="voice-answer",
                        scale=2,
                    )
                speech_info = gr.Markdown(elem_id="speech-status")
                speech_timer = gr.Timer(3)
            with gr.Tab("Обработка данных", elem_id="tab-data"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=300):
                        table_file = gr.File(
                            file_types=[".csv", ".tsv", ".xlsx"], label="Таблица измерений"
                        )
                        x_col = gr.Dropdown(label="Аргумент X")
                        y_col = gr.Dropdown(label="Функция Y")
                        degree = gr.Radio(
                            [("Линейная", 1), ("Квадратичная", 2), ("Кубическая", 3)],
                            value=1,
                            label="Аппроксимация МНК",
                        )
                        stat_col = gr.Dropdown(label="Статистика и погрешность столбца")
                        confidence = gr.Radio(
                            [0.9, 0.95, 0.99], value=0.95, label="Доверительная вероятность P"
                        )
                        formula = gr.Textbox(
                            label="Вычисляемый столбец", placeholder="R, Ом = U_В / I_мА * 1000"
                        )
                        add_btn = gr.Button("Добавить столбец", elem_id="btn-add-column")
                        analyze_btn = gr.Button(
                            "Обработать", variant="primary", elem_id="btn-analyze"
                        )
                        data_lab = gr.Dropdown(
                            labs,
                            value=None,
                            label="Лабораторная работа для выводов",
                            info="Выводы рассчитываются по методичке этой работы",
                        )
                        explain_btn = gr.Button(
                            "Сформулировать выводы (LLM)", elem_id="btn-explain"
                        )
                    with gr.Column(scale=2):
                        table_view = gr.Dataframe(label="Данные", interactive=False, max_height=260)
                        plot = gr.Image(label="График", type="filepath", height=380)
                        results = gr.Markdown(latex_delimiters=LATEX_DELIMITERS)
                        insight = gr.Markdown(latex_delimiters=LATEX_DELIMITERS)
            with gr.Tab("Отчет", elem_id="tab-report"):
                gr.Markdown(
                    "Черновик отчета строится по методичке выбранной работы, результатам "
                    "обработки данных и вашим материалам. Рисунки из методички, таблица "
                    "измерений и график вставляются в отчет сами и нумеруются по требованиям "
                    "кафедры."
                )
                with gr.Row():
                    notes = gr.Textbox(
                        lines=5, label="Заметки студента: условия опыта, наблюдения, замечания"
                    )
                    with gr.Column():
                        report_images = gr.File(
                            file_count="multiple",
                            file_types=["image"],
                            label="Фото и рисунки для отчета: установка, схема, осциллограммы",
                        )
                        figure_notes = gr.Textbox(
                            lines=2,
                            label="Подписи к рисункам по одной на строку, иначе их даст модель",
                        )
                report_btn = gr.Button(
                    "Сформировать черновик отчета", variant="primary", elem_id="btn-report"
                )
                report_file = gr.File(label="Отчет в формате DOCX")
                report_md = gr.Markdown(latex_delimiters=LATEX_DELIMITERS)
            with gr.Tab("О проекте", elem_id="tab-about"):
                gr.Markdown(ABOUT)

        option_inputs = [
            mode,
            provider,
            model,
            temperature,
            num_ctx,
            history_tokens,
            use_summary,
            top_k,
            allow_web,
            exam_n,
            lab,
        ]
        data_outputs = [table_view, x_col, y_col, stat_col]

        def dialog(message, history, sess, speak_on, voice_id, values):
            options = make_options(*values)
            assistant.apply_context(sess, options)
            history = list(history or [])
            files = (message or {}).get("files") or []
            attached = [Path(f).name for f in files if not is_voice(f)]
            if files:
                status = "Читаю файлы: " + ", ".join(attached) if attached else "Распознаю речь"
                yield (
                    [*history, {"role": "assistant", "content": f"{status}..."}],
                    "busy",
                    fresh_audio(None),
                    *context_outputs(sess),
                    *[gr.skip()] * 4,
                )
            text = ((message or {}).get("text") or "").strip()
            got = assistant.ingest(sess, files, options)
            tables = (
                table_updates(sess)
                if any(n.startswith("Загружена таблица") for n in got.notes)
                else [gr.skip()] * 4
            )
            question = f"{text} {got.speech}".strip()
            for image in got.images:
                history.append({"role": "user", "content": {"path": image}})
            for name in attached:
                if name in got.videos:
                    history.append({"role": "user", "content": f"Видео: {name}"})
                elif Path(name).suffix.lower() not in IMAGE_EXTENSIONS:
                    history.append({"role": "user", "content": f"Файл: {name}"})
            if text:
                history.append({"role": "user", "content": text})
            if got.speech:
                history.append({"role": "user", "content": f"Голосом: {got.speech}"})
            for note in got.notes:
                history.append(
                    {
                        "role": "assistant",
                        "content": note,
                        "metadata": {"title": "Вложения", "status": "done"},
                    }
                )
            if got.speech_error:
                history.append({"role": "assistant", "content": got.speech_error})
            if not question and got.videos:
                question = VIDEO_REQUEST.format(names=", ".join(got.videos))
            if not question and got.documents:
                question = SUMMARY_REQUEST.format(names=", ".join(got.documents))
            if not question and not got.images:
                audio = None
                if got.speech_error:
                    audio = assistant.speak(
                        "Не удалось распознать вопрос, подробности на экране.", voice_id
                    )
                yield history, "ready", fresh_audio(audio), *context_outputs(sess), *tables
                return
            reply = Reply()
            audio = fresh_audio(None)
            for reply in assistant.respond(sess, question, options, got.images):
                shown = render(reply, assistant.find_image)
                yield history + shown, "busy", audio, *context_outputs(sess), *tables
                audio, tables = gr.skip(), [gr.skip()] * 4
            voiced = (speak_on or bool(got.speech)) and not reply.error
            audio = assistant.speak(reply.text, voice_id) if voiced else None
            yield (
                history + render(reply, assistant.find_image),
                "ready",
                fresh_audio(audio),
                *context_outputs(sess),
                *tables,
            )
            if assistant.update_memory(sess, options):
                yield gr.skip(), None, gr.skip(), *context_outputs(sess), *tables

        def input_state(component, state):
            if state is None:
                return gr.skip()
            return component(value=None, interactive=state == "ready")

        def on_message(message, history, sess, speak_on, voice_id, *values):
            for chat, state, *rest in dialog(message, history, sess, speak_on, voice_id, values):
                yield chat, input_state(gr.MultimodalTextbox, state), *rest

        def on_voice(recording, history, sess, voice_id, *values):
            message = {"text": "", "files": [recording] if recording else []}
            for chat, state, *rest in dialog(message, history, sess, False, voice_id, values):
                yield chat, input_state(gr.Audio, state), *rest

        def on_speech_status():
            status = (
                f"{VOICE_HINT} Распознавание речи: {stt_status(settings)}. "
                f"Озвучивание: {tts_status()}."
            )
            return status, gr.Timer(active=stt_loading(settings) or piper_loading())

        def on_check_voice(voice_id):
            path = assistant.speak("Голос ЛабМентора работает. Задайте вопрос голосом.", voice_id)
            if path is None:
                gr.Warning(NO_VOICE)
            elif voice_note():
                gr.Warning(voice_note())
            return fresh_audio(path), on_speech_status()[0]

        dialog_outputs = [answer_audio, context_info, summary_box, *data_outputs]
        chat_input.submit(
            on_message,
            [chat_input, chatbot, session, speak, voice, *option_inputs],
            [chatbot, chat_input, *dialog_outputs],
        ).then(on_speech_status, None, [speech_info, speech_timer])
        voice_in.stop_recording(
            on_voice,
            [voice_in, chatbot, session, voice, *option_inputs],
            [chatbot, voice_in, *dialog_outputs],
        ).then(on_speech_status, None, [speech_info, speech_timer])
        check_voice.click(on_check_voice, voice, [answer_audio, speech_info])

        def on_voice_choice(voice_id):
            if voice_id in MALE_VOICES:
                preload_piper(True)
            return on_speech_status()

        voice.change(on_voice_choice, voice, [speech_info, speech_timer])
        speech_timer.tick(on_speech_status, None, [speech_info, speech_timer])
        demo.load(on_speech_status, None, [speech_info, speech_timer])

        def on_example(evt: gr.SelectData):
            return {"text": evt.value["text"], "files": []}

        chatbot.example_select(on_example, None, chat_input)

        def on_clear(sess):
            sess.memory.clear()
            sess.exam = None
            sess.forget_files()
            return [], fresh_audio(None), *context_outputs(sess)

        clear_btn.click(on_clear, session, [chatbot, answer_audio, context_info, summary_box])
        chatbot.clear(on_clear, session, [chatbot, answer_audio, context_info, summary_box])

        def on_context_settings(sess, *values):
            assistant.apply_context(sess, make_options(*values))
            return context_outputs(sess)

        for control in (num_ctx, history_tokens, top_k):
            control.release(
                on_context_settings, [session, *option_inputs], [context_info, summary_box]
            )
        use_summary.change(
            on_context_settings, [session, *option_inputs], [context_info, summary_box]
        )
        demo.load(on_context_settings, [session, *option_inputs], [context_info, summary_box])
        provider.change(
            lambda p: settings.chat_model if p == settings.provider else DEFAULT_MODELS[p]["chat"],
            provider,
            model,
        )

        def lab_lists():
            return gr.Dropdown(choices=assistant.lab_titles()), gr.Dropdown(
                choices=assistant.lab_titles()
            )

        def on_kb_upload(files, *values):
            if files:
                gr.Info("; ".join(assistant.add_uploads(files, make_options(*values))))
            return assistant.kb.describe(), *lab_lists()

        kb_upload.upload(on_kb_upload, [kb_upload, *option_inputs], [kb_info, lab, data_lab])

        def on_forget():
            gr.Info(f"Удалено загруженных файлов: {assistant.forget_uploads()}")
            return assistant.kb.describe(), *lab_lists()

        forget_btn.click(on_forget, None, [kb_info, lab, data_lab])
        lab.input(
            lambda title: title if title in assistant.lab_titles() else gr.skip(), lab, data_lab
        )
        data_lab.input(lambda title: title, data_lab, lab)

        def on_table(file, sess):
            if not file:
                return *[gr.skip()] * 4, gr.skip()
            try:
                note = assistant.load_table(sess, file)
            except Exception as exc:
                raise gr.Error(f"Не удалось прочитать таблицу: {exc}") from exc
            return *table_updates(sess), note

        table_file.upload(on_table, [table_file, session], [*data_outputs, results])

        def on_add_column(spec, sess, x, y):
            if sess.data.df is None:
                raise gr.Error("Сначала загрузите таблицу")
            try:
                column = add_column(sess.data.df, *parse_column_spec(spec))
            except Exception as exc:
                raise gr.Error(f"Ошибка в формуле: {exc}") from exc
            return *table_updates(sess, x, y, column), f"Добавлен столбец {column}"

        add_btn.click(on_add_column, [formula, session, x_col, y_col], [*data_outputs, results])

        def on_analyze(sess, x, y, deg, p, stat):
            try:
                return assistant.analyze(sess, x, y, int(deg), float(p), stat or None)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc

        analyze_btn.click(
            on_analyze, [session, x_col, y_col, degree, confidence, stat_col], [plot, results]
        )

        def on_explain(sess, title, *values):
            if not title:
                raise gr.Error("Выберите лабораторную работу, по которой нужны выводы")
            options = make_options(*values)
            options.lab = title
            try:
                text = assistant.explain_results(sess, options)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return f"#### Выводы по работе «{title}»\n\n{prettify_math(text)}"

        explain_btn.click(on_explain, [session, data_lab, *option_inputs], insight)

        def on_report(sess, notes_text, images, captions, *values):
            options = make_options(*values)
            if not options.lab:
                raise gr.Error("Выберите лабораторную работу в боковой панели")
            try:
                figures = assistant.report_figures(images or [], captions, options)
                markdown, path = assistant.make_report(sess, options, notes_text, figures)
            except Exception as exc:
                raise gr.Error(str(exc)) from exc
            return path, served(prettify_math(markdown), assistant.find_image, captions=True)

        report_btn.click(
            on_report,
            [session, notes, report_images, figure_notes, *option_inputs],
            [report_file, report_md],
        )

    return demo


def launch_options(settings: Settings) -> dict:
    font = [gr.themes.LocalFont("IBM Plex Sans"), "Arial", "sans-serif"]
    return {
        "theme": gr.themes.Soft(primary_hue="blue", secondary_hue="sky", font=font),
        "css": icon_css(),
        "favicon_path": str(STATIC / "favicon.png"),
        "allowed_paths": [
            str(settings.output_dir),
            str(PROJECT_DIR / "data"),
            str(settings.manuals_dir),
            str(settings.uploads_dir),
        ],
    }
