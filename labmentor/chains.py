from __future__ import annotations

import json
import re
from operator import itemgetter

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough
from langchain_core.utils.json import parse_json_markdown

from .knowledge import format_docs

FORMULAS = (
    "Формулы записывай в LaTeX: внутри строки между знаками $, отдельной формулой на своей строке "
    "между $$, например $R = U / I$. Русские обозначения и единицы измерения внутри формул "
    "оформляй командой \\text."
)

ROLE = (
    "Ты ЛабМентор, виртуальный ассистент лабораторного практикума по инженерным дисциплинам "
    "(электротехника, сопротивление материалов, физика). Ты помогаешь студенту подготовиться к "
    "лабораторной работе, обработать результаты измерений, оформить отчет и подготовиться к "
    "защите. Отвечай по-русски, ясно и по существу, используй формулы и единицы СИ. Объясняй ход "
    "рассуждений, чтобы студент понимал решение. Если данных недостаточно, честно скажи об этом.\n"
    + FORMULAS
)

ROUTES = ("manual", "research", "data", "flowchart", "exam", "vision", "chat")

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты диспетчер мультиагентного ассистента лабораторного практикума. Определи, кто "
            "должен обработать запрос студента, и ответь одним словом из списка:\n"
            "manual: вопрос по методическим указаниям (цель, теория, оборудование, порядок "
            "выполнения, формулы, требования к отчету, техника безопасности);\n"
            "research: нужны справочные данные или сведения из интернета (константы, свойства "
            "материалов, стандарты, характеристики приборов);\n"
            "data: обработка таблицы с результатами измерений (статистика, погрешности, МНК, "
            "график);\n"
            "flowchart: просьба нарисовать схему или блок-схему порядка выполнения работы;\n"
            "exam: просьба проверить знания, задать контрольные вопросы, подготовить к защите;\n"
            "vision: вопрос о фото, схеме, осциллограмме, графике или видео, которые уже "
            "присланы;\n"
            "chat: приветствие, благодарность или общий вопрос без источников.\n"
            "Вопрос о загруженном студентом файле (методичка, статья, заметки) отдавай manual.",
        ),
        (
            "human",
            "Загружена таблица с данными: {has_table}.\n"
            "Файлы, загруженные студентом в диалоге:\n{files}\n"
            "Запрос: {question}",
        ),
    ]
).partial(files="нет")

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROLE + "\n\nОтвечай на основе фрагментов методических указаний и файлов студента. "
            "Если в них нет ответа, скажи об этом и подскажи, где искать. Указывай источник в "
            "квадратных скобках, например [lab_01_diod-stab.md]. Если во фрагментах есть рисунок, "
            "который помогает ответить, вставь его отдельной строкой в том же виде, что во "
            "фрагменте: ![подпись](путь), а на остальные нужные рисунки ссылайся по номеру "
            "(рис. 8), программа покажет их в чате.\n"
            "Пиши конкретно по фрагментам и не заменяй их общими словами вроде «обычно в "
            "методических указаниях приводится». Если спрашивают, о чем работа или что в ней "
            "делать, опиши цель и суть работы, оборудование, затем по порядку все пункты задания "
            "из методички: что собрать, какие режимы и значения установить, что измерить и какие "
            "графики построить, что сделать в программе моделирования и что включить в отчет. Не "
            "повторяй одни и те же предложения.\n\n"
            "Файлы, загруженные студентом в этом диалоге:\n{files}\n\n"
            "Резюме предыдущей части диалога: {summary}\n\n"
            "Фрагменты методических указаний:\n"
            "{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
).partial(files="нет")

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROLE + "\n\nФайлы, загруженные студентом в этом диалоге:\n{files}\n\n"
            "Резюме предыдущей части диалога: {summary}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
).partial(files="нет")

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты сжимаешь историю диалога студента с ассистентом лабораторного практикума. Обнови "
            "резюме, добавив в него новые реплики. Сохрани главное: номер и тему работы, "
            "полученные результаты и числа, вопросы студента и данные ему рекомендации. Не более "
            "120 слов, без вступления.",
        ),
        ("human", "Текущее резюме: {summary}\n\nНовые реплики:\n{dialog}\n\nОбновленное резюме:"),
    ]
)

QUIZ_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты преподаватель, принимающий защиту лабораторных работ по инженерным дисциплинам. "
            "Составь {n} контрольных вопросов разной сложности строго по материалу методических "
            "указаний: теория, порядок выполнения, обработка результатов и погрешности, техника "
            "безопасности. К каждому вопросу дай краткий эталонный ответ (1-3 предложения). Верни "
            "только JSON: "
            '{{"questions": [{{"question": "...", "answer": "..."}}]}}',
        ),
        ("human", "Запрос студента: {topic}\n\nМетодические указания:\n{context}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты экзаменатор на защите лабораторной работы. Сравни ответ студента с эталоном и "
            "оцени его по шкале от 0 до 10. Засчитывай верные по смыслу ответы, даже если они "
            "сформулированы иначе. "
            'Верни только JSON: {{"score": <целое от 0 до 10>, "feedback": "<2-3 предложения: что '
            'верно, что упущено>"}}',
        ),
        ("human", "Вопрос: {question}\nЭталонный ответ: {reference}\nОтвет студента: {answer}"),
    ]
)

VISION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROLE
            + "\n\nСтудент прислал изображение: фото лабораторной установки, электрическую схему, "
            "осциллограмму, график или фрагмент отчета. Опиши, что изображено, сними числовые "
            "значения с учетом масштаба и цены деления, найди ошибки (подключение приборов, "
            "оформление графика) и ответь на вопрос студента. Если изображений несколько, "
            "называй их по порядку. Если прислана раскадровка видео, рассказывай о ходе опыта по "
            "времени, ссылаясь на номера и время кадров, и учитывай распознанную речь из видео."
            "\n\n"
            "Файлы, загруженные студентом в этом диалоге:\n{files}\n\n"
            "Резюме предыдущей части диалога: {summary}",
        ),
        MessagesPlaceholder("history"),
        MessagesPlaceholder("request"),
    ]
).partial(files="нет")

OCR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты распознаешь текст на изображении страницы документа. Перепиши весь текст страницы "
            "дословно, сохраняя порядок абзацев, заголовки и таблицы. Формулы записывай в LaTeX. "
            "Не добавляй своих пояснений.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Распознай текст этой страницы."},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        ),
    ]
)

CAPTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты подписываешь рисунки для отчета по лабораторной работе. По изображению придумай "
            "короткое название рисунка из 3...8 слов, например «Схема лабораторной установки» или "
            "«Осциллограмма выходного напряжения». Ответь только названием, без кавычек и точки.",
        ),
        (
            "human",
            [
                {"type": "text", "text": "Подпиши этот рисунок."},
                {"type": "image_url", "image_url": {"url": "{image_url}"}},
            ],
        ),
    ]
)

FLOWCHART_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "По фрагментам методических указаний составь алгоритм, о котором просит студент "
            "(обычно порядок выполнения работы или обработки результатов). Каждый шаг не длиннее "
            "15 слов, "
            'всего от 4 до 10 шагов. Верни только JSON: {{"title": "<название схемы>", "steps": '
            '["<шаг>"]}}',
        ),
        ("human", "Запрос: {question}\n\nФрагменты методических указаний:\n{context}"),
    ]
)

INSIGHT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROLE + "\n\nСформулируй выводы для отчета по лабораторной работе «{lab}» на основе "
            "таблицы измерений студента и результатов, которые программа уже получила "
            "(статистика, МНК). Выводы должны относиться именно к этой работе: опирайся на ее "
            "цели, расчетные формулы и требования к отчету из фрагментов методички.\n"
            "1. Назови исследованную зависимость и скажи, подтверждает ли ее эксперимент (вид "
            "зависимости, коэффициент детерминации).\n"
            "2. Рассчитай величины, которые требует эта работа: запиши формулу из методички, "
            "подставь числа из результатов обработки и таблицы и получи значение с погрешностью, "
            "например параметр модели из углового коэффициента прямой.\n"
            "3. Сравни результат с теоретическим, номинальным или справочным значением из "
            "методички.\n"
            "4. Назови возможные причины расхождений.\n"
            "Коэффициенты МНК, средние значения и доверительные границы бери из результатов "
            "обработки, не пересчитывай их. Если для расчета не хватает данных (нужного столбца "
            "или условий опыта), прямо скажи, чего не хватает. Если таблица не похожа на "
            "измерения этой работы, предупреди об этом. Числа в выводах пиши с десятичной "
            "запятой.",
        ),
        (
            "human",
            "Лабораторная работа: {lab}\n"
            "Таблица: {table}\n\n"
            "Данные таблицы:\n"
            "{data}\n\n"
            "Результаты обработки:\n"
            "{results}\n\n"
            "Фрагменты методических указаний этой работы:\n"
            "{context}",
        ),
    ]
).partial(data="нет")

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты помогаешь студенту оформить отчет по лабораторной работе по требованиям кафедры "
            "(ГОСТ 2.105). Составь черновик отчета в Markdown с разделами: ## Цель работы, ## "
            "Оборудование, ## Краткие теоретические сведения, ## Ход работы, ## Результаты "
            "измерений и их обработка, ## Выводы. Опирайся на методические указания и результаты "
            "обработки, рассчитанные программой, числа не выдумывай. Если сведений не хватает, "
            "оставь пометку [заполнить]. Пиши кратко, в научном стиле.\n"
            "Правила оформления кафедры:\n"
            "- заголовки пиши с прописной буквы, без точки в конце;\n"
            "- таблицы нумеруй по порядку, до таблицы сошлись на нее в тексте (в таблице 2), над "
            "таблицей пиши строку «Таблица 2 – Наименование»; в заголовках граф после названия "
            "величины через запятую ставь единицы измерения, например «U, В»;\n"
            "- нужные рисунки из методических указаний (схемы, характеристики) вставляй "
            "отдельной строкой в том же виде, что в методичке: ![Рис. 1. Наименование](путь), "
            "номера и подписи рисунков программа поставит сама;\n"
            "- выносные формулы нумеруй по порядку командой \\tag{{1}}, ссылайся на них номером "
            "в скобках, например (1); формулы, идущие подряд без текста, разделяй запятой;\n"
            "- символы формулы поясняй сразу под ней, каждый с новой строки в порядке появления "
            "в формуле, первая строка начинается словом «где» без двоеточия, например «где $U$ – "
            "напряжение, В;»;\n"
            "- элементы электрических схем обозначай буквой и номером по порядку без пропусков, "
            "слева направо и сверху вниз: резисторы R1, R2, конденсаторы C1, индуктивности L1, "
            "источники ЭДС E1, источники тока I1.\n" + FORMULAS,
        ),
        (
            "human",
            "Работа: {lab}\n\n"
            "Методические указания:\n"
            "{manual}\n\n"
            "Результаты обработки данных:\n"
            "{results}\n\n"
            "Таблицы и рисунки программы: {inserts}\n\n"
            "Заметки студента: {notes}",
        ),
    ]
)


def parse_route(text: str) -> str:
    words = re.findall(r"[a-z]+", text.lower())
    return next((w for w in words if w in ROUTES), "manual")


def parse_json(text: str) -> dict | list:
    try:
        return parse_json_markdown(text)
    except Exception:
        match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.S)
        if not match:
            raise ValueError("Модель не вернула JSON") from None
        return json.loads(match.group(1))


def parse_quiz(text: str) -> list[dict]:
    try:
        data = parse_json(text)
        items = data.get("questions", []) if isinstance(data, dict) else data
        questions = [
            {
                "question": str(i.get("question") or i.get("вопрос", "")).strip(),
                "answer": str(i.get("answer") or i.get("ответ", "")).strip(),
            }
            for i in items
            if isinstance(i, dict)
        ]
    except Exception:
        questions = [
            {"question": q.strip(), "answer": ""}
            for q in re.findall(r"^\s*\d+[.)]\s*(.+\?)\s*$", text, flags=re.M)
        ]
    questions = [q for q in questions if q["question"]]
    if not questions:
        raise ValueError("Не удалось сформировать вопросы")
    return questions


def parse_grade(text: str) -> dict:
    try:
        data = parse_json(text)
        score = int(round(float(data.get("score", 0))))
        feedback = str(data.get("feedback", "")).strip()
    except Exception:
        match = re.search(r"\d+", text)
        score, feedback = (int(match.group()) if match else 0), text.strip()
    return {"score": max(0, min(10, score)), "feedback": feedback}


def parse_caption(text: str) -> str:
    line = next((line for line in text.splitlines() if line.strip()), "")
    return line.strip().strip("«»\"'*. ")[:120]


def parse_flowchart(text: str) -> dict:
    data = parse_json(text)
    steps = [str(step).strip() for step in data.get("steps", []) if str(step).strip()][:12]
    if not steps:
        raise ValueError("Модель не вернула шаги алгоритма")
    return {"title": str(data.get("title") or "Алгоритм выполнения работы"), "steps": steps}


def build_router_chain(llm: BaseChatModel) -> Runnable:
    return ROUTER_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_route)


def build_rag_chain(llm: BaseChatModel, retriever: BaseRetriever) -> Runnable:
    answer = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    retrieve = RunnablePassthrough.assign(docs=itemgetter("question") | retriever)
    return retrieve | RunnablePassthrough.assign(answer=answer)


def build_chat_chain(llm: BaseChatModel) -> Runnable:
    return CHAT_PROMPT | llm | StrOutputParser()


def build_summary_chain(llm: BaseChatModel) -> Runnable:
    return SUMMARY_PROMPT | llm | StrOutputParser()


def build_quiz_chain(llm: BaseChatModel) -> Runnable:
    return QUIZ_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_quiz)


def build_grade_chain(llm: BaseChatModel) -> Runnable:
    return GRADE_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_grade)


def build_vision_chain(llm: BaseChatModel) -> Runnable:
    return VISION_PROMPT | llm | StrOutputParser()


def build_ocr_chain(llm: BaseChatModel) -> Runnable:
    return OCR_PROMPT | llm | StrOutputParser()


def build_caption_chain(llm: BaseChatModel) -> Runnable:
    return CAPTION_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_caption)


def build_flowchart_chain(llm: BaseChatModel) -> Runnable:
    return FLOWCHART_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_flowchart)


def build_insight_chain(llm: BaseChatModel) -> Runnable:
    return INSIGHT_PROMPT | llm | StrOutputParser()


def build_report_chain(llm: BaseChatModel) -> Runnable:
    return REPORT_PROMPT | llm | StrOutputParser()
