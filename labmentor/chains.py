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

ROUTES = ("manual", "research", "data", "flowchart", "exam", "chat")

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
            "chat: приветствие, благодарность или общий вопрос без источников.",
        ),
        ("human", "Загружена таблица с данными: {has_table}.\nЗапрос: {question}"),
    ]
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            ROLE + "\n\nОтвечай на основе фрагментов методических указаний. Если в них нет ответа, "
            "скажи об этом и подскажи, где искать. Указывай источник в квадратных скобках, "
            "например [lr1_ohm_law.md].\n\n"
            "Резюме предыдущей части диалога: {summary}\n\n"
            "Фрагменты методических указаний:\n"
            "{context}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", ROLE + "\n\nРезюме предыдущей части диалога: {summary}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ]
)

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
            "оформление графика) и ответь на вопрос студента.\n\n"
            "Резюме предыдущей части диалога: {summary}",
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            [
                {"type": "text", "text": "{question}"},
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
            ROLE
            + "\n\nПрограмма уже обработала результаты измерений (статистика, МНК). Сформулируй "
            "выводы для отчета: подтверждает ли эксперимент теоретическую зависимость, чему равна "
            "искомая величина с погрешностью, как она соотносится с номинальным или справочным "
            "значением, какие причины могли вызвать расхождения. Числа не пересчитывай, используй "
            "приведенные результаты.",
        ),
        (
            "human",
            "Лабораторная работа: {lab}\n"
            "Таблица: {table}\n\n"
            "Результаты обработки:\n"
            "{results}\n\n"
            "Фрагменты методических указаний:\n"
            "{context}",
        ),
    ]
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Ты помогаешь студенту оформить отчет по лабораторной работе по требованиям кафедры. "
            "Составь черновик отчета в Markdown с разделами: ## Цель работы, ## Оборудование, ## "
            "Краткие теоретические сведения, ## Ход работы, ## Результаты измерений и их "
            "обработка, ## Выводы. Опирайся на методические указания и результаты обработки, "
            "рассчитанные программой, числа не выдумывай. Если сведений не хватает, оставь "
            "пометку [заполнить]. Пиши кратко, в научном стиле. " + FORMULAS,
        ),
        (
            "human",
            "Работа: {lab}\n\n"
            "Методические указания:\n"
            "{manual}\n\n"
            "Результаты обработки данных:\n"
            "{results}\n\n"
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


def build_flowchart_chain(llm: BaseChatModel) -> Runnable:
    return FLOWCHART_PROMPT | llm | StrOutputParser() | RunnableLambda(parse_flowchart)


def build_insight_chain(llm: BaseChatModel) -> Runnable:
    return INSIGHT_PROMPT | llm | StrOutputParser()


def build_report_chain(llm: BaseChatModel) -> Runnable:
    return REPORT_PROMPT | llm | StrOutputParser()
