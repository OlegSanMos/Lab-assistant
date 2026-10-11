from __future__ import annotations

from pathlib import Path

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel

from .chains import ROLE
from .data_analysis import DataWorkspace
from .knowledge import KnowledgeBase
from .tools import (
    calculator,
    make_data_tools,
    make_manual_search_tool,
    web_search,
    wikipedia_search,
)

RESEARCH_PROMPT = ROLE + (
    "\n\n"
    "Ты агент-исследователь. Для ответа используй инструменты: search_lab_manuals (методические "
    "указания), web_search и wikipedia_search (интернет), calculator (расчеты). Сначала ищи в "
    "методичках, справочные данные проверяй в интернете. В конце ответа перечисли источники: имена "
    "файлов или ссылки."
)

DATA_PROMPT = ROLE + (
    "\n\n"
    "Ты агент-аналитик экспериментальных данных. Работай с таблицей студента только через "
    "инструменты: describe_data, column_statistics, fit_dependency, add_computed_column, "
    "calculator. Сначала вызови describe_data. Никогда не выдумывай числа: все значения бери из "
    "результатов инструментов. В ответе приведи расчетные формулы, результат с погрешностью и "
    "доверительной вероятностью и краткую физическую интерпретацию."
)


def context_prompt(files: str, summary: str) -> str:
    return (
        f"\n\nФайлы, загруженные студентом в этом диалоге:\n{files}\n\n"
        f"Резюме предыдущей части диалога: {summary}"
    )


def build_research_agent(
    llm: BaseChatModel,
    kb: KnowledgeBase,
    top_k: int,
    allow_web: bool,
    summary: str,
    files: str = "нет",
    focus: dict | None = None,
):
    tools = [make_manual_search_tool(kb, top_k, **(focus or {})), calculator]
    if allow_web:
        tools += [web_search, wikipedia_search]
    prompt = RESEARCH_PROMPT + context_prompt(files, summary)
    return create_agent(llm, tools, system_prompt=prompt, name="research_agent")


def build_data_agent(
    llm: BaseChatModel, ws: DataWorkspace, output_dir: Path, summary: str, files: str = "нет"
):
    prompt = DATA_PROMPT + context_prompt(files, summary)
    return create_agent(
        llm, make_data_tools(ws, output_dir), system_prompt=prompt, name="data_agent"
    )
