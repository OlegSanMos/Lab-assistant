from __future__ import annotations

import json
from pathlib import Path

import requests
from langchain_core.tools import BaseTool, tool

from .data_analysis import (
    DataWorkspace,
    add_column,
    column_stats,
    describe_table,
    fit_polynomial,
    format_stats,
    safe_eval,
)
from .graphics import plot_fit
from .knowledge import KnowledgeBase, format_docs

WIKI_API = "https://ru.wikipedia.org/w/api.php"
HTTP_HEADERS = {"User-Agent": "LabMentor/1.0 (educational project)"}


@tool
def web_search(query: str) -> str:
    """Поиск в интернете (DuckDuckGo). Используй для справочных данных: свойства материалов,
    физические константы, стандарты (ГОСТ), характеристики приборов."""
    try:
        from ddgs import DDGS

        results = DDGS().text(query, region="ru-ru", max_results=5)
    except Exception as exc:
        return f"Поиск в интернете недоступен: {exc}"
    if not results:
        return "Ничего не найдено"
    return "\n\n".join(
        f"{i}. {r.get('title', '')}\n{r.get('href', '')}\n{r.get('body', '')}"
        for i, r in enumerate(results, start=1)
    )


@tool
def wikipedia_search(query: str) -> str:
    """Краткая справка из русской Википедии по термину, закону или явлению."""
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 2,
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "exlimit": 2,
        "inprop": "url",
    }
    try:
        data = requests.get(WIKI_API, params=params, headers=HTTP_HEADERS, timeout=15).json()
    except Exception as exc:
        return f"Википедия недоступна: {exc}"
    pages = sorted(
        data.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 0)
    )
    if not pages:
        return "Ничего не найдено"
    return "\n\n".join(
        f"{p['title']} ({p.get('fullurl', '')})\n{p.get('extract', '')[:1500]}" for p in pages
    )


@tool
def calculator(expression: str) -> str:
    """Вычисляет арифметическое выражение: + - * / ** ^, скобки, sqrt, log (натуральный),
    log10, exp, sin, cos, tan, pi, e. Пример: sqrt(0.5 * 100)."""
    try:
        return f"{expression} = {float(safe_eval(expression)):.6g}"
    except Exception as exc:
        return f"Ошибка вычисления: {exc}"


def make_manual_search_tool(kb: KnowledgeBase, k: int = 4) -> BaseTool:

    @tool
    def search_lab_manuals(query: str) -> str:
        """Поиск по методическим указаниям к лабораторным работам: цель, теория, формулы,
        порядок выполнения, обработка результатов, требования к отчету."""
        docs = kb.search(query, k)
        return format_docs(docs) if docs else "В методических указаниях ничего не найдено"

    return search_lab_manuals


def make_data_tools(ws: DataWorkspace, output_dir: Path) -> list[BaseTool]:

    def table():
        if ws.df is None:
            raise ValueError(
                "Таблица не загружена. Попросите студента прикрепить CSV или XLSX файл"
            )
        return ws.df

    @tool
    def describe_data() -> str:
        """Структура таблицы измерений: столбцы, число строк, первые значения. Вызывай первым."""
        try:
            return describe_table(table(), ws.name)
        except Exception as exc:
            return f"Ошибка: {exc}"

    @tool
    def column_statistics(column: str, confidence: float = 0.95) -> str:
        """Среднее, СКО, коэффициент Стьюдента и доверительная граница погрешности столбца.
        confidence: доверительная вероятность 0.90, 0.95 или 0.99."""
        try:
            result = format_stats(column_stats(table(), column, confidence))
            ws.last_results = result
            return result
        except Exception as exc:
            return f"Ошибка: {exc}"

    @tool
    def fit_dependency(x_column: str, y_column: str, degree: int = 1) -> str:
        """Аппроксимация зависимости y(x) методом наименьших квадратов (degree от 1 до 3)
        и построение графика. Возвращает уравнение, погрешности коэффициентов и R²."""
        try:
            fit = fit_polynomial(table(), x_column, y_column, degree)
            plot = plot_fit(table(), fit, output_dir)
            ws.images.append(plot)
            ws.last_plot, ws.last_results = plot, fit.describe()
            return f"{fit.describe()} График построен и показан студенту."
        except Exception as exc:
            return f"Ошибка: {exc}"

    @tool
    def add_computed_column(name: str, formula: str) -> str:
        """Добавляет вычисляемый столбец по формуле из имен столбцов и функций sqrt, log, exp.
        Примеры: name='R, Ом', formula='U_В / I_мА * 1000'; name='ln U', formula='log(U_В)'."""
        try:
            column = add_column(table(), name, formula)
            values = json.dumps(
                [round(float(v), 6) for v in table()[column].head(8)], ensure_ascii=False
            )
            return f"Добавлен столбец {column}. Первые значения: {values}"
        except Exception as exc:
            return f"Ошибка: {exc}"

    return [describe_data, column_statistics, fit_dependency, add_computed_column, calculator]
