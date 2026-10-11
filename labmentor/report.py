from __future__ import annotations

import re
import uuid
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from .formulas import TAGS, latex_to_text

FONT = "Times New Roman"
LEFT, CENTER = WD_ALIGN_PARAGRAPH.LEFT, WD_ALIGN_PARAGRAPH.CENTER
DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
TAG = re.compile(r"\\tag\*?\{([^{}]*)\}")
CAPTION = re.compile(
    r"^(Таблица|Рисунок)\s+((?:[А-ЯA-Z]\.)?\d+(?:\.\d+)*)\s*([.:\u2013\u2014-]?)\s*(.*)$"
)
FORMULA_NUMBER = re.compile(r"^\((\d+(?:\.\d+)*)\)$")
HEADING = re.compile(r"^(#{2,4})\s+(.*)")
BULLET = re.compile(r"^\s*[-*]\s+(.*)")
NUMBERED = re.compile(r"^\s*\d+[.)]\s+(.*)")
RULE = re.compile(r"^\s*([-*_])\1{2,}\s*$")


def _add_text(paragraph, text: str, bold: bool = False) -> None:
    for part in re.split(r"(\*\*[^*]+\*\*|<su[bp]>.*?</su[bp]>)", text):
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            _add_text(paragraph, part[2:-2], bold=True)
        elif part:
            run = paragraph.add_run(TAGS.sub("", part))
            run.bold = bold or None
            if part.startswith("<sub>"):
                run.font.subscript = True
            elif part.startswith("<sup>"):
                run.font.superscript = True


def _plain_font(rpr) -> None:
    fonts = rpr.get_or_add_rFonts()
    for name in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        fonts.attrib.pop(qn(f"w:{name}"), None)
    for name in ("ascii", "hAnsi", "eastAsia", "cs"):
        fonts.set(qn(f"w:{name}"), FONT)


def _setup(document) -> None:
    section = document.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin = section.bottom_margin = Cm(2)
    zoom = document.settings.element.find(qn("w:zoom"))
    if zoom is not None:
        zoom.set(qn("w:percent"), "100")
    _plain_font(document.styles.element.xpath("w:docDefaults/w:rPrDefault/w:rPr")[0])
    normal = document.styles["Normal"]
    _plain_font(normal.element.get_or_add_rPr())
    normal.font.size = Pt(14)
    fmt = normal.paragraph_format
    fmt.line_spacing, fmt.first_line_indent = 1.5, Cm(1.25)
    fmt.space_before = fmt.space_after = Pt(0)
    fmt.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    for level in (1, 2, 3):
        style = document.styles[f"Heading {level}"]
        _plain_font(style.element.get_or_add_rPr())
        style.font.size, style.font.bold, style.font.italic = Pt(14), True, False
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.alignment = LEFT


def _number(value) -> str:
    if pd.isna(value):
        return ""
    return f"{value:.6g}".replace(".", ",") if isinstance(value, float) else str(value)


def _has_data(table: pd.DataFrame | None, plot: str | None) -> tuple[bool, bool]:
    return table is not None and len(table) > 0, bool(plot) and Path(plot).exists()


def report_inserts(table: pd.DataFrame | None, plot: str | None) -> str:
    has_table, has_plot = _has_data(table, plot)
    items = ["таблицу 1 с результатами измерений"] if has_table else []
    if has_plot:
        items.append("рисунок 1 с графиком зависимости")
    if not items:
        return "нет, свои таблицы и рисунки нумеруй с 1."
    return (
        "программа сама вставит в начало раздела «Результаты измерений и их обработка» "
        + " и ".join(items)
        + " и сошлется на них. Не повторяй эти данные, свои таблицы и рисунки нумеруй со "
        "следующего номера."
    )


class _Writer:
    def __init__(self, document) -> None:
        self.doc = document
        self.tables = self.formulas = 0
        self.after_blank = self.captioned = False

    def add(self, text: str = "", indent: bool = True, align=None, bold: bool = False):
        paragraph = self.doc.add_paragraph()
        if not indent:
            paragraph.paragraph_format.first_line_indent = Cm(0)
        if align is not None:
            paragraph.paragraph_format.alignment = align
        _add_text(paragraph, text, bold)
        self.after_blank = not text
        return paragraph

    def blank(self) -> None:
        if not self.after_blank:
            self.add()

    def heading(self, text: str, level: int) -> None:
        text = TAGS.sub("", text).replace("*", "").strip().rstrip(".").strip()
        if not text:
            return
        if level == 1:
            self.blank()
        self.doc.add_heading(text[0].upper() + text[1:], level=level)
        self.after_blank = False
        if level == 1:
            self.blank()

    def caption(self, kind: str, number: str, name: str) -> None:
        if kind == "Таблица":
            text = f"Таблица {number} – {name}" if name else f"Таблица {number}"
            self.add(text, indent=False, align=LEFT).paragraph_format.keep_with_next = True
            if number.isdigit():
                self.tables = max(self.tables, int(number))
            self.captioned = True
        else:
            self.add(f"Рисунок {number}. {name}" if name else f"Рисунок {number}", False, CENTER)

    def table(self, rows: list[list[str]]) -> None:
        if not self.captioned:
            self.caption("Таблица", str(self.tables + 1), "")
        width = max(len(row) for row in rows)
        grid = self.doc.add_table(rows=len(rows), cols=width)
        grid.style = "Table Grid"
        for row, values in zip(grid.rows, rows, strict=True):
            for cell, value in zip(row.cells, values + [""] * (width - len(values)), strict=True):
                fmt = cell.paragraphs[0].paragraph_format
                fmt.first_line_indent, fmt.line_spacing, fmt.alignment = Cm(0), 1.0, CENTER
                _add_text(cell.paragraphs[0], value)
        grid.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        self.after_blank = self.captioned = False

    def markdown_table(self, lines: list[str]) -> None:
        rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
        rows = [row for row in rows if not all(re.fullmatch(r":?-+:?", cell) for cell in row)]
        if rows:
            self.table(rows)

    def formula(self, source: str, number: str = "") -> None:
        tag = TAG.search(source)
        number = tag.group(1).strip() if tag else number
        if number.isdigit():
            self.formulas = int(number)
        elif not number:
            self.formulas += 1
            number = str(self.formulas)
        text = latex_to_text(f"$${TAG.sub('', source)}$$", markup=True).strip()
        paragraph = self.add(indent=False, align=LEFT)
        stops = paragraph.paragraph_format.tab_stops
        stops.add_tab_stop(Cm(8.25), WD_TAB_ALIGNMENT.CENTER)
        stops.add_tab_stop(Cm(16.5), WD_TAB_ALIGNMENT.RIGHT)
        _add_text(paragraph, f"\t{text}\t({number})")
        self.after_blank = False

    def measurements(self, table: pd.DataFrame | None, plot: str | None) -> None:
        has_table, has_plot = _has_data(table, plot)
        if has_table:
            labels = table.attrs.get("labels", {})
            self.add("Результаты измерений приведены в таблице 1.")
            self.caption("Таблица", "1", "Результаты измерений")
            rows = [[labels.get(column, column) for column in table.columns]]
            rows += [[_number(v) for v in row] for row in table.head(50).itertuples(index=False)]
            self.table(rows)
        if has_plot:
            self.add(
                "Экспериментальная зависимость и ее аппроксимация по методу наименьших квадратов "
                "показаны на рисунке 1."
            )
            picture = self.add(indent=False, align=CENTER)
            picture.add_run().add_picture(plot, width=Cm(15))
            picture.paragraph_format.keep_with_next = True
            self.caption("Рисунок", "1", "Экспериментальная зависимость и аппроксимация МНК")


def _items(markdown: str) -> list[tuple[str, str]]:
    items, start = [], 0
    for match in DISPLAY.finditer(markdown):
        text = latex_to_text(markdown[start : match.start()], markup=True)
        items += [("line", line) for line in text.splitlines()]
        items.append(("formula", match.group(1) or match.group(2)))
        start = match.end()
    text = latex_to_text(markdown[start:], markup=True)
    return items + [("line", line) for line in text.splitlines()]


def _next_line(items: list[tuple[str, str]], index: int) -> str:
    for kind, line in items[index:]:
        if kind != "line":
            return ""
        if line.strip():
            return line.strip()
    return ""


def _formula_number(items: list[tuple[str, str]], index: int) -> tuple[str, int]:
    for position in range(index, len(items)):
        kind, line = items[position]
        if kind == "line" and not line.strip():
            continue
        match = FORMULA_NUMBER.match(line.strip()) if kind == "line" else None
        return (match.group(1), position + 1) if match else ("", index)
    return "", index


def _caption(line: str, next_line: str) -> re.Match | None:
    match = CAPTION.match(line.replace("*", "").strip())
    if not match or not (match.group(3) or not match.group(4)):
        return None
    if match.group(1) == "Таблица" and not next_line.startswith("|"):
        return None
    return match


def _is_row(item: tuple[str, str]) -> bool:
    return item[0] == "line" and item[1].lstrip().startswith("|")


def export_report_docx(
    markdown: str,
    title: str,
    out_dir: Path,
    table: pd.DataFrame | None = None,
    plot: str | None = None,
) -> str:
    document = Document()
    _setup(document)
    writer = _Writer(document)
    writer.add("Отчет по лабораторной работе", indent=False, align=CENTER, bold=True)
    if title:
        writer.add(title, indent=False, align=CENTER, bold=True)
    items, index, inserted = _items(markdown), 0, False
    while index < len(items):
        kind, line = items[index]
        index += 1
        if kind == "formula":
            number, index = _formula_number(items, index)
            writer.formula(line, number)
            continue
        line = line.rstrip()
        if not line.strip() or line.startswith("# ") or RULE.match(line):
            continue
        if _is_row((kind, line)):
            rows = [line]
            while index < len(items) and _is_row(items[index]):
                rows.append(items[index][1])
                index += 1
            writer.markdown_table(rows)
            continue
        heading, caption = HEADING.match(line), _caption(line, _next_line(items, index))
        bullet, number = BULLET.match(line), NUMBERED.match(line)
        if heading:
            title_text = TAGS.sub("", heading.group(2)).strip()
            writer.heading(title_text, len(heading.group(1)) - 1)
            if not inserted and "результат" in title_text.lower():
                writer.measurements(table, plot)
                inserted = True
        elif caption:
            writer.caption(caption.group(1), caption.group(2), caption.group(4).strip())
        elif bullet:
            _add_text(document.add_paragraph(style="List Bullet"), bullet.group(1))
            writer.after_blank = False
        elif number:
            _add_text(document.add_paragraph(style="List Number"), number.group(1))
            writer.after_blank = False
        elif re.match(r"где\b", line.strip()):
            writer.add(re.sub(r"^где\s*:?\s*", "где ", line.strip()), indent=False)
        else:
            writer.add(line)
    if not inserted and any(_has_data(table, plot)):
        writer.heading("Результаты измерений и их обработка", 1)
        writer.measurements(table, plot)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"report_{uuid.uuid4().hex[:8]}.docx"
    document.save(str(path))
    return str(path)
