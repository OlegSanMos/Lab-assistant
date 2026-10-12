from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.table import Table

from labmentor import chains
from labmentor.data_analysis import fit_polynomial, load_table
from labmentor.graphics import plot_fit
from labmentor.report import export_report_docx, report_inserts

MARKDOWN = r"""# Отчет по лабораторной работе 1.1

## цель работы
Изучить вольт-амперную характеристику диода.

## Краткие теоретические сведения.
Прямая ветвь ВАХ описывается выражением

$$I = I_0 \exp\left(\frac{U}{m \varphi_T}\right) \tag{1}$$

где: $I_0$ – тепловой ток, А;
$m$ – коэффициент неидеальности.

### Линеаризация
После логарифмирования

$$\ln I = \ln I_0 + \frac{U}{m \varphi_T}$$

Мощность на резисторе R1

$$P = U I$$ (3)

---

## Результаты измерений и их обработка
Параметры модели приведены в таблице 2.

**Таблица 2.** Параметры модели диода

| Параметр | Значение |
|---|---|
| $I_0$, А | $2{,}1 \cdot 10^{-9}$ |
| $m$ | 1,82 |

## Выводы
Получено $m = 1{,}82$.
"""


def build(tmp_path, sample):
    df = load_table(sample("diode_iv.csv"))
    plot = plot_fit(df, fit_polynomial(df, "U_В", "I_мА", 2, 0.95), tmp_path)
    path = export_report_docx(MARKDOWN, "Лабораторная работа 1.1. Диод", tmp_path, df, plot)
    return Document(path), list(Document(path).iter_inner_content())


def position(items, text):
    return next(
        i for i, item in enumerate(items) if not isinstance(item, Table) and item.text == text
    )


def test_report_page_fonts_and_spacing(tmp_path, sample):
    document, _ = build(tmp_path, sample)
    section = document.sections[0]
    assert (round(section.page_width.cm, 1), round(section.page_height.cm, 1)) == (21.0, 29.7)
    normal = document.styles["Normal"]
    assert normal.font.name == "Times New Roman" and normal.font.size == Pt(14)
    assert normal.paragraph_format.line_spacing == 1.5
    assert round(normal.paragraph_format.first_line_indent.cm, 2) == 1.25
    for level in (1, 2):
        style = document.styles[f"Heading {level}"]
        assert style.font.name == "Times New Roman" and style.font.size == Pt(14)
        assert style.font.bold and str(style.font.color.rgb) == "000000"
        assert style.element.rPr.rFonts.get(qn("w:asciiTheme")) is None


def test_report_headings_and_blank_lines(tmp_path, sample):
    _, items = build(tmp_path, sample)
    for text in ("Цель работы", "Краткие теоретические сведения", "Выводы"):
        index = position(items, text)
        assert items[index].style.name == "Heading 1"
        assert items[index - 1].text == "" and items[index + 1].text == ""
    index = position(items, "Линеаризация")
    assert items[index].style.name == "Heading 2"
    assert items[index - 1].text and items[index + 1].text


def test_report_tables_and_figure_captions(tmp_path, sample):
    _, items = build(tmp_path, sample)
    tables = [i for i, item in enumerate(items) if isinstance(item, Table)]
    captions = [items[i - 1] for i in tables]
    assert [c.text for c in captions] == [
        "Таблица 1 – Результаты измерений",
        "Таблица 2 – Параметры модели диода",
    ]
    assert all(c.paragraph_format.first_line_indent == 0 for c in captions)
    assert items[tables[0] - 2].text == "Результаты измерений приведены в таблице 1."
    measured, model = items[tables[0]], items[tables[1]]
    assert [c.text for c in measured.rows[0].cells] == ["U, В", "I, мА"]
    assert [c.text for c in measured.rows[1].cells] == ["0,4", "0,0181"]
    assert measured.rows[0]._tr.trPr.find(qn("w:tblHeader")) is not None
    assert [c.text for c in model.rows[1].cells] == ["I0, А", "2,1 · 10-9"] and len(model.rows) == 3
    figure = position(items, "Рисунок 1. Экспериментальная зависимость и аппроксимация МНК")
    assert items[figure - 1]._p.xpath(".//pic:pic") and "рисунке 1" in items[figure - 2].text
    assert items[figure].paragraph_format.alignment == WD_ALIGN_PARAGRAPH.CENTER
    texts = [item.text for item in items if not isinstance(item, Table)]
    assert not any(t.startswith(("|", "---", "**")) for t in texts)


def test_report_formulas_are_numbered_with_explanations(tmp_path, sample):
    _, items = build(tmp_path, sample)
    formulas = [i.text for i in items if not isinstance(i, Table) and i.text.startswith("\t")]
    assert formulas == [
        "\tI = I0 exp(U/(m φT))\t(1)",
        "\tln I = ln I0 + U/(m φT)\t(2)",
        "\tP = U I\t(3)",
    ]
    index = position(items, "где I0 – тепловой ток, А;")
    assert items[index].paragraph_format.first_line_indent == 0


def test_report_prompt_carries_department_rules(tmp_path, sample):
    system = chains.REPORT_PROMPT.messages[0].prompt.template
    for rule in ("ГОСТ 2.105", "без точки в конце", "Таблица 2 – Наименование", "«где»", "R1, R2"):
        assert rule in system
    messages = chains.REPORT_PROMPT.format_messages(
        lab="ЛР 1.1", manual="", results="", inserts=report_inserts(None, None), notes="нет"
    )
    assert "\\tag{1}" in messages[0].content and "нумеруй с 1" in messages[1].content
    df = load_table(sample("diode_iv.csv"))
    plot = plot_fit(df, fit_polynomial(df, "U_В", "I_мА", 2, 0.95), tmp_path)
    inserts = report_inserts(df, plot, [("figures/a.png", "Схема установки")])
    assert "таблицу 1" in inserts and "график зависимости" in inserts and "со 2" in inserts
    assert "![Схема установки](figures/a.png)" in inserts
