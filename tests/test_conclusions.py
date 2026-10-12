from labmentor.assistant import Options
from labmentor.data_analysis import add_column, load_table, parse_column_spec, table_text


def lab(assistant, number: str) -> str:
    return next(title for title in assistant.lab_titles() if f"работа {number}." in title)


def diode_session(assistant, sample):
    session = assistant.new_session()
    assistant.load_table(session, sample("diode_iv.csv"))
    column = add_column(session.data.df, *parse_column_spec("ln_I = ln(I_мА)"))
    assistant.analyze(session, "U_В", column, 1, 0.95, None)
    return session


def prompt(llm) -> str:
    return "\n".join(message.content for message in llm.calls[-1])


def test_conclusions_follow_the_selected_lab(course_assistant, sample):
    session = diode_session(course_assistant, sample)
    diode = lab(course_assistant, "1.1")
    llm = course_assistant.llm(Options())
    text = course_assistant.explain_results(session, Options(lab=diode))
    system, human = llm.calls[-1][0].content, llm.calls[-1][-1].content
    assert f"по лабораторной работе «{diode}»" in system
    assert "Рассчитай величины, которые требует эта работа" in system
    assert "U, В" in human and "ln_I" in human and "0.481" in human
    assert "МНК-аппроксимация ln_I(U_В)" in human and "Цели работы" in human
    assert "коэффициент неидеальности" in human and "Подготовка отчёта" in human
    assert "[lab_01_diod-stab.md]" in human and "lab_02_bjt.md" not in human
    assert session.data.insight == text and session.data.insight_lab == diode


def test_report_reuses_conclusions_of_the_same_lab(course_assistant, sample):
    session = diode_session(course_assistant, sample)
    diode, transistor = lab(course_assistant, "1.1"), lab(course_assistant, "1.2")
    llm = course_assistant.llm(Options())
    text = course_assistant.explain_results(session, Options(lab=diode))
    course_assistant.make_report(session, Options(lab=transistor), "")
    assert text not in prompt(llm)
    course_assistant.make_report(session, Options(lab=diode), "")
    assert f"Выводы по результатам, сформулированные ранее:\n{text}" in prompt(llm)
    course_assistant.analyze(session, "U_В", "I_мА", 2, 0.95, None)
    assert session.data.insight == "" and session.data.insight_lab == ""


def test_table_for_the_model_uses_units_and_limits_rows(sample):
    df = load_table(sample("ohm_law.csv"))
    text = table_text(df, limit=5)
    lines = text.splitlines()
    assert lines[0].split() == ["U,", "В", "I,", "мА"] and len(lines) == 7
    assert lines[-1] == f"и еще строк: {len(df) - 5}"
    assert table_text(df).count("\n") == len(df)
