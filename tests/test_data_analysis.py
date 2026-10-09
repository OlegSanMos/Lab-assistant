import math
from pathlib import Path

import pytest

from labmentor.data_analysis import (
    add_column,
    column_stats,
    fit_polynomial,
    load_table,
    parse_column_spec,
    safe_eval,
    student_t,
)
from labmentor.graphics import draw_flowchart, plot_fit

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"


def test_load_csv_with_semicolon_and_decimal_comma():
    df = load_table(SAMPLES / "ohm_law.csv")
    assert list(df.columns) == ["U_В", "I_мА"]
    assert df["U_В"].dtype.kind == "f" and len(df) == 14
    assert df.attrs["labels"]["I_мА"] == "I, мА"


def test_load_xlsx_and_regular_csv():
    assert list(load_table(SAMPLES / "young_modulus.xlsx").columns) == ["F_кН", "Δl_мм"]
    assert len(load_table(SAMPLES / "rc_discharge.csv")) == 17


def test_load_csv_in_cp1251(tmp_path):
    path = tmp_path / "data.csv"
    path.write_bytes("Сила, Н;Длина, мм\n1,5;2,0\n3,0;4,1\n".encode("cp1251"))
    df = load_table(path)
    assert list(df.columns) == ["Сила_Н", "Длина_мм"]
    assert df["Сила_Н"].tolist() == [1.5, 3.0]


def test_student_coefficients():
    assert student_t(13, 0.95) == 2.160
    assert student_t(22, 0.95) == 2.086
    assert student_t(500, 0.99) == 2.576


def test_ohm_law_processing():
    df = load_table(SAMPLES / "ohm_law.csv")
    fit = fit_polynomial(df, "U_В", "I_мА")
    resistance = 1000 / fit.coefficients[0]
    assert abs(resistance - 101.5) < 1.0
    assert fit.r2 > 0.999
    column = add_column(df, "R, Ом", "U_В / I_мА * 1000")
    stats = column_stats(df, column)
    assert column == "R_Ом" and stats["n"] == 14
    assert stats["mean"] - stats["delta"] < 101.5 < stats["mean"] + stats["delta"] + 0.5


def test_rc_time_constant_by_linearization():
    df = load_table(SAMPLES / "rc_discharge.csv")
    add_column(df, "ln U", "log(U_В)")
    tau = -1 / fit_polynomial(df, "t_с", "ln_U").coefficients[0]
    assert 0.95 < tau < 1.15


def test_safe_eval_blocks_code():
    assert math.isclose(safe_eval("sqrt(0.5 * 100)"), math.sqrt(50))
    assert safe_eval("2^10") == 1024
    for expression in ["__import__('os')", "(1).__class__", "open('x')", "9**9**9"]:
        with pytest.raises(ValueError):
            safe_eval(expression)


def test_parse_column_spec():
    assert parse_column_spec("R, Ом = U_В / I_мА") == ("R, Ом", "U_В / I_мА")
    with pytest.raises(ValueError):
        parse_column_spec("U_В / I_мА")


def test_images_are_generated(tmp_path):
    df = load_table(SAMPLES / "ohm_law.csv")
    plot = plot_fit(df, fit_polynomial(df, "U_В", "I_мА"), tmp_path)
    chart = draw_flowchart("Порядок работы", ["Собрать цепь", "Снять показания приборов"], tmp_path)
    assert Path(plot).stat().st_size > 10_000 and Path(chart).stat().st_size > 5_000
