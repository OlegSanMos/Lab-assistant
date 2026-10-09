from __future__ import annotations

import ast
import io
import math
import operator
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx"}

_T_DF = list(range(1, 21)) + [25, 30, 40, 60, 120]
STUDENT_T = {
    0.90: [
        6.314,
        2.920,
        2.353,
        2.132,
        2.015,
        1.943,
        1.895,
        1.860,
        1.833,
        1.812,
        1.796,
        1.782,
        1.771,
        1.761,
        1.753,
        1.746,
        1.740,
        1.734,
        1.729,
        1.725,
        1.708,
        1.697,
        1.684,
        1.671,
        1.658,
    ],
    0.95: [
        12.706,
        4.303,
        3.182,
        2.776,
        2.571,
        2.447,
        2.365,
        2.306,
        2.262,
        2.228,
        2.201,
        2.179,
        2.160,
        2.145,
        2.131,
        2.120,
        2.110,
        2.101,
        2.093,
        2.086,
        2.060,
        2.042,
        2.021,
        2.000,
        1.980,
    ],
    0.99: [
        63.657,
        9.925,
        5.841,
        4.604,
        4.032,
        3.707,
        3.499,
        3.355,
        3.250,
        3.169,
        3.106,
        3.055,
        3.012,
        2.977,
        2.947,
        2.921,
        2.898,
        2.878,
        2.861,
        2.845,
        2.787,
        2.750,
        2.704,
        2.660,
        2.617,
    ],
}
_T_INFINITY = {0.90: 1.645, 0.95: 1.960, 0.99: 2.576}


@dataclass
class DataWorkspace:
    df: pd.DataFrame | None = None
    name: str = ""
    images: list[str] = field(default_factory=list)
    last_results: str = ""
    last_plot: str | None = None


def student_t(dof: int, p: float = 0.95) -> float:
    p = min(STUDENT_T, key=lambda q: abs(q - p))
    if dof < 1:
        return float("nan")
    if dof > _T_DF[-1]:
        return _T_INFINITY[p]
    return STUDENT_T[p][max(i for i, f in enumerate(_T_DF) if f <= dof)]


def clean_column_name(name: object) -> str:
    text = re.sub(r"\W+", "_", str(name).strip()).strip("_") or "col"
    return f"c_{text}" if text[0].isdigit() else text


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        raw = path.read_bytes()
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            text = raw.decode("cp1251")
        header = re.sub(r'"[^"]*"', "", text.splitlines()[0]) if text.strip() else ""
        sep = ";" if ";" in header else "\t" if "\t" in header else ","
        decimal = "," if sep != "," and re.search(r"\d,\d", text) else "."
        df = pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal)
    df = df.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    labels, names = {}, []
    for original in df.columns:
        name = clean_column_name(original)
        while name in names:
            name += "_"
        names.append(name)
        labels[name] = str(original)
    df.columns = names
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            values = pd.to_numeric(
                df[col].astype(str).str.replace(",", ".").str.replace(r"\s", "", regex=True),
                errors="coerce",
            )
            if values.notna().mean() >= 0.8:
                df[col] = values
    df.attrs["labels"] = labels
    return df


def describe_table(df: pd.DataFrame, name: str = "") -> str:
    labels = df.attrs.get("labels", {})
    columns = ", ".join(f"{c} ({labels.get(c, c)})" for c in df.columns)
    return (
        f"Таблица {name}: {len(df)} строк. Столбцы (имя для формул и исходное название): "
        f"{columns}.\n"
        f"Первые строки:\n{df.head(8).to_string(index=False)}"
    )


def numeric(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df.columns:
        raise ValueError(f"Нет столбца {column}. Доступны: {', '.join(df.columns)}")
    return pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)


def column_stats(df: pd.DataFrame, column: str, p: float = 0.95) -> dict:
    x = numeric(df, column)
    n = x.size
    if n < 2:
        raise ValueError("Для оценки погрешности нужно не менее двух измерений")
    mean, std = x.mean(), x.std(ddof=1)
    sem = std / math.sqrt(n)
    t = student_t(n - 1, p)
    delta = t * sem
    return {
        "column": column,
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "t": t,
        "p": p,
        "delta": delta,
        "relative": abs(delta / mean) * 100 if mean else float("nan"),
        "min": x.min(),
        "max": x.max(),
    }


def format_stats(st: dict) -> str:
    return (
        f"Статистика {st['column']} (n = {st['n']}): среднее {st['mean']:.5g}, СКО "
        f"{st['std']:.3g}, "
        f"СКО среднего {st['sem']:.3g}, t({st['p']}; {st['n'] - 1}) = {st['t']}, "
        f"доверительная граница {st['delta']:.3g} ({st['relative']:.2g} %). "
        f"Результат: {st['column']} = {st['mean']:.5g} ± {st['delta']:.2g}, P = {st['p']}. "
        f"Диапазон значений: {st['min']:.5g}...{st['max']:.5g}."
    )


@dataclass
class FitResult:
    x: str
    y: str
    coefficients: np.ndarray
    stderr: np.ndarray
    r2: float
    n: int
    t: float
    p: float

    @property
    def degree(self) -> int:
        return len(self.coefficients) - 1

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.polyval(self.coefficients, x)

    def equation(self) -> str:
        terms = []
        for power, c in zip(range(self.degree, -1, -1), self.coefficients, strict=True):
            var = "" if power == 0 else "x" if power == 1 else f"x^{power}"
            terms.append(f"{c:.4g}" + (f"·{var}" if var else ""))
        return "y = " + " + ".join(terms).replace("+ -", "- ")

    def describe(self) -> str:
        names = [f"a{power}" for power in range(self.degree, -1, -1)]
        if self.degree == 1:
            names = ["k (угловой коэффициент)", "b (свободный член)"]
        coeffs = "; ".join(
            f"{name} = {c:.5g} ± {self.t * s:.2g}"
            for name, c, s in zip(names, self.coefficients, self.stderr, strict=True)
        )
        return (
            f"МНК-аппроксимация {self.y}({self.x}), степень {self.degree}, n = {self.n}: "
            f"{self.equation()}. "
            f"Коэффициенты с доверительными границами (P = {self.p}): {coeffs}. "
            f"Коэффициент детерминации R² = {self.r2:.5f}."
        )


def fit_polynomial(
    df: pd.DataFrame, x_col: str, y_col: str, degree: int = 1, p: float = 0.95
) -> FitResult:
    for col in (x_col, y_col):
        numeric(df, col)
    data = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce").dropna()
    x, y = data[x_col].to_numpy(float), data[y_col].to_numpy(float)
    degree = int(min(max(degree, 1), 3))
    dof = x.size - degree - 1
    if dof < 1:
        raise ValueError(f"Для полинома степени {degree} нужно не менее {degree + 2} точек")
    coefficients, cov = np.polyfit(x, y, degree, cov="unscaled")
    residuals = y - np.polyval(coefficients, x)
    s2 = float(residuals @ residuals) / dof
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(residuals @ residuals) / ss_tot if ss_tot else 1.0
    return FitResult(
        x_col, y_col, coefficients, np.sqrt(np.diag(cov) * s2), r2, x.size, student_t(dof, p), p
    )


_FUNCTIONS = {
    "sqrt": np.sqrt,
    "log": np.log,
    "ln": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arctan": np.arctan,
    "abs": np.abs,
    "radians": np.radians,
    "degrees": np.degrees,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_eval(expression: str, variables: dict | None = None):
    names = {**_CONSTANTS, **(variables or {})}

    def evaluate(node: ast.AST):
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"Неизвестное имя: {node.id}")
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Pow) and np.max(np.abs(right)) > 100:
                raise ValueError("Слишком большой показатель степени")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](evaluate(node.operand))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _FUNCTIONS
            and not node.keywords
        ):
            return _FUNCTIONS[node.func.id](*[evaluate(arg) for arg in node.args])
        raise ValueError("Недопустимый элемент выражения")

    return evaluate(ast.parse(expression.replace("^", "**"), mode="eval"))


def add_column(df: pd.DataFrame, name: str, formula: str) -> str:
    variables = {col: pd.to_numeric(df[col], errors="coerce") for col in df.columns}
    with np.errstate(all="ignore"):
        values = safe_eval(formula, variables)
    column = clean_column_name(name)
    df[column] = values
    df.attrs.setdefault("labels", {})[column] = name.strip()
    return column


def parse_column_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError("Используйте запись вида: R_Ом = U_В / I_мА * 1000")
    name, formula = spec.split("=", 1)
    return name.strip(), formula.strip()
