from __future__ import annotations

import re

LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
    {"left": "\\(", "right": "\\)", "display": False},
]

MATH = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\\((.+?)\\\)|\$([^$\n]+?)\$", re.S)

SYMBOLS = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "Gamma": "Γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "Theta": "Θ",
    "kappa": "κ",
    "lambda": "λ",
    "Lambda": "Λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "Pi": "Π",
    "rho": "ρ",
    "sigma": "σ",
    "Sigma": "Σ",
    "tau": "τ",
    "phi": "φ",
    "varphi": "φ",
    "Phi": "Φ",
    "chi": "χ",
    "psi": "ψ",
    "Psi": "Ψ",
    "omega": "ω",
    "Omega": "Ω",
    "cdot": "·",
    "times": "×",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "approx": "≈",
    "sim": "~",
    "simeq": "≃",
    "equiv": "≡",
    "neq": "≠",
    "ne": "≠",
    "le": "≤",
    "leq": "≤",
    "ge": "≥",
    "geq": "≥",
    "ll": "≪",
    "gg": "≫",
    "propto": "∝",
    "infty": "∞",
    "partial": "∂",
    "nabla": "∇",
    "sum": "Σ",
    "prod": "Π",
    "int": "∫",
    "to": "→",
    "rightarrow": "→",
    "Rightarrow": "⇒",
    "Leftrightarrow": "⇔",
    "circ": "°",
    "degree": "°",
    "perp": "⊥",
    "parallel": "∥",
    "ldots": "…",
    "dots": "…",
    "cdots": "…",
    "langle": "⟨",
    "rangle": "⟩",
    "prime": "′",
    "lt": "<",
    "gt": ">",
    "%": "%",
    "$": "$",
    "_": "_",
    "{": "{",
    "}": "}",
    "#": "#",
    "&": "&",
    ",": " ",
    ";": " ",
    ":": " ",
    " ": " ",
    "!": "",
    "quad": " ",
    "qquad": " ",
    "\\": "; ",
    "left": "",
    "right": "",
    "big": "",
    "Big": "",
    "bigl": "",
    "bigr": "",
    "Bigl": "",
    "Bigr": "",
    "displaystyle": "",
    "limits": "",
    "nolimits": "",
}
TEXT_COMMANDS = {
    "text",
    "textrm",
    "mathrm",
    "mathbf",
    "textbf",
    "mathit",
    "textit",
    "mathsf",
    "mathtt",
    "operatorname",
    "mbox",
    "boldsymbol",
    "mathcal",
    "mathbb",
}
ACCENTS = {
    "bar": "\u0304",
    "overline": "\u0304",
    "hat": "\u0302",
    "widehat": "\u0302",
    "vec": "\u20d7",
    "dot": "\u0307",
    "ddot": "\u0308",
    "tilde": "\u0303",
}
OPERATORS = set("·×÷±∓≈~≃≡≠≤≥≪≫∝→⇒⇔")
SUPERSCRIPT = dict(zip("0123456789+-=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ", strict=True))
SUBSCRIPT = dict(
    zip("0123456789+-=()aehijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ", strict=True)
)
TAGS = re.compile(r"</?su[bp]>")


def prettify_math(text: str) -> str:
    return MATH.sub(lambda m: re.sub(r"(?<=\d),(?=\d)", "{,}", m.group(0)), text)


def latex_to_text(text: str, markup: bool = False) -> str:

    def convert(match: re.Match) -> str:
        display = match.group(1) is not None or match.group(2) is not None
        formula = next(group for group in match.groups() if group is not None)
        result = re.sub(r"\s+", " ", _Formula(formula, markup).parse()).replace(" ;", ";").strip()
        return f"\n{result}\n" if display else result

    return MATH.sub(convert, text)


def _wrap(text: str) -> str:
    text = text.strip()
    return f"({text})" if re.search(r"[\s+\-·×/=<>±]", TAGS.sub("", text)) else text


class _Formula:
    def __init__(self, source: str, markup: bool) -> None:
        self.src, self.pos, self.markup, self.depth = source, 0, markup, 0

    def parse(self, group: bool = False) -> str:
        out = []
        while self.pos < len(self.src):
            if self.src[self.pos] == "}":
                self.pos += 1
                if group:
                    break
                continue
            out.append(self.atom())
        return "".join(out)

    def atom(self) -> str:
        char = self.src[self.pos]
        self.pos += 1
        if char == "{":
            return self.parse(group=True)
        if char == "\\":
            return self.command()
        if char in "^_":
            self.depth += 1
            argument = self.argument()
            self.depth -= 1
            return self.script(argument, superscript=char == "^")
        return {"&": "", "~": " "}.get(char, char)

    def argument(self) -> str:
        self.skip_spaces()
        return self.atom() if self.pos < len(self.src) and self.src[self.pos] != "}" else ""

    def skip_spaces(self) -> None:
        while self.pos < len(self.src) and self.src[self.pos] == " ":
            self.pos += 1

    def command(self) -> str:
        start = self.pos
        while self.pos < len(self.src) and self.src[self.pos].isalpha():
            self.pos += 1
        if self.pos == start and self.pos < len(self.src):
            self.pos += 1
        name = self.src[start : self.pos]
        if name in ("frac", "dfrac", "tfrac", "cfrac"):
            numerator = self.argument()
            return f"{_wrap(numerator)}/{_wrap(self.argument())}"
        if name == "sqrt":
            degree = ""
            if self.src.startswith("[", self.pos):
                end = self.src.find("]", self.pos)
                degree, self.pos = self.src[self.pos + 1 : end], end + 1
            return {"3": "∛", "4": "∜"}.get(degree, "√") + _wrap(self.argument())
        if name in TEXT_COMMANDS:
            return self.argument()
        if name in ACCENTS:
            argument = self.argument()
            return argument + ACCENTS[name] if len(argument) == 1 else argument
        if name in ("begin", "end"):
            self.argument()
            return ""
        if name in ("left", "right") and self.src.startswith(".", self.pos):
            self.pos += 1
        if name not in SYMBOLS:
            return name
        symbol = SYMBOLS[name]
        if name.isalpha():
            rest = self.src[self.pos :].lstrip(" ")
            if rest[:1].isalnum() or rest[:1] in ("\\", "("):
                self.skip_spaces()
        if symbol in OPERATORS:
            return f" {symbol} "
        return f" {symbol}" if name in ("sum", "prod", "int") else symbol

    def script(self, text: str, superscript: bool) -> str:
        if text.strip() in ("°", "′"):
            return text.strip()
        if self.markup and self.depth == 0:
            tag = "sup" if superscript else "sub"
            return f"<{tag}>{text}</{tag}>"
        table = SUPERSCRIPT if superscript else SUBSCRIPT
        if text and all(char in table for char in text):
            return "".join(table[char] for char in text)
        return f"^{_wrap(text)}" if superscript else f"_{text}"
