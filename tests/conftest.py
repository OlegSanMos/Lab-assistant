from __future__ import annotations

import json
import math
import shutil
import wave
from pathlib import Path
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from labmentor.assistant import LabAssistant
from labmentor.config import PROJECT_DIR, Settings

SAMPLES = PROJECT_DIR / "data" / "samples"


class ScriptedChatModel(BaseChatModel):
    responder: Any
    calls: list = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs
    ) -> ChatResult:
        self.calls.append(messages)
        result = self.responder(messages)
        message = result if isinstance(result, AIMessage) else AIMessage(content=result)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def bind_tools(self, tools, **kwargs):
        return self


FORMULA_ANSWER = (
    "Закон Ома для участка цепи: $I = \\frac{U}{R}$, отсюда $R = \\frac{U}{I}$.\n\n"
    "$$P = U \\cdot I \\le 0,5\\ \\text{Вт}$$\n\n"
    "Погрешность: \\(\\Delta R = t_{0,95} \\cdot \\frac{S}{\\sqrt{n}}\\), "
    "постоянная времени \\[\\tau = R_{\\text{н}} C\\] [lr1_ohm_law.md]"
)


def make_wav(path: Path, seconds: float = 1.0) -> Path:
    rate = 16000
    frames = b"".join(
        int(8000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True)
        for i in range(int(rate * seconds))
    )
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(frames)
    return path


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


def lab_responder(messages: list[BaseMessage]) -> AIMessage | str:
    system = messages[0].text if messages else ""
    last = messages[-1]
    question = last.text
    tool_results = [m for m in messages if isinstance(m, ToolMessage)]
    if "Ты диспетчер" in system:
        text = question.lower()
        if "график" in text or "погрешност" in text:
            return "data"
        if "схем" in text:
            return "flowchart"
        if "интернет" in text:
            return "research"
        if "проверь меня" in text:
            return "exam"
        return "manual"
    if "агент-аналитик" in system:
        if not tool_results:
            return tool_call("describe_data", {}, "call_1")
        if len(tool_results) == 1:
            return tool_call(
                "fit_dependency", {"x_column": "U_В", "y_column": "I_мА", "degree": 1}, "call_2"
            )
        return "Сопротивление R = 1/k, где k взят из результата МНК. " + tool_results[-1].text[:80]
    if "агент-исследователь" in system:
        if not tool_results:
            return tool_call("search_lab_manuals", {"query": "модуль упругости стали"}, "call_r1")
        return "По методичке E ≈ 2,0...2,1 · 10⁵ МПа [lr2_young_modulus.md]."
    if "Ты сжимаешь историю" in system:
        return "Студент изучает лабораторную работу по закону Ома."
    if "контрольных вопросов" in system:
        return (
            "```json\n"
            + json.dumps(
                {
                    "questions": [
                        {"question": "Сформулируйте закон Ома.", "answer": "I = U / R"},
                        {
                            "question": "Как включается амперметр?",
                            "answer": "Последовательно с резистором.",
                        },
                    ]
                },
                ensure_ascii=False,
            )
            + "\n```"
        )
    if "Ты экзаменатор" in system:
        return '{"score": 8, "feedback": "Верно, но без формулы."}'
    if "составь алгоритм" in system:
        return json.dumps(
            {
                "title": "Порядок выполнения ЛР 1",
                "steps": ["Собрать цепь", "Установить ноль", "Снять ВАХ", "Выключить источник"],
            },
            ensure_ascii=False,
        )
    if "Отвечай на основе фрагментов методических указаний" in system:
        if "формул" in question.lower():
            return FORMULA_ANSWER
        return "Цель работы: проверить закон Ома и определить сопротивление [lr1_ohm_law.md]."
    if "прислал изображение" in system:
        return "На осциллограмме экспоненциальный разряд, tau около 1 с."
    if "Составь черновик отчета" in system:
        return (
            "## Цель работы\nПроверить закон Ома.\n## Результаты измерений и их обработка\n"
            "- R = 101,5 Ом\n## Выводы\nЗакон Ома подтвержден."
        )
    return "Здравствуйте! Чем помочь?"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        provider="openai",
        openai_api_key="test-key",
        embeddings_provider="none",
        storage_dir=tmp_path,
    )


@pytest.fixture
def scripted_llm() -> ScriptedChatModel:
    return ScriptedChatModel(responder=lab_responder)


@pytest.fixture
def assistant(settings: Settings, scripted_llm: ScriptedChatModel) -> LabAssistant:
    return LabAssistant(settings, llm_factory=lambda **kwargs: scripted_llm)


@pytest.fixture
def sample(tmp_path: Path):

    def copy(name: str) -> str:
        target = tmp_path / name
        shutil.copy(SAMPLES / name, target)
        return str(target)

    return copy
