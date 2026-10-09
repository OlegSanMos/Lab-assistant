from __future__ import annotations

from functools import partial

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.runnables import Runnable

count_tokens = partial(count_tokens_approximately, chars_per_token=3.0)


def plan_context(
    num_ctx: int, history_tokens: int, top_k: int, chunk_size: int, answer_tokens: int
) -> dict:
    system, question = 500, 300
    rag = top_k * chunk_size // 3
    free = num_ctx - system - question - rag - answer_tokens
    return {
        "num_ctx": num_ctx,
        "system": system,
        "rag": rag,
        "answer": answer_tokens,
        "history": max(256, min(history_tokens, free)),
    }


def format_dialog(messages: list[BaseMessage]) -> str:
    names = {"human": "Студент", "ai": "Ассистент"}
    return "\n".join(f"{names.get(m.type, m.type)}: {m.text}" for m in messages)


class ConversationMemory:
    def __init__(self, max_tokens: int = 2000, use_summary: bool = True) -> None:
        self.messages: list[BaseMessage] = []
        self.max_tokens = max_tokens
        self.use_summary = use_summary
        self.summary = ""
        self._summarized = 0

    def add_turn(self, question: str, answer: str) -> None:
        self.messages += [HumanMessage(content=question), AIMessage(content=answer)]

    def window(self) -> list[BaseMessage]:
        return trim_messages(
            self.messages,
            max_tokens=self.max_tokens,
            token_counter=count_tokens,
            strategy="last",
            start_on="human",
            allow_partial=False,
        )

    def evicted(self) -> list[BaseMessage]:
        return self.messages[: len(self.messages) - len(self.window())]

    def summary_text(self) -> str:
        return self.summary if self.use_summary and self.summary else "нет"

    def update_summary(self, summary_chain: Runnable) -> bool:
        if not self.use_summary:
            return False
        evicted = self.evicted()
        new_messages = evicted[self._summarized :]
        if not new_messages:
            return False
        self.summary = summary_chain.invoke(
            {"summary": self.summary or "пока пусто", "dialog": format_dialog(new_messages)}
        ).strip()
        self._summarized = len(evicted)
        return True

    def stats(self) -> dict:
        window = self.window()
        return {
            "tokens": count_tokens(window),
            "limit": self.max_tokens,
            "in_window": len(window),
            "total": len(self.messages),
            "summary_words": len(self.summary.split()) if self.use_summary else 0,
        }

    def clear(self) -> None:
        self.messages = []
        self.summary = ""
        self._summarized = 0
