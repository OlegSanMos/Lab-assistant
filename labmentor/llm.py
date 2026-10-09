from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from .config import Settings

PROVIDERS = ("openai", "ollama")


class LLMUnavailableError(RuntimeError):
    pass


def create_chat_model(
    settings: Settings,
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> BaseChatModel:
    provider = (provider or settings.provider).lower()
    model = model or settings.chat_model
    temperature = settings.temperature if temperature is None else temperature

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
            num_ctx=num_ctx or settings.num_ctx,
            num_predict=settings.max_answer_tokens,
        )
    if provider == "openai":
        if not settings.openai_api_key:
            raise LLMUnavailableError(
                "Не задан OPENAI_API_KEY. Укажите ключ в файле .env или выберите провайдера ollama."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            temperature=temperature,
            max_tokens=settings.max_answer_tokens,
            timeout=120,
            max_retries=2,
        )
    raise LLMUnavailableError(f"Неизвестный провайдер LLM: {provider}")


def create_embeddings(settings: Settings) -> Embeddings | None:
    provider = settings.embeddings_provider
    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        inner: Embeddings = OllamaEmbeddings(
            model=settings.embedding_model, base_url=settings.ollama_base_url
        )
    elif provider == "openai" and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        inner = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url or None,
            check_embedding_ctx_length=False,
        )
    else:
        return None
    cache_file = (
        settings.cache_dir / f"embeddings_{provider}_{_slug(settings.embedding_model)}.json"
    )
    return CachedEmbeddings(inner, cache_file)


class CachedEmbeddings(Embeddings):
    def __init__(self, inner: Embeddings, cache_file: Path) -> None:
        self.inner = inner
        self.cache_file = cache_file
        self._cache: dict[str, list[float]] = {}
        if cache_file.exists():
            self._cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [hashlib.sha1(text.encode("utf-8")).hexdigest() for text in texts]
        missing = [
            (key, text) for key, text in zip(keys, texts, strict=True) if key not in self._cache
        ]
        if missing:
            vectors = self.inner.embed_documents([text for _, text in missing])
            self._cache.update(
                {key: vector for (key, _), vector in zip(missing, vectors, strict=True)}
            )
            self.cache_file.write_text(json.dumps(self._cache), encoding="utf-8")
        return [self._cache[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.inner.embed_query(text)


def check_setup(settings: Settings) -> list[str]:
    problems = []
    if settings.provider == "openai" and not settings.openai_api_key:
        problems.append("Не задан OPENAI_API_KEY в файле .env (или укажите LLM_PROVIDER=ollama).")
    needed = set()
    if settings.provider == "ollama":
        needed |= {settings.chat_model, settings.vision_model}
    if settings.embeddings_provider == "ollama":
        needed.add(settings.embedding_model)
    if needed:
        try:
            reply = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=3).json()
            installed = {model["name"] for model in reply.get("models", [])}
        except Exception:
            return problems + [
                f"Ollama не отвечает по адресу {settings.ollama_base_url}: запустите Ollama."
            ]
        for model in sorted(needed):
            if (model if ":" in model else f"{model}:latest") not in installed:
                problems.append(f"Модель {model} не загружена: выполните ollama pull {model}")
    return problems


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name)
