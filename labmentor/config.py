from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_DIR / ".env")

DEFAULT_MODELS = {
    "openai": {
        "chat": "gpt-4.1-mini",
        "vision": "gpt-4.1-mini",
        "embeddings": "text-embedding-3-small",
    },
    "ollama": {"chat": "qwen2.5:7b", "vision": "qwen2.5vl:7b", "embeddings": "bge-m3"},
}


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass
class Settings:
    provider: str = "openai"
    chat_model: str = ""
    vision_model: str = ""
    embeddings_provider: str = ""
    embedding_model: str = ""
    temperature: float = 0.3
    openai_api_key: str = ""
    openai_base_url: str = ""
    ollama_base_url: str = "http://localhost:11434"
    num_ctx: int = 8192
    max_answer_tokens: int = 1500
    history_tokens: int = 2000
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 4
    stt_engine: str = "auto"
    stt_model: str = "whisper-1"
    whisper_size: str = "small"
    tts_voice: str = "ru-RU-SvetlanaNeural"
    manuals_dir: Path = PROJECT_DIR / "data" / "manuals"
    storage_dir: Path = PROJECT_DIR / "storage"

    def __post_init__(self) -> None:
        self.provider = self.provider.lower()
        self.manuals_dir, self.storage_dir = Path(self.manuals_dir), Path(self.storage_dir)
        defaults = DEFAULT_MODELS.get(self.provider, DEFAULT_MODELS["openai"])
        self.chat_model = self.chat_model or defaults["chat"]
        self.vision_model = self.vision_model or defaults["vision"]
        self.embeddings_provider = (self.embeddings_provider or self.provider).lower()
        emb_defaults = DEFAULT_MODELS.get(self.embeddings_provider, defaults)
        self.embedding_model = self.embedding_model or emb_defaults["embeddings"]
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self.storage_dir / "outputs"

    @property
    def cache_dir(self) -> Path:
        return self.storage_dir / "cache"

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            provider=_env("LLM_PROVIDER", "openai"),
            chat_model=_env("CHAT_MODEL"),
            vision_model=_env("VISION_MODEL"),
            embeddings_provider=_env("EMBEDDINGS_PROVIDER"),
            embedding_model=_env("EMBEDDING_MODEL"),
            temperature=float(_env("TEMPERATURE", "0.3")),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_base_url=_env("OPENAI_BASE_URL"),
            ollama_base_url=_env("OLLAMA_BASE_URL", "http://localhost:11434"),
            num_ctx=int(_env("NUM_CTX", "8192")),
            max_answer_tokens=int(_env("MAX_ANSWER_TOKENS", "1500")),
            history_tokens=int(_env("HISTORY_TOKENS", "2000")),
            top_k=int(_env("TOP_K", "4")),
            stt_engine=_env("STT_ENGINE", "auto"),
            stt_model=_env("STT_MODEL", "whisper-1"),
            whisper_size=_env("WHISPER_SIZE", "small"),
            tts_voice=_env("TTS_VOICE", "ru-RU-SvetlanaNeural"),
        )


def use_system_certificates() -> bool:
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:
        return False
    return True
