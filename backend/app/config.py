from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).parent.parent / ".env", extra="ignore")

    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"


settings = Settings()
