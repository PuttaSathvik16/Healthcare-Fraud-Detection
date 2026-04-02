from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HFD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=lambda: Path.cwd())
    data_raw_dir: Path | None = None
    data_processed_dir: Path | None = None
    model_registry_path: Path | None = None
    fraud_bundle_path: Path | None = None
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    def resolved_raw_dir(self) -> Path:
        return self.data_raw_dir or (self.project_root / "data" / "raw")

    def resolved_processed_dir(self) -> Path:
        return self.data_processed_dir or (self.project_root / "data" / "processed")

    def resolved_model_path(self) -> Path:
        return self.model_registry_path or (self.project_root / "models" / "artifacts" / "model.joblib")

    def resolved_fraud_bundle_path(self) -> Path:
        return self.fraud_bundle_path or (self.project_root / "models" / "fraud_model.pkl")


@lru_cache
def get_settings() -> Settings:
    return Settings()
