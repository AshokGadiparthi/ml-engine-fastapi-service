"""Configuration for ML Engine FastAPI Service."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # App info
    APP_NAME: str = "ML Engine API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MODELS_DIR: Path = BASE_DIR / "models"
    LOGS_DIR: Path = BASE_DIR / "logs"
    UPLOADS_DIR: Path = BASE_DIR / "uploads"
    
    # ML Engine path (your existing ml_engine)
    ML_ENGINE_PATH: str = os.getenv("ML_ENGINE_PATH", "/home/claude/kedro-ml-engine/src")
    
    # Job settings
    MAX_CONCURRENT_JOBS: int = 4
    JOB_TIMEOUT_SECONDS: int = 3600  # 1 hour
    
    # CORS
    CORS_ORIGINS: list = ["http://localhost:4200", "http://localhost:8080", "*"]
    
    class Config:
        env_file = ".env"


# Create settings instance
settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
