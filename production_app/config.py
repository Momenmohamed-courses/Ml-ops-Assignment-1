import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    model_path: Path
    log_level: str
    app_env: str
    test_size: float
    random_seed: int
    max_iter: int

def load_settings()-> Settings:
    test_size=float(os.getenv("TEST_SIZE","0.2"))
    if test_size <= 0 or test_size > 1:
        raise ValueError(f"invalid TEST_SIZE: {test_size}")
    
    return Settings(
        model_path=Path(os.getenv("MODEL_PATH", "models/pipeline.joblib")),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        app_env=os.getenv("APP_ENV", "development"),
        test_size=test_size,
        random_seed=int(os.getenv("RANDOM_SEED", "42")),
        max_iter=int(os.getenv("MAX_ITER", "1000")),
    )
settings=load_settings()