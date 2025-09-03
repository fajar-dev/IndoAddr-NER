from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_dir: str = "models/indoaddr-ner"  # path to fine-tuned model
    model_name: str = "indobenchmark/indobert-base-p1"
    device: str = "cuda" if __import__('os').environ.get('USE_CUDA','0')=='1' else "cpu"
    aggregation_strategy: str = "simple"

    class Config:
        env_file = ".env"

settings = Settings()
