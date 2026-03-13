from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "gpt-4o-mini"
    chroma_persist_dir: str = ".chroma_db"
    top_k_retrieval: int = 6
    temperature: float = 0.0

    class Config:
        env_file = ".env"


settings = Settings()
