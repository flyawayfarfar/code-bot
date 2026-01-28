# app/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Switches
    # use_local_llm:         bool = True           # DEPRECATED/UNUSED (logic moved to main.py)
    # use_local_embeddings:  bool = True           # DEPRECATED
    
    # "ollama", "openai", "google"
    embedding_provider:    str  = "google"

    # Data / Vector DB
    data_dir:              str  = "data"
    chroma_dir:            str  = "chroma_db"
    chroma_dir_local: str = "chroma_db_ollama"
    chroma_dir_openai: str = "chroma_db_openai"
    chroma_dir_google: str = "chroma_googleai"
    k_neighbors:           int  = 5

    # Ollama
    ollama_base_url:       str  = "http://localhost:11434"
    ollama_model:          str  = "llama3.1:8b"
    ollama_embed_model:    str  = "nomic-embed-text"   # good local embedder
    local_embedding_model: str  = "nomic-embed-text"   # alias for older code

    # (optional) OpenAI fallback
    openai_api_key:        str | None = None

    # (optional) Google fallback
    google_api_key:        str | None = None
    google_model:          str = "gemini-2.5-flash-lite"

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def active_chroma_dir(self) -> str:
        if self.embedding_provider == "google":
            return self.chroma_dir_google
        if self.embedding_provider == "openai":
            return self.chroma_dir_openai
        return self.chroma_dir_local
    
settings = Settings()
