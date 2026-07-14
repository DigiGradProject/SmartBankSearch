from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[1]
    documents_path: Path = project_root / "data" / "documents.json"
    cleaned_jsonl_path: Path = project_root.parent / "nbe-scrape-cleaner" / "output" / "documents.jsonl"
    rescrape_json_path: Path = project_root.parent / "output"
    scrape_root: Path = project_root.parent / "nbe_complete_scrape"
    chroma_path: Path = project_root / "data" / "chroma"
    # Bump when corpus schema/content changes materially.
    chroma_collection: str = "nbe_chunks_bge_m3_v3"
    intent_filter_confidence: float = 0.75
    intent_filter_enabled: bool = True

    bm25_enabled: bool = True
    bm25_index_path: Path = project_root / "data" / "bm25" / "corpus.pkl"
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = 60

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_enabled: bool = True
    hybrid_alpha: float = 0.7  # dense weight; (1 - alpha) = sparse/lexical
    retrieval_candidate_multiplier: int = 4

    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 50
    retrieval_top_k: int = 8
    context_max_tokens: int = 2000
    confidence_threshold: float = 0.42
    citation_min_score: float = 0.45
    max_citations: int = 3

    ollama_base_url: str = "http://localhost:11434"
    # Tier 1: fast/cheap Arabic-capable default
    ollama_model: str = "qwen3:8b"
    # Tier 2: deeper reasoning when needed
    ollama_model_tier2: str = "qwen3:14b"
    llm_tier2_enabled: bool = True
    llm_timeout_seconds: float = 90.0
    llm_fallback_enabled: bool = True

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 7000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"


settings = Settings()
