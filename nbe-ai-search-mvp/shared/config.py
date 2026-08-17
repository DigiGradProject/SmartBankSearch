from pathlib import Path
from typing import Literal, Self

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shared.retrieval_mode import RetrievalMode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    project_root: Path = Path(__file__).resolve().parents[1]
    documents_path: Path = project_root / "data" / "documents.json"
    cleaned_jsonl_path: Path = project_root.parent / "nbe-scrape-cleaner" / "output" / "documents.jsonl"
    rescrape_json_path: Path = project_root.parent / "output"
    scrape_root: Path = project_root.parent / "nbe_complete_scrape"
    chroma_path: Path = project_root / "data" / "chroma"
    # Bump when corpus schema/content changes materially.
    chroma_collection: str = "nbe_chunks_bge_m3_v6"
    intent_filter_confidence: float = 0.75
    intent_filter_enabled: bool = True
    # Staged metadata filter (analysis.md Phase A/B)
    filter_high_confidence: float = 0.90
    filter_min_hits_high_conf: int = 1
    filter_min_hits_low_conf: int = 3
    filter_allow_broad_fallback: bool = True
    # Retrieval control plane (replaces force_canonical_inject as the primary switch)
    retrieval_mode: Literal["PURE_SEMANTIC", "ENTERPRISE"] = RetrievalMode.ENTERPRISE.value
    business_rules_enabled: bool = True
    # Deprecated: hard document injection is removed. Kept for .env backward compat.
    # When True with legacy env, enables business_rules soft boosts only (no inject).
    force_canonical_inject: bool = False

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
    # Rare Tier-2 query rewrite model (feature-flagged)
    ollama_rewrite_model: str = "qwen3:4b"
    llm_tier2_enabled: bool = True
    llm_timeout_seconds: float = 90.0
    llm_fallback_enabled: bool = True

    # Enterprise feature flags (Phase 3–4)
    # Layered intent: BGE-M3 semantic primary, regex fallback (see intent_fallback.py)
    semantic_intent_enabled: bool = True
    semantic_intent_high_confidence: float = 0.90
    semantic_intent_regex_fallback_threshold: float = 0.60
    semantic_intent_min_score: float = 0.42
    minilm_intent_fallback_enabled: bool = False
    minilm_intent_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    gliner_enabled: bool = False
    gliner_model: str = "urchade/gliner_multi-v2.1"
    llm_rewrite_enabled: bool = False
    audit_log_enabled: bool = True
    audit_log_path: Path = project_root / "data" / "logs" / "search_audit.jsonl"

    # Retrieval contract
    rerank_pool_size: int = 20
    rerank_keep_size: int = 5
    eval_intent_min_accuracy: float = 0.90
    eval_url_min_hit_rate: float = 0.70

    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_enabled: bool = False

    # Semantic cache (Chroma similarity)
    semantic_cache_enabled: bool = True
    semantic_cache_collection: str = "nbe_semantic_cache_v7"
    semantic_cache_threshold: float = 0.95
    semantic_cache_ttl_seconds: int = 86400

    # Self-evaluation / compression / planner
    self_eval_enabled: bool = True
    context_compression_enabled: bool = True
    query_planner_enabled: bool = True
    query_planner_max_intents: int = 2

    analytics_log_enabled: bool = True
    analytics_log_path: Path = project_root / "data" / "logs" / "retrieval_analytics.jsonl"
    feedback_log_path: Path = project_root / "data" / "logs" / "feedback.jsonl"

    api_host: str = "0.0.0.0"
    api_port: int = 7000
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    @model_validator(mode="after")
    def _migrate_legacy_force_inject(self) -> Self:
        """Map deprecated force_canonical_inject → soft business rules (never hard inject)."""
        if self.force_canonical_inject and not self.business_rules_enabled:
            self.business_rules_enabled = True
        # Hard injection is permanently disabled regardless of legacy flag.
        self.force_canonical_inject = False
        return self

    @property
    def retrieval_mode_enum(self) -> RetrievalMode:
        return RetrievalMode(self.retrieval_mode)


settings = Settings()
