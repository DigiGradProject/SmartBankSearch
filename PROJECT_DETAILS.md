# NBE AI Search MVP: Detailed Project Explanation

This document provides a comprehensive overview of the NBE AI Search MVP project, detailing its purpose, overall flow, the models and algorithms employed, their rationale, and the function of each major directory and component.

## 1. Project Overview and Vision

The **NBE AI Search MVP** is an on-premises, Retrieval-Augmented Generation (RAG) search platform designed for the National Bank of Egypt website content. Its primary goal is to introduce an "AI Search Mode" that coexists with the traditional keyword search, providing natural-language, bilingual (Arabic/English), evidence-grounded answers with source citations. The entire stack is designed to be fully on-premises, ensuring no cloud dependency and no data leaving NBE infrastructure.

**Business Problem Addressed:** Traditional keyword search often fails to meet customer needs due to exact term matching limitations and the inability to synthesize answers across multiple pages. The AI Search MVP aims to bridge this gap by enabling natural questions and providing direct, cited answers.

**Scope of MVP:**
The MVP focuses on the AI Search pipeline starting *after* content acquisition (scraping and initial extraction are external to this project, handled by `nbe-scrape-cleaner`). It takes a stable collection of JSON documents as input and processes them to enable a sophisticated, bilingual semantic search experience.

## 2. Overall Project Flow

The project consists of two main conceptual phases: **Data Preparation** (handled by `nbe-scrape-cleaner`) and **AI Search & Retrieval** (handled by `nbe-ai-search-mvp`).

### Phase 1: Data Preparation (`nbe-scrape-cleaner`)

This component is responsible for transforming raw scraped website content into a clean, structured format suitable for ingestion into the AI search system.

1.  **Input:** Takes a ZIP archive (e.g., `nbe_complete_scrape.zip`) containing raw scraped data, likely a collection of HTML files or semi-processed JSON.
2.  **Cleaning Pipeline:** The `clean_pipeline.py` script orchestrates various cleaning modules:
    *   **Boilerplate Removal (`cleaner/boilerplate.py`):** Identifies and removes common, non-content elements like headers, footers, navigation, and advertisements from web pages.
    *   **Noise Removal (`cleaner/noise.py`):** Filters out irrelevant text or HTML elements that don't contribute to the core content.
    *   **Table Processing (`cleaner/tables.py`):** Extracts and structures data from HTML tables, preserving their integrity.
    *   **Document Classification (`cleaner/doc_classifier.py`):** Categorizes documents based on their content or type (e.g., "news article," "product page").
    *   **Duplicate Detection (`cleaner/duplicates.py`):** Identifies and handles redundant content to prevent over-representation in the index.
    *   **ID Management (`cleaner/ids.py`):** Assigns and manages unique identifiers for each processed document.
    *   **Title Extraction (`cleaner/title.py`):** Extracts and standardizes document titles for better indexing and display.
    *   **Archiving (`cleaner/archive.py`):** Manages the storage of cleaned data.
3.  **Output:** Generates `output/documents.jsonl` (one cleaned document per line for streaming efficiency and avoiding OS path-length limits) and `report.md` (detailing size reduction and near-duplicate reviews).

### Phase 2: AI Search & Retrieval (`nbe-ai-search-mvp`)

This is the core AI search application, handling ingestion of prepared data, query processing, retrieval, answer generation, and API exposure.

#### 2.1 Ingestion Pipeline (Offline / Batch)

This process transforms the cleaned JSON documents into an indexable format for fast and relevant retrieval.

1.  **Load Documents:** The `ingestion/document-processing-service/` (conceptually mapped to `ingestion/` in the code) loads `output/documents.jsonl`.
2.  **Document Processing & Normalization:**
    *   Strips residual HTML/markup and normalizes whitespace.
    *   Applies **Arabic-specific normalization (`shared/arabic_normalize.py`)** (ADR-03) using custom modules (potentially `pyarabic` or `CAMeL Tools`). This ensures consistency across different orthographic variations (e.g., Alef forms, Taa Marbuta) and Egyptian colloquialisms, crucial for accurate Arabic search.
3.  **Structure-Aware Chunking (`ingestion/chunking-service/`):** Documents are split into smaller, manageable chunks (typically 300-500 tokens with 50-token overlap). This chunking is *structure-aware* (ADR-03, FR-03), meaning it respects HTML headings, lists, and tables to prevent critical information from being split across chunks, which is vital for banking content like fee schedules.
4.  **Embedding Generation (`ingestion/embedding-service/`):**
    *   For each chunk, a **`content_hash` (`shared/content_hash.py`)** is computed. This enables **idempotent upserting** (ADR-02, FR-11): if a chunk's content (and thus its hash) hasn't changed, it's skipped during re-ingestion, saving computation.
    *   **Embeddings** are generated for each chunk.
5.  **Vector Database Storage:** The chunks and their associated embeddings (along with metadata like `document_id`, `chunk_id`, `url`, `title`, `language`) are stored in a **Vector Database (VDB)**. For the MVP, **ChromaDB** is used for easier local setup, with a plan to transition to Milvus for larger scale (ADR-04).

#### 2.2 Query-Time Flow (Online)

This is the real-time process when a user submits a query via the AI Search UI.

1.  **User Interaction:** A user selects "AI Search Mode" on the NBE website and inputs a natural language query.
2.  **API Gateway (`services/api/main.py`, `services/api/orchestrator.py`):** The query is received by the REST API, which acts as the entry point and orchestrates calls to various backend services.
3.  **Search Service (`services/search_service/search.py`):**
    *   **Language Detection & Normalization:** The user's query language is detected, and the query text undergoes consistent Arabic normalization (if applicable) using `shared/arabic_normalize.py`.
    *   **Query Embedding:** The normalized query is embedded using the same embedding model as the documents.
    *   **Semantic Retrieval:** The embedded query is used to perform a vector similarity search against the VDB to retrieve the top `k` most relevant document chunks (FR-05).
    *   **Hybrid Retrieval (`services/search_service/hybrid_retriever.py`):** The system likely combines lexical (keyword-based) and semantic (vector-based) search results to ensure comprehensive retrieval.
    *   **Reranking (`services/search_service/reranker.py`):** The initial set of retrieved chunks is re-ordered using a more sophisticated model to prioritize the most relevant ones. The `bge-reranker-v2-m3` is specifically chosen to pair with the `BGE-M3` embeddings for this task.
    *   **Other Search Modifiers:**
        *   `services/search_service/autocomplete.py` and `suggestions.py`: Provide real-time query assistance.
        *   `services/search_service/query_expand.py` and `synonyms.py`: Broaden the search with related terms.
        *   `services/search_service/intent_classifier.py` and `services/rag/semantic_intent.py`: Attempt to understand the user's intent to tailor search results or RAG behavior.
        *   `services/search_service/account_rank.py`, `keyword_rank.py`, `intent_boost.py`: Apply specific ranking signals.
        *   `services/rag/metadata_filter.py`: Filters based on document metadata.
4.  **Context Builder (`services/context_builder/builder.py`, `compressor.py`):** Assembles a bounded context window for the LLM from the top-ranked retrieved chunks. `compressor.py` can be used to reduce context size if needed, prioritizing essential information to fit within the LLM's token limits. Citations are mapped during this stage.
5.  **Local LLM Service (`services/llm_service/llm.py`):**
    *   The bounded context and the original query are sent to a locally hosted Large Language Model (LLM).
    *   **LLM Tiering (ADR-05):** For the MVP, a single-tier model is recommended, but the setup supports:
        *   **Tier 1 LLM:** `Qwen3-8B` via Ollama (for fast responses, especially for Arabic/colloquial queries).
        *   **Tier 2 LLM:** `Qwen3-14B` (or `Qwen3-30B`) via Ollama (for deeper reasoning on complex queries).
    *   **Grounded-Only Answers (ADR-06, Section 9.1):** The LLM is strictly instructed to generate an answer *only* from the provided context and to state uncertainty if the context is insufficient. This prevents hallucinations.
6.  **Confidence Gate (`services/rag/confidence.py`):** After the LLM generates a response, a composite confidence score is calculated (based on retrieval similarity, reranker scores, and the LLM's own assessment of context sufficiency or uncertainty).
7.  **Response Handling:**
    *   **Confident Answer:** If the confidence is above a threshold (FR-08, FR-10), the generated answer, along with **source page citations (title + URL)** (FR-09), is returned to the user.
    *   **Abstention:** If confidence is below the threshold, an explicit "no confident answer found" message is returned instead of a speculative response (ADR-06).
    *   **Extractive Fallback:** If the local LLM (Ollama) is unavailable, the system defaults to an extractive fallback, grounding answers purely in the retrieved chunks.
8.  **Frontend (`frontend/ai-search-toggle/`):** The React-based frontend receives the API response and renders the answer, citations, or the "no confident answer" message to the user.

## 3. Key Models and Algorithms Used and Their Rationale

### Data Preparation (`nbe-scrape-cleaner`)

*   **Document Classification (`cleaner/doc_classifier.py`)**:
    *   **Model/Algorithm:** Could involve rule-based systems (keyword matching, URL patterns) or machine learning classifiers (e.g., Naive Bayes, SVM, or small neural networks) trained on document features (TF-IDF, word embeddings).
    *   **Rationale:** To categorize documents (e.g., "product page," "policy") for targeted processing and more informed downstream retrieval, improving overall relevance and data utility.
*   **Boilerplate/Noise Removal (`cleaner/boilerplate.py`, `cleaner/noise.py`)**:
    *   **Model/Algorithm:** Heuristic rules based on HTML DOM analysis, text density algorithms (e.g., similar to Readability.js), or potentially supervised learning for more complex noise patterns.
    *   **Rationale:** To ensure only main content is ingested, removing distracting and irrelevant elements. This significantly improves the quality of data for embedding and LLM processing, reducing noise and improving relevance.
*   **Duplicate Detection (`cleaner/duplicates.py`)**:
    *   **Model/Algorithm:** Hashing (MD5, SHA256) for exact duplicates; MinHashing or Locality Sensitive Hashing (LSH) for near-duplicate detection.
    *   **Rationale:** To avoid redundant indexing, saving storage, improving search result diversity, and ensuring LLMs don't process identical information multiple times, thus increasing efficiency and relevance.

### AI Search MVP (`nbe-ai-search-mvp`)

*   **Arabic Normalization (`shared/arabic_normalize.py`)**:
    *   **Model/Algorithm:** Rule-based text preprocessing using specific Arabic NLP libraries (`pyarabic` or `CAMeL Tools`). Handles standardizing variations in Hamza, Alef, Yeh, Taa Marbuta, and removing diacritics/elongations.
    *   **Rationale (ADR-03):** Essential for Arabic language search. Orthographic variations can cause different forms of the same word to be treated as distinct by search algorithms. Consistent normalization ensures accurate matching and retrieval for both Arabic and mixed-language content.
*   **Structure-Aware Chunking (conceptualized in `ingestion/chunking-service/`)**:
    *   **Model/Algorithm:** Intelligent text splitting that considers the document's HTML structure (headings, lists, tables) rather than just arbitrary character limits. Fixed token-size fallback with overlap is also used (e.g., ~300-500 tokens with ~50-token overlap).
    *   **Rationale (ADR-03):** Crucial for preserving the integrity of structured content, especially in banking documents (e.g., not splitting a table row mid-sentence). This prevents loss of context and ensures high-quality RAG results.
*   **Embedding Model (`BAAI/bge-m3`)**:
    *   **Model/Algorithm:** A multilingual dense + sparse hybrid embedding model. `BGE-M3` is known for its strong performance across many languages, including Arabic, and its ability to provide both semantic (dense) and keyword-like (sparse) representations.
    *   **Rationale:** Provides robust multilingual capabilities, allowing cross-language semantic matching (e.g., an English query retrieving a relevant Arabic document). The hybrid nature leverages the strengths of both semantic and keyword matching, leading to more comprehensive and relevant retrieval. This model is self-hosted due to the "no cloud AI APIs" requirement.
*   **Vector Database (ChromaDB / Milvus)**:
    *   **Model/Algorithm:** Specialized database for storing and querying high-dimensional vectors, typically using approximate nearest neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World) for efficient similarity searches.
    *   **Rationale (ADR-04):** Enables fast and scalable retrieval of semantically similar chunks based on query embeddings. ChromaDB is used for MVP simplicity, with Milvus as a planned scalable alternative.
*   **Query Expansion (`services/search_service/query_expand.py`)**:
    *   **Model/Algorithm:** Rule-based expansions, synonym dictionaries (`services/search_service/synonyms.py`), or potentially embedding-based nearest neighbors to find semantically similar terms.
    *   **Rationale:** To broaden the search query's scope and improve recall by finding documents that might use synonyms or related phrases not explicitly in the user's original query.
*   **Intent Classification (`services/search_service/intent_classifier.py`, `services/rag/semantic_intent.py`)**:
    *   **Model/Algorithm:** Text classification models (e.g., FastText, BERT, or simpler models like Logistic Regression with TF-IDF features). Could also involve few-shot learning with LLMs.
    *   **Rationale:** To understand the user's underlying goal (e.g., "find specific information," "ask a comparative question") and tailor the retrieval or RAG generation strategy accordingly, providing more precise and relevant answers.
*   **Reranker Model (`BAAI/bge-reranker-v2-m3`)**:
    *   **Model/Algorithm:** A cross-encoder model specifically designed to re-score the top `N` documents returned by the initial retrieval phase. It takes the query and each retrieved document/chunk as a pair to provide a more fine-grained relevance score.
    *   **Rationale:** Significantly improves the precision of retrieved results by re-ordering them based on a deeper understanding of query-document relevance. This model is chosen to be compatible with `BGE-M3` embeddings.
*   **Large Language Models (LLMs) (`services/llm_service/llm.py`)**:
    *   **Model/Algorithm:** Transformer-based generative models. For this MVP, **Qwen3-8B** (Tier 1) and **Qwen3-14B** (Tier 2) are used via **Ollama** for local serving.
    *   **Rationale (ADR-05):** The core of the RAG system. These models synthesize coherent, human-like answers from the retrieved context. The tiered approach allows for a balance between speed (8B model) and deeper reasoning (14B model) for complex queries, all while adhering to the on-premises requirement.
*   **Confidence Gate (`services/rag/confidence.py`)**:
    *   **Model/Algorithm:** A composite scoring mechanism that considers factors like retrieval similarity scores, reranker scores, and the LLM's own assessment of context sufficiency or uncertainty.
    *   **Rationale (ADR-06, FR-10):** A critical safety mechanism, especially in a banking context. It prevents the LLM from generating speculative or hallucinated answers when the retrieved evidence is weak or insufficient. If confidence is below a threshold, the system explicitly abstains from answering.
*   **Context Compression (`services/context_builder/compressor.py`)**:
    *   **Model/Algorithm:** May involve summarization techniques (extractive or abstractive), redundancy detection, or importance-based chunk selection (e.g., using a smaller re-ranker to identify the most salient sentences/chunks).
    *   **Rationale:** To reduce the token count sent to the LLM, which optimizes latency and cost, and ensures the context fits within the LLM's maximum token window without losing critical information.
*   **Idempotent Upsert (`shared/content_hash.py`)**:
    *   **Model/Algorithm:** Uses cryptographic hashing (e.g., SHA256) of document chunk content. During ingestion, if a chunk's hash matches an existing one in the VDB, it's skipped.
    *   **Rationale (ADR-02, FR-11):** Ensures that re-running the ingestion pipeline is safe, efficient, and doesn't create duplicate entries or unnecessarily re-embed unchanged content, saving computational resources (especially GPU time).

## 4. Explanation of Folder Structure and Purpose

The project is logically organized into major directories, each serving a distinct purpose in the overall architecture.

### Root Level

*   `.git/`, `.gitignore`: Standard Git version control files.
*   `diagnose_page.py`: A utility script likely used for analyzing web page structure or issues, possibly for improving scraping or cleaning.
*   `nbe-ai-search-mvp/`: Contains the core AI search application.
*   `nbe-scrape-cleaner/`: Houses the data scraping and cleaning utilities.
*   `output/`: General directory for storing output files, such as scraped JSONs or processed data.
*   `plan-mvp.md`: This very document, detailing the implementation plan and architecture.
*   `priority_urls.txt`: A list of URLs that are prioritized for scraping, guiding the data acquisition process.
*   `README.md`: The main project README, providing a high-level overview.
*   `rescrape_playwright.py`: A script using Playwright for rescraping, likely for refreshing data or handling specific URL updates.

### `nbe-scrape-cleaner/`

Dedicated to data acquisition and preparation.

*   `cleaner/`: Core modules for cleaning and processing scraped data. Each Python file here (`archive.py`, `boilerplate.py`, `build.py`, `doc_classifier.py`, `duplicates.py`, `ids.py`, `noise.py`, `report.py`, `tables.py`, `title.py`) implements a specific step in the cleaning pipeline.
*   `clean_pipeline.py`: The main script that orchestrates the execution of the cleaning modules.
*   `output/`: Stores processed data specifically from the cleaning pipeline (e.g., `documents.jsonl`).
*   `README.md`: Provides specific instructions and details for the scraper and cleaner.
*   `report.md`: A markdown file containing a report on the cleaning process (e.g., size reduction, near-duplicate list).
*   `tests/`: Unit tests for the cleaning functionalities (`test_cleaning.py`).

### `nbe-ai-search-mvp/`

The main directory for the AI-powered search MVP.

*   `.env.example`: An example file for environment variables, outlining necessary configurations.
*   `data/`: Intended for data files, which may include schemas, configuration for the vector database, or other static data.
    *   `schemas/`: Defines JSON contracts for documents and chunks.
*   `docs/`: Project documentation, including Architecture Decision Records (ADR).
*   `frontend/`: The client-side application (React) for the AI Search UI toggle.
*   `infra/`: Infrastructure-as-code or deployment-related files (e.g., Docker Compose, Kubernetes configurations).
*   `ingestion/`: Handles the process of taking cleaned data and preparing it for search. This conceptually contains the `document-processing-service`, `chunking-service`, and `embedding-service`.
*   `observability/`: Contains configurations and tools for monitoring and logging (e.g., Prometheus dashboards, alerting rules).
*   `plan.md`: A planning document specific to the AI search MVP (this was identified as `plan-mvp.md` in the root).
*   `README.md`: Provides an overview, features, and quick start guide for the AI Search MVP.
*   `requirements.txt`: Lists Python dependencies for the AI Search MVP.
*   `scripts/`: Utility scripts for various tasks, such as exporting documents or running ingestion.
*   `services/`: Core backend services that implement the search and RAG logic.
    *   `api/`: Defines the REST API endpoints (`main.py`) and orchestrates calls (`orchestrator.py`).
    *   `context_builder/`: Logic for assembling and compressing context for the LLM (`builder.py`, `compressor.py`).
    *   `feedback/`: Components for user feedback collection and storage (`service.py`, `store.py`).
    *   `llm_service/`: Interfaces with the local Large Language Models (`llm.py`).
    *   `rag/`: Components related to Retrieval Augmented Generation, including analytics, audit, business rules, confidence scoring, entity extraction, explainability, hybrid ranking, query planning, and rewriting.
    *   `search_service/`: Core search functionalities such as autocomplete, ranking (account, keyword), intent classification, hybrid retrieval, reranking, and query expansion.
*   `shared/`: Common utilities, configurations, and data models used across multiple services.
    *   `arabic_normalize.py`: Module for Arabic text normalization.
    *   `config.py`: Centralized application configurations.
    *   `content_hash.py`: Module for generating content hashes for idempotency.
    *   `document_quality.py`: Utilities for evaluating document quality.
    *   `logging.py`: Centralized logging configuration.
    *   `retrieval_mode.py`: Defines different retrieval strategies.
    *   `schemas.py`: Pydantic models for data structures.
    *   `url_canonical.py`: Utilities for URL canonicalization.
*   `tests/`: Comprehensive test suite for unit, integration, and retrieval aspects of the MVP.

## 5. Key Architectural Decisions (ADRs)

The `plan-mvp.md` document includes several Architecture Decision Records (ADRs) that explain critical choices and their rationale:

*   **ADR-01: Exclude live/time-sensitive data from this MVP's RAG pipeline:** The MVP focuses solely on static, public web content to avoid risks of stale answers with false confidence. Live data sources are deferred to future phases via a separate, non-embedding tool-calling path.
*   **ADR-02: Idempotent ingestion keyed on `content_hash`:** To prevent redundant re-embedding and vector storage, re-ingestion safely skips unchanged content based on a computed `content_hash` per chunk.
*   **ADR-03: Arabic-specific normalization as a distinct pipeline stage:** A dedicated normalization step for Arabic text (handling orthographic variations) is crucial for improving retrieval quality due to the unique characteristics of the language.
*   **ADR-04: Vector database choice scoped to MVP:** ChromaDB is used for local MVP setup due to ease of use, with the understanding that a more scalable solution like Milvus will be considered for a full-platform deployment.
*   **ADR-05: Single-tier local LLM for MVP:** To simplify the MVP, a single, appropriately sized local LLM (Qwen3-8B/14B) is used, deferring more complex tiered LLM routing to a post-MVP phase when actual query complexity data is available.
*   **ADR-06: Confidence gate enforced as a hard requirement:** In a banking context, hallucinated answers pose significant risks. Therefore, the system is designed to explicitly abstain (return "no confident answer") rather than guess when retrieved evidence is weak, making this a first-class, tested requirement.

This detailed explanation should provide a thorough understanding of the NBE AI Search MVP project.