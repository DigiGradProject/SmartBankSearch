# Project Overview

This project appears to be a comprehensive AI-powered search solution, combining web scraping, data cleaning, and an advanced AI search MVP. It's structured into two main components: `nbe-scrape-cleaner` for data acquisition and preparation, and `nbe-ai-search-mvp` for the core AI search functionalities, including ingestion, retrieval, and API services.

## Project Structure

This section details the purpose of each main directory and significant files.

### Root Level Files and Directories:

*   `.continue/`: Likely a configuration directory for a development environment or CI/CD.
*   `.git/`: Git version control system directory.
*   `.gitignore`: Specifies intentionally untracked files to ignore.
*   `diagnose_page.py`: A script likely used for diagnosing issues with web pages, possibly related to scraping or content extraction.
*   `nbe-ai-search-mvp/`: **Main AI Search Application**. Contains the core logic for the AI-powered search MVP, including frontend, backend services, ingestion pipelines, and tests.
*   `nbe-scrape-cleaner/`: **Data Scraper and Cleaner**. Handles web scraping, data extraction, and cleaning of raw scraped content.
*   `output/`: Directory for storing output files, likely from scraping or processing. The presence of numerous `.json` files suggests scraped and processed data.
*   `plan-mvp.md`: A markdown document outlining the plan for the Minimum Viable Product.
*   `priority_urls.txt`: A list of URLs that are prioritized for scraping.
*   `rescrape_playwright.py`: A Python script, likely using Playwright, for rescraping URLs, possibly for data refresh or error recovery.

### `nbe-scrape-cleaner/` Directory:

This directory is dedicated to obtaining and preparing data.

*   `cleaner/`: Contains the core logic for cleaning and processing scraped data.
    *   `archive.py`: Manages archiving of scraped data.
    *   `boilerplate.py`: Handles boilerplate content detection and removal from web pages.
    *   `build.py`: Likely responsible for building or structuring the cleaned data.
    *   `doc_classifier.py`: Classifies documents based on their content or type.
    *   `duplicates.py`: Identifies and handles duplicate content.
    *   `ids.py`: Manages unique identifiers for documents.
    *   `noise.py`: Detects and removes noisy or irrelevant content.
    *   `report.py`: Generates reports on the cleaning process.
    *   `tables.py`: Processes and extracts data from HTML tables.
    *   `title.py`: Extracts and standardizes document titles.
    *   `__init__.py`: Makes `cleaner` a Python package.
*   `clean_pipeline.py`: The main script orchestrating the data cleaning pipeline.
*   `output/`: Stores output specifically from the cleaning process.
*   `README.md`: Specific README for the scraper and cleaner.
*   `report.md`: A markdown report, possibly detailing the results of cleaning or scraping.
*   `tests/`: Unit tests for the `nbe-scrape-cleaner` component.
    *   `test_cleaning.py`: Tests for the data cleaning functionalities.

### `nbe-ai-search-mvp/` Directory:

This directory houses the core AI search application.

*   `.env.example`: Example environment variables for configuration.
*   `data/`: Placeholder for data files, potentially processed data or models.
*   `docs/`: Documentation for the AI search MVP.
*   `frontend/`: Frontend application code (e.g., React, Vue, Angular).
*   `infra/`: Infrastructure-as-code or deployment related files.
*   `ingestion/`: Handles the ingestion of cleaned data into the search system.
*   `observability/`: Contains tools or configurations for monitoring and logging.
*   `plan.md`: Planning document specific to the AI search MVP.
*   `README.md`: Specific README for the AI search MVP.
*   `requirements.txt`: Python dependencies for the project.
*   `scripts/`: Utility scripts for various tasks.
*   `services/`: Core backend services for the search application.
    *   `api/`: Defines the main API endpoints.
        *   `main.py`: Entry point for the API.
        *   `orchestrator.py`: Orchestrates calls between different services.
    *   `context_builder/`: Builds context for RAG (Retrieval Augmented Generation).
        *   `builder.py`: Logic for context construction.
        *   `compressor.py`: Compresses context to fit LLM token limits.
    *   `feedback/`: Handles user feedback mechanisms.
        *   `service.py`: Feedback collection service.
        *   `store.py`: Stores feedback data.
    *   `llm_service/`: Interfaces with Large Language Models.
        *   `llm.py`: LLM integration logic.
    *   `rag/`: Retrieval Augmented Generation components. This is a crucial part of the AI search.
        *   `analytics.py`: RAG-related analytics.
        *   `audit.py`: Auditing for RAG responses.
        *   `business_rules.py`: Applies business rules to RAG.
        *   `confidence.py`: Assesses confidence in RAG outputs.
        *   `critical_rules.py`: Handles critical rules for RAG.
        *   `decision_engine.py`: Makes decisions based on RAG results.
        *   `entity_extractor.py`: Extracts entities from text.
        *   `explainability.py`: Provides explanations for RAG results.
        *   `hybrid_rank.py`: Hybrid ranking for RAG results.
        *   `intent_fallback.py`: Handles fallback for intent recognition.
        *   `intent_prototypes.py`: Defines prototypes for intents.
        *   `language.py`: Language processing for RAG.
        *   `metadata_filter.py`: Filters based on metadata.
        *   `metrics.py`: RAG performance metrics.
        *   `query_planner.py`: Plans queries for RAG.
        *   `query_rewrite.py`: Rewrites queries for better RAG performance.
        *   `query_understanding.py`: Understands user queries.
        *   `response_formatter.py`: Formats RAG responses.
        *   `self_eval.py`: Self-evaluation of RAG.
        *   `semantic_cache.py`: Caches semantic queries/responses.
        *   `semantic_intent.py`: Semantic intent recognition.
        *   `synonym_ontology.py`: Manages synonyms and an ontology.
    *   `search_service/`: Core search functionalities.
        *   `account_rank.py`: Ranks results based on account relevance.
        *   `autocomplete.py`: Provides search autocomplete suggestions.
        *   `card_catalog.py`: Manages a catalog of search results/cards.
        *   `certificate_catalog.py`: Manages a catalog of certificates.
        *   `hybrid_retriever.py`: Combines multiple retrieval methods.
        *   `intent_boost.py`: Boosts results based on detected intent.
        *   `intent_classifier.py`: Classifies user intent.
        *   `keyword_rank.py`: Ranks results based on keywords.
        *   `language.py`: Language-specific search processing.
        *   `product_detail.py`: Handles product detail retrieval.
        *   `query_catalog.py`: Manages a catalog of queries.
        *   `query_expand.py`: Expands user queries.
        *   `rate_guidance.py`: Provides guidance on rates (if financial context).
        *   `reranker.py`: Reranks initial search results.
        *   `search.py`: Main search logic.
        *   `suggestions.py`: Provides search suggestions.
        *   `synonyms.py`: Manages search synonyms.
    *   `__init__.py`: Makes `services` a Python package.
*   `shared/`: Shared utilities, configurations, and schemas used across services.
    *   `arabic_normalize.py`: Normalization for Arabic text.
    *   `config.py`: Application configurations.
    *   `content_hash.py`: Generates content hashes.
    *   `document_quality.py`: Evaluates document quality.
    *   `logging.py`: Centralized logging configuration.
    *   `retrieval_mode.py`: Defines retrieval modes.
    *   `schemas.py`: Data schemas (e.g., Pydantic models).
    *   `url_canonical.py`: Canonicalization of URLs.
    *   `__init__.py`: Makes `shared` a Python package.
*   `tests/`: Contains various tests for the AI search MVP.
    *   `integration/`: Integration tests.
        *   `test_retrieval_modes.py`: Tests different retrieval modes.
        *   `test_search_api_enterprise.py`: Tests the enterprise search API.
    *   `retrieval/`: Retrieval-specific tests and data.
        *   `golden_set.jsonl`: Golden set for retrieval evaluation.
        *   `validation_set.jsonl`: Validation set for retrieval evaluation.
    *   `unit/`: Unit tests for individual components. (Many specific test files listed, covering various modules).

## Overall Project Flow

The project flow can be broken down into several main stages:

1.  **Data Acquisition (Scraping):**
    *   `rescrape_playwright.py` or other unlisted scraping scripts are used to fetch web content.
    *   `priority_urls.txt` guides which URLs to scrape first.
    *   Raw scraped data might be stored in the `output/` directory.

2.  **Data Cleaning and Preprocessing (`nbe-scrape-cleaner`):**
    *   The raw scraped data is fed into the `nbe-scrape-cleaner` pipeline (`clean_pipeline.py`).
    *   Various modules within `nbe-scrape-cleaner/cleaner/` process the data:
        *   `boilerplate.py`: Removes common webpage boilerplate (headers, footers, navigation).
        *   `noise.py`: Filters out irrelevant text or HTML elements.
        *   `tables.py`: Extracts structured data from HTML tables.
        *   `doc_classifier.py`: Categorizes documents to enable targeted processing or retrieval.
        *   `duplicates.py`: Eliminates redundant content.
        *   `title.py`: Standardizes document titles for better indexing and display.
        *   `ids.py`: Assigns unique identifiers.
        *   `archive.py`: Manages storage of cleaned data.
    *   Reports (`report.md`) are generated to monitor cleaning effectiveness.
    *   Cleaned data is then ready for ingestion.

3.  **Data Ingestion (`nbe-ai-search-mvp/ingestion`):**
    *   Cleaned data is taken from the `nbe-scrape-cleaner` output and indexed into a search engine or vector database. This stage might involve:
        *   **Chunking:** Breaking down large documents into smaller, manageable chunks for retrieval (likely handled in `nbe-ai-search-mvp/tests/unit/test_chunking.py` suggests this is a feature).
        *   **Embedding Generation:** Converting text chunks into numerical vector representations using embedding models.
        *   **Metadata Extraction:** Extracting relevant metadata for filtering and ranking.

4.  **AI Search and Retrieval (`nbe-ai-search-mvp/services/search_service` and `nbe-ai-search-mvp/services/rag`):**
    *   **User Query Processing:**
        *   `autocomplete.py`, `suggestions.py`: Assist users with query formulation.
        *   `query_expand.py`, `synonyms.py`: Enhance the initial query for broader retrieval.
        *   `arabic_normalize.py`: If applicable, normalizes Arabic text for consistent search.
        *   `intent_classifier.py`, `semantic_intent.py`: Understands the user's underlying intent behind the query.
        *   `query_understanding.py`, `query_planner.py`: Deeply analyzes the query and plans the best retrieval strategy.
    *   **Retrieval:**
        *   `hybrid_retriever.py`: Combines keyword-based search (e.g., BM25) and semantic search (vector similarity) to fetch relevant documents/chunks.
        *   `metadata_filter.py`: Filters retrieved documents based on specified metadata.
    *   **Ranking:**
        *   `reranker.py`: Re-orders initial retrieval results based on relevance.
        *   `account_rank.py`, `keyword_rank.py`, `intent_boost.py`: Apply specific ranking signals (e.g., personalized ranking, keyword matching, intent-based boosting).
    *   **Context Building (`nbe-ai-search-mvp/services/context_builder`):**
        *   `builder.py`: Assembles the retrieved chunks into a coherent context for the LLM.
        *   `compressor.py`: Reduces context size if needed, to fit within LLM token limits.
    *   **Generative AI (RAG - `nbe-ai-search-mvp/services/rag` and `nbe-ai-search-mvp/services/llm_service`):**
        *   `llm.py`: Interfaces with the actual Large Language Model.
        *   The assembled context and user query are sent to the LLM.
        *   The LLM generates a response, potentially drawing upon:
            *   `business_rules.py`, `critical_rules.py`: Ensures responses adhere to predefined guidelines.
            *   `confidence.py`: Assesses the reliability of the generated response.
            *   `explainability.py`: Provides insights into *why* a particular answer was given.
            *   `self_eval.py`: The LLM might perform self-correction or evaluation.
            *   `decision_engine.py`: Guides the LLM's response generation process.
    *   **Response Formatting:**
        *   `response_formatter.py`: Structures the LLM's output into a user-friendly format.

5.  **API and Frontend (`nbe-ai-search-mvp/services/api` and `nbe-ai-search-mvp/frontend`):**
    *   `nbe-ai-search-mvp/services/api/main.py` exposes the search and RAG functionalities via a RESTful API.
    *   `nbe-ai-search-mvp/services/api/orchestrator.py` manages the flow of requests through the various backend services.
    *   The `frontend/` application consumes this API to provide a user interface for search and interaction.

6.  **Feedback and Improvement (`nbe-ai-search-mvp/services/feedback`):**
    *   User interactions and feedback are captured via `service.py` and stored by `store.py`.
    *   This feedback is crucial for evaluating and improving the search system, potentially feeding into re-training or rule adjustments.

## Key Models and Algorithms

Given the file structure, here's a breakdown of anticipated models and algorithms, and their rationale:

### `nbe-scrape-cleaner` (Data Preparation)

*   **Document Classification (`cleaner/doc_classifier.py`):**
    *   **Algorithm/Model:** Could range from rule-based systems (e.g., keyword matching, URL patterns) to machine learning classifiers (e.g., Naive Bayes, SVM, or even simpler neural networks) trained on document features (TF-IDF, word embeddings).
    *   **Why:** To categorize documents (e.g., "product page," "news article," "policy document") for targeted processing, indexing, or to inform downstream retrieval strategies in the AI search. This helps in maintaining data quality and relevance.
*   **Boilerplate/Noise Removal (`cleaner/boilerplate.py`, `cleaner/noise.py`):**
    *   **Algorithm/Model:** Heuristic rules based on HTML tag analysis, DOM tree structure, text density algorithms (e.g., Readability.js inspired approaches, or custom algorithms like `trafilatura`'s content extraction). Potentially supervised learning for more complex noise patterns.
    *   **Why:** To extract only the main content from a webpage, removing headers, footers, advertisements, navigation, and other irrelevant elements. This improves the quality of data ingested into the search system and reduces noise for LLMs.
*   **Duplicate Detection (`cleaner/duplicates.py`):**
    *   **Algorithm/Model:** Hashing (e.g., MD5, SHA256) of document content, MinHashing for approximate similarity, or Locality Sensitive Hashing (LSH) for near-duplicate detection.
    *   **Why:** To prevent redundant content from being indexed, saving storage, improving search result diversity, and ensuring LLMs don't process the same information multiple times.

### `nbe-ai-search-mvp` (AI Search)

*   **Arabic Normalization (`shared/arabic_normalize.py`):**
    *   **Algorithm/Model:** Rule-based text preprocessing (e.g., standardizing different forms of Hamza, Alef, Yeh).
    *   **Why:** Essential for Arabic language processing to ensure that variations in character forms do not hinder search or matching algorithms, leading to more accurate retrieval.
*   **Query Expansion (`services/search_service/query_expand.py`):**
    *   **Algorithm/Model:** Rule-based expansions, synonym dictionaries (from `services/search_service/synonyms.py`), WordNet, embedding-based nearest neighbors (finding semantically similar terms), or LLM-generated expansions.
    *   **Why:** To broaden the search scope and retrieve documents that might not contain the exact query terms but are semantically relevant.
*   **Intent Classification (`services/search_service/intent_classifier.py`, `services/rag/semantic_intent.py`):**
    *   **Algorithm/Model:** Text classification models (e.g., FastText, BERT, RoBERTa, or simpler models like Logistic Regression with TF-IDF features). Could also involve few-shot learning with LLMs.
    *   **Why:** To understand the user's goal (e.g., "find a product," "ask a question," "compare rates") and tailor the search results or RAG response accordingly. This is crucial for guiding the `decision_engine.py` and `query_planner.py`.
*   **Hybrid Retrieval (`services/search_service/hybrid_retriever.py`, `services/rag/hybrid_rank.py`):**
    *   **Algorithm/Model:** Combines lexical search (e.g., BM25, TF-IDF) with semantic search (vector similarity using dense embeddings from models like Sentence-BERT, OpenAI embeddings, or custom fine-tuned models).
    *   **Why:** To leverage the strengths of both approaches: lexical search for keyword matching and semantic search for conceptual understanding, leading to more comprehensive and relevant results.
*   **Reranking (`services/search_service/reranker.py`):**
    *   **Algorithm/Model:** Typically uses a more sophisticated model (e.g., Cross-encoders like BERT, ELECTRA, or a neural network trained on relevance judgments) to re-score the top N documents returned by the initial retrieval phase. Learning-to-Rank algorithms (e.g., LambdaMART) could also be employed.
    *   **Why:** To refine the order of retrieved documents, prioritizing the most relevant ones for the LLM context or direct display, significantly improving final result quality.
*   **Large Language Models (LLMs) (`services/llm_service/llm.py`):**
    *   **Algorithm/Model:** Pre-trained transformer models (e.g., GPT series, Llama, Falcon, Mistral) accessed via APIs or hosted locally.
    *   **Why:** The core of the RAG system, used for generating coherent, contextually relevant, and human-like answers based on the retrieved documents and user query.
*   **Context Compression (`services/context_builder/compressor.py`):**
    *   **Algorithm/Model:** Summarization techniques (extractive or abstractive, potentially using smaller LLMs), redundancy removal, or importance-based chunk selection (e.g., using a smaller re-ranker to pick most salient sentences/chunks).
    *   **Why:** To reduce the number of tokens passed to the main LLM, optimizing cost and latency, and ensuring the context fits within the LLM's token window without losing critical information.
*   **Confidence Scoring (`services/rag/confidence.py`):**
    *   **Algorithm/Model:** Could involve several heuristics: checking for hallucinations (e.g., comparing LLM output against retrieved sources), measuring consistency across multiple retrieved documents, using token probabilities from the LLM, or training a separate classifier to predict response quality.
    *   **Why:** To provide an indication of how reliable the generated answer is, allowing the system to flag potentially uncertain or hallucinated responses to the user or for human review.
*   **Entity Extraction (`services/rag/entity_extractor.py`):**
    *   **Algorithm/Model:** Named Entity Recognition (NER) models (e.g., SpaCy, Stanford NER, or fine-tuned BERT-like models).
    *   **Why:** To identify and extract key entities (e.g., product names, dates, organizations, locations) from queries or documents, which can be used for metadata filtering, structured retrieval, or enhancing LLM understanding.

## How to Set Up and Run (Placeholder)

1.  **Clone the repository:**
    ```bash
    git clone [repository_url]
    cd [project_directory]
    ```
2.  **Set up `nbe-scrape-cleaner`:**
    *   Install dependencies (`pip install -r requirements.txt` if available).
    *   Run scraping and cleaning pipelines (e.g., `python clean_pipeline.py`).
3.  **Set up `nbe-ai-search-mvp`:**
    *   Set up environment variables (copy `.env.example` to `.env` and fill it).
    *   Install dependencies (`pip install -r nbe-ai-search-mvp/requirements.txt`).
    *   Run ingestion process.
    *   Start the API service (e.g., `python nbe-ai-search-mvp/services/api/main.py`).
    *   Set up and run the `frontend/` application.

Detailed instructions would be in the respective `README.md` files within `nbe-scrape-cleaner` and `nbe-ai-search-mvp`.

## Contributing (Placeholder)

Information on how to contribute to the project.

## License (Placeholder)

Information about the project's license.