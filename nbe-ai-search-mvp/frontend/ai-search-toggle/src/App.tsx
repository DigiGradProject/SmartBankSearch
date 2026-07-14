import { useEffect, useMemo, useRef, useState } from "react";

type SearchMode = "traditional" | "ai";

type Citation = {
  title: string;
  url: string;
};

type SearchSuggestion = {
  query: string;
  label: string;
  url?: string | null;
  reason?: string | null;
  score?: number | null;
};

type SearchResponse = {
  answer: string | null;
  confidence: number;
  citations: Citation[];
  answered: boolean;
  language?: "ar" | "en";
  abstention_reason?: string | null;
  suggestions?: SearchSuggestion[];
  guidance?: string | null;
};

type AutocompleteResponse = {
  query: string;
  language: "ar" | "en";
  suggestions: SearchSuggestion[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const AUTOCOMPLETE_DEBOUNCE_MS = 250;

export default function App() {
  const [mode, setMode] = useState<SearchMode>("ai");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [autocompleteItems, setAutocompleteItems] = useState<SearchSuggestion[]>([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);
  const blurTimeoutRef = useRef<number | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const isArabic = useMemo(() => /[\u0600-\u06FF]/.test(query), [query]);

  useEffect(() => {
    if (mode !== "ai") {
      setAutocompleteItems([]);
      setShowAutocomplete(false);
      return;
    }

    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const params = new URLSearchParams({
          q: query,
          language: "auto",
          limit: "8",
        });
        const response = await fetch(`${API_BASE}/v1/autocomplete?${params.toString()}`, {
          signal: controller.signal,
        });
        if (!response.ok) {
          return;
        }
        const payload = (await response.json()) as AutocompleteResponse;
        setAutocompleteItems(payload.suggestions);
        setActiveSuggestion(-1);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
      }
    }, AUTOCOMPLETE_DEBOUNCE_MS);

    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [query, mode]);

  async function runSearch(searchQuery: string) {
    setLoading(true);
    setError(null);
    setResult(null);
    setShowAutocomplete(false);

    try {
      const response = await fetch(`${API_BASE}/v1/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: searchQuery, language: "auto" }),
      });
      if (!response.ok) {
        throw new Error(`Search failed (${response.status})`);
      }
      const payload = (await response.json()) as SearchResponse;
      setResult(payload);
      setQuery(searchQuery);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unexpected error");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;

    if (mode === "traditional") {
      setResult(null);
      setError("Traditional keyword search remains on the existing NBE website.");
      return;
    }

    await runSearch(query);
  }

  function selectSuggestion(item: SearchSuggestion) {
    setQuery(item.query);
    setShowAutocomplete(false);
    void runSearch(item.query);
  }

  function handleInputKeyDown(event: React.KeyboardEvent<HTMLInputElement>) {
    if (!showAutocomplete || autocompleteItems.length === 0) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();
      setActiveSuggestion((current) => (current + 1) % autocompleteItems.length);
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();
      setActiveSuggestion((current) =>
        current <= 0 ? autocompleteItems.length - 1 : current - 1
      );
      return;
    }

    if (event.key === "Enter" && activeSuggestion >= 0) {
      event.preventDefault();
      selectSuggestion(autocompleteItems[activeSuggestion]);
      return;
    }

    if (event.key === "Escape") {
      setShowAutocomplete(false);
      setActiveSuggestion(-1);
    }
  }

  function handleInputFocus() {
    if (blurTimeoutRef.current) {
      window.clearTimeout(blurTimeoutRef.current);
      blurTimeoutRef.current = null;
    }
    setShowAutocomplete(true);
  }

  function handleInputBlur() {
    blurTimeoutRef.current = window.setTimeout(() => {
      setShowAutocomplete(false);
      setActiveSuggestion(-1);
    }, 150);
  }

  return (
    <div className="page" dir={isArabic ? "rtl" : "ltr"}>
      <header className="hero">
        <p className="eyebrow">National Bank of Egypt</p>
        <h1>AI Search MVP</h1>
        <p className="subtitle">
          Bilingual semantic search with grounded answers and source citations.
        </p>
      </header>

      <section className="panel">
        <div className="mode-toggle" role="tablist" aria-label="Search mode">
          <button
            type="button"
            className={mode === "traditional" ? "active" : ""}
            onClick={() => setMode("traditional")}
          >
            Traditional Search
          </button>
          <button
            type="button"
            className={mode === "ai" ? "active" : ""}
            onClick={() => setMode("ai")}
          >
            AI Search Mode
          </button>
        </div>

        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrap">
            <input
              ref={inputRef}
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setShowAutocomplete(true);
              }}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
              onKeyDown={handleInputKeyDown}
              placeholder={isArabic ? "اكتب سؤالك هنا..." : "Ask a question about NBE services..."}
              autoComplete="off"
              aria-autocomplete="list"
              aria-expanded={showAutocomplete && autocompleteItems.length > 0}
              aria-controls="search-autocomplete"
            />
            {mode === "ai" && showAutocomplete && autocompleteItems.length > 0 && (
              <ul id="search-autocomplete" className="autocomplete-list" role="listbox">
                {autocompleteItems.map((item, index) => (
                  <li key={`${item.query}-${item.label}`} role="option" aria-selected={index === activeSuggestion}>
                    <button
                      type="button"
                      className={index === activeSuggestion ? "active" : ""}
                      onMouseDown={(event) => event.preventDefault()}
                      onClick={() => selectSuggestion(item)}
                    >
                      <span className="autocomplete-label">{item.label}</span>
                      {item.reason === "popular" && (
                        <span className="autocomplete-badge">
                          {isArabic ? "شائع" : "Popular"}
                        </span>
                      )}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button type="submit" disabled={loading || !query.trim()}>
            {loading ? "Searching..." : "Search"}
          </button>
        </form>

        {mode === "ai" && !query.trim() && autocompleteItems.length > 0 && (
          <div className="quick-suggestions">
            <h4>{isArabic ? "اقتراحات سريعة" : "Quick suggestions"}</h4>
            <div className="suggestion-list">
              {autocompleteItems.slice(0, 6).map((item) => (
                <button
                  key={`quick-${item.query}`}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => selectSuggestion(item)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        )}

        {error && <div className="alert error">{error}</div>}

        {result && (
          <article className="result">
            {result.answered ? (
              <>
                <div className="meta">
                  <span>Confidence: {(result.confidence * 100).toFixed(0)}%</span>
                  {result.language && <span>Language: {result.language.toUpperCase()}</span>}
                </div>
                <p className="answer">{result.answer}</p>
                {result.citations.length > 0 && (
                  <div className="citations">
                    <h3>Sources</h3>
                    <ul>
                      {result.citations.map((citation) => (
                        <li key={citation.url}>
                          <a href={citation.url} target="_blank" rel="noreferrer">
                            {citation.title}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </>
            ) : (
              <div className="no-answer">
                <h3>
                  {result.guidance && /عائد|فائدة|فائده|yield|لا تتوفر نسبة|غير متوفرة كرقم/i.test(result.guidance)
                    ? (isArabic ? "النسبة غير متوفرة في الفهرس" : "Rate not in indexed content")
                    : (isArabic ? "لم نجد إجابة مؤكدة" : "No confident answer found")}
                </h3>
                <p>{result.guidance || (isArabic
                  ? "جرّب أحد الاقتراحات التالية أو أعد صياغة سؤالك."
                  : "Try one of the suggestions below or rephrase your question.")}</p>
                {result.abstention_reason &&
                  !/^(insufficient_context|rate_not_in_index)$/.test(result.abstention_reason) && (
                  <p className="reason">Reason: {result.abstention_reason}</p>
                )}
                {result.suggestions && result.suggestions.length > 0 && (
                  <div className="suggestions">
                    <h4>{isArabic ? "صفحات ومواضيع مقترحة" : "Suggested pages and topics"}</h4>
                    <div className="suggestion-list">
                      {result.suggestions.map((item) => (
                        item.url ? (
                          <a
                            key={`${item.query}-${item.label}`}
                            className="suggestion-chip suggestion-link"
                            href={item.url}
                            target="_blank"
                            rel="noreferrer"
                          >
                            {item.label}
                          </a>
                        ) : (
                          <button
                            key={`${item.query}-${item.label}`}
                            type="button"
                            className="suggestion-chip"
                            onClick={() => runSearch(item.query)}
                          >
                            {item.label}
                          </button>
                        )
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </article>
        )}
      </section>
    </div>
  );
}
