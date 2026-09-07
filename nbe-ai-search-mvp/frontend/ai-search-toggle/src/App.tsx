import { useEffect, useMemo, useRef, useState } from "react";

type SearchMode = "traditional" | "ai";

type Citation = {
  title: string;
  url: string;
  category?: string | null;
  relevance_score?: number | null;
  reranker_score?: number | null;
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
  confidence_reason?: string | null;
  query_hash?: string | null;
  cache_hit?: boolean;
  faithfulness?: string | null;
};

type AutocompleteResponse = {
  query: string;
  language: "ar" | "en";
  suggestions: SearchSuggestion[];
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const AUTOCOMPLETE_DEBOUNCE_MS = 250;

const DEFAULT_SUGGESTIONS_EN: SearchSuggestion[] = [
  { query: "How can I open a current account?", label: "Open a current account" },
  { query: "What are the requirements for a personal loan?", label: "Personal loan requirements" },
  { query: "How can I activate my debit card?", label: "Activate a debit card" },
  { query: "What are NBE mortgage finance options?", label: "Mortgage finance options" },
];

const DEFAULT_SUGGESTIONS_AR: SearchSuggestion[] = [
  { query: "ما هي شروط فتح حساب جاري؟", label: "شروط فتح حساب جاري" },
  { query: "ما هي شروط الحصول على قرض شخصي؟", label: "شروط القرض الشخصي" },
  { query: "كيف يمكنني تفعيل بطاقة الخصم المباشر؟", label: "تفعيل بطاقة الخصم المباشر" },
  { query: "ما هي خيارات التمويل العقاري؟", label: "خيارات التمويل العقاري" },
];

export default function App() {
  const [mode, setMode] = useState<SearchMode>("ai");
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<SearchResponse | null>(null);
  const [autocompleteItems, setAutocompleteItems] = useState<SearchSuggestion[]>([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [activeSuggestion, setActiveSuggestion] = useState(-1);
  const [lastSubmittedQuery, setLastSubmittedQuery] = useState("");
  const blurTimeoutRef = useRef<number | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const [feedbackSent, setFeedbackSent] = useState<"helpful" | "not_helpful" | null>(null);

  const isArabic = useMemo(() => /[\u0600-\u06FF]/.test(query), [query]);
  const quickSuggestions = useMemo(
    () => autocompleteItems.slice(0, 6).length > 0
      ? autocompleteItems.slice(0, 6)
      : (isArabic ? DEFAULT_SUGGESTIONS_AR : DEFAULT_SUGGESTIONS_EN),
    [autocompleteItems, isArabic],
  );

  async function sendFeedback(vote: "helpful" | "not_helpful") {
    if (!result?.query_hash || feedbackSent) return;
    try {
      await fetch(`${API_BASE}/v1/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query_hash: result.query_hash,
          vote,
          question: query,
          answer: result.answer,
          confidence: result.confidence,
          docs: result.citations.map((c) => ({ title: c.title, url: c.url })),
        }),
      });
      setFeedbackSent(vote);
    } catch {
      // non-blocking
    }
  }

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
    setFeedbackSent(null);
    setLastSubmittedQuery(searchQuery);
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
    } catch {
      setError(isArabic ? "حدث خطأ ما. يرجى إعادة المحاولة." : "Something went wrong. Please try your search again.");
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
    <main className="page" dir={isArabic ? "rtl" : "ltr"}>
      <section className="ai-search-card" aria-labelledby="ai-search-title">
        <div className="card-accent card-accent-top" aria-hidden="true" />
        <div className="card-accent card-accent-bottom" aria-hidden="true" />

        <header className="search-header">
          <div className="search-brand-mark" aria-hidden="true">
            <span>✦</span>
          </div>
          <div>
            <p className="eyebrow">National Bank of Egypt</p>
            <h1 id="ai-search-title">AI Search</h1>
            <p className="subtitle">
              {isArabic ? "اعثر على إجابات من المعلومات الرسمية للبنك الأهلي المصري" : "Find answers from NBE's official information"}
            </p>
          </div>
          <div className="trusted-badge">
            <span className="trusted-dot" aria-hidden="true" />
            <span>{isArabic ? "مصدر رسمي" : "Official source"}</span>
          </div>
        </header>

        <div className="mode-toggle" role="tablist" aria-label="Search mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "ai"}
            className={mode === "ai" ? "active" : ""}
            onClick={() => setMode("ai")}
          >
            <span className="mode-icon" aria-hidden="true">✦</span>
            <span>AI Search Mode</span>
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "traditional"}
            className={mode === "traditional" ? "active" : ""}
            onClick={() => setMode("traditional")}
          >
            <span className="mode-icon" aria-hidden="true">⌕</span>
            <span>Traditional Search</span>
          </button>
        </div>

        <form onSubmit={handleSearch} className="search-form">
          <div className="search-input-wrap">
            <span className="input-icon" aria-hidden="true">✧</span>
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
              placeholder={isArabic
                ? "اسأل عن منتجات وخدمات وحسابات وبطاقات وقروض البنك الأهلي المصري..."
                : "Ask about NBE products, services, accounts, cards, loans..."}
              autoComplete="off"
              aria-label={isArabic ? "اسأل عن البنك الأهلي المصري" : "Ask about NBE"}
              aria-autocomplete="list"
              aria-expanded={showAutocomplete && autocompleteItems.length > 0}
              aria-controls="search-autocomplete"
            />
            <button className="search-submit" type="submit" disabled={loading || !query.trim()}>
              <span aria-hidden="true">⌕</span>
              <span>{loading ? (isArabic ? "جارٍ البحث" : "Searching") : (isArabic ? "بحث" : "Search")}</span>
            </button>
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
                      <span className="autocomplete-leading" aria-hidden="true">⌕</span>
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
        </form>

        {mode === "ai" && !query.trim() && (
          <section className="suggestions" aria-labelledby="popular-searches-title">
            <div className="section-heading">
              <span className="section-icon" aria-hidden="true">◷</span>
              <h2 id="popular-searches-title">{isArabic ? "عمليات البحث الشائعة" : "Popular searches"}</h2>
            </div>
            <div className="suggestion-list">
              {quickSuggestions.map((item) => (
                <button
                  key={`quick-${item.query}`}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => selectSuggestion(item)}
                >
                  <span>{item.label}</span>
                  <span className="chip-arrow" aria-hidden="true">↗</span>
                </button>
              ))}
            </div>
          </section>
        )}

        {loading && (
          <div className="loading-state" role="status" aria-live="polite">
            <span className="loading-orb" aria-hidden="true">✦</span>
            <div>
              <strong>{isArabic ? "جارٍ البحث في معلومات البنك الأهلي المصري" : "AI is searching NBE information"}</strong>
              <span className="loading-dots" aria-hidden="true"><i /> <i /> <i /></span>
            </div>
          </div>
        )}

        {error && !loading && (
          <div className="error-state" role="alert">
            <div className="state-icon" aria-hidden="true">!</div>
            <div>
              <strong>{isArabic ? "حدث خطأ ما" : "Something went wrong"}</strong>
              <p>{error}</p>
            </div>
            <button type="button" className="retry-button" onClick={() => void runSearch(lastSubmittedQuery || query)}>
              {isArabic ? "إعادة المحاولة" : "Try again"}
            </button>
          </div>
        )}

        {result && !loading && (
          <article className="result" aria-live="polite">
            {result.answered ? (
              <>
                <div className="result-heading">
                  <div>
                    <span className="result-kicker">{isArabic ? "إجابة الذكاء الاصطناعي" : "AI Answer"}</span>
                    <h2>{isArabic ? "إليك ما وجدناه" : "Here is what we found"}</h2>
                  </div>
                  <div className="result-meta" aria-label="Answer details">
                    <span className="confidence-pill">{Math.round(result.confidence * 100)}% {isArabic ? "ثقة" : "confidence"}</span>
                    {result.language && <span>{result.language.toUpperCase()}</span>}
                    {result.cache_hit && <span>{isArabic ? "من الذاكرة" : "Cached"}</span>}
                  </div>
                </div>
                {result.confidence_reason && (
                  <p className="confidence-reason">{result.confidence_reason}</p>
                )}
                <div className="answer" dir={result.language === "ar" || isArabic ? "rtl" : "ltr"}>
                  {result.answer}
                </div>
                {result.citations.length > 0 && (
                  <section className="citations" aria-labelledby="sources-title">
                    <div className="section-heading">
                      <span className="section-icon" aria-hidden="true">✓</span>
                      <h3 id="sources-title">{isArabic ? "المصادر" : "Sources"}</h3>
                    </div>
                    <div className="citation-list">
                      {result.citations.map((citation) => (
                        <a
                          className="citation-card"
                          key={citation.url}
                          href={citation.url}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <span className="citation-icon" aria-hidden="true">↗</span>
                          <span className="citation-copy">
                            <strong>{citation.title}</strong>
                            <span>{isArabic ? "الموقع الرسمي للبنك الأهلي المصري" : "Official NBE website"}</span>
                          </span>
                          {(citation.relevance_score != null || citation.category) && (
                            <span className="citation-meta">
                              {citation.category ? `${citation.category} · ` : ""}
                              {citation.relevance_score != null
                                ? `${Math.round(citation.relevance_score * 100)}%`
                                : ""}
                            </span>
                          )}
                        </a>
                      ))}
                    </div>
                  </section>
                )}
                <div className="feedback-row">
                  <span>{isArabic ? "هل كانت الإجابة مفيدة؟" : "Was this answer helpful?"}</span>
                  <button
                    type="button"
                    className={feedbackSent === "helpful" ? "active" : ""}
                    disabled={!!feedbackSent}
                    onClick={() => sendFeedback("helpful")}
                    aria-label="Helpful"
                  >
                    <span aria-hidden="true">👍</span>
                    <span>{isArabic ? "مفيدة" : "Helpful"}</span>
                  </button>
                  <button
                    type="button"
                    className={feedbackSent === "not_helpful" ? "active" : ""}
                    disabled={!!feedbackSent}
                    onClick={() => sendFeedback("not_helpful")}
                    aria-label="Not helpful"
                  >
                    <span aria-hidden="true">👎</span>
                    <span>{isArabic ? "غير مفيدة" : "Not helpful"}</span>
                  </button>
                  {feedbackSent && (
                    <span className="feedback-thanks">
                      {isArabic ? "شكراً لملاحظتك" : "Thanks for your feedback"}
                    </span>
                  )}
                </div>
              </>
            ) : (
              <div className="no-answer">
                <div className="state-icon" aria-hidden="true">?</div>
                <div>
                  <h2>
                    {result.guidance && /عائد|فائدة|فائده|yield|لا تتوفر نسبة|غير متوفرة كرقم/i.test(result.guidance)
                      ? (isArabic ? "النسبة غير متوفرة في الفهرس" : "Rate not in indexed content")
                      : (isArabic ? "لم نجد إجابة مؤكدة" : "No confident answer found")}
                  </h2>
                  <p>{result.guidance || (isArabic
                    ? "جرّب أحد الاقتراحات التالية أو أعد صياغة سؤالك."
                    : "Try one of the suggestions below or rephrase your question.")}</p>
                  {result.abstention_reason &&
                    !/^(insufficient_context|rate_not_in_index)$/.test(result.abstention_reason) && (
                    <p className="reason">Reason: {result.abstention_reason}</p>
                  )}
                  {result.suggestions && result.suggestions.length > 0 && (
                    <div className="suggestions">
                      <div className="section-heading">
                        <span className="section-icon" aria-hidden="true">⌕</span>
                        <h3>{isArabic ? "صفحات ومواضيع مقترحة" : "Suggested pages and topics"}</h3>
                      </div>
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
              </div>
            )}
          </article>
        )}
      </section>
    </main>
  );
}
