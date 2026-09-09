import { useEffect, useMemo, useRef, useState } from "react";
import type { FC, ReactNode } from "react";

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

type PopularItem = { label: string; query: string };

const POPULAR_SEARCHES: Record<"en" | "ar", PopularItem[]> = {
  en: [
    { label: "Open a current account", query: "How can I open a current account?" },
    { label: "Transfer money", query: "How can I transfer money?" },
    { label: "Debit card", query: "How can I activate my debit card?" },
    { label: "Loan requirements", query: "What are the requirements for a loan?" },
    { label: "Mortgage finance", query: "What are the mortgage finance options?" },
  ],
  ar: [
    { label: "فتح حساب جاري", query: "كيف يمكنني فتح حساب جاري؟" },
    { label: "تحويل الأموال", query: "كيف يمكنني تحويل الأموال؟" },
    { label: "بطاقة الخصم المباشر", query: "كيف يمكنني تفعيل بطاقة الخصم المباشر؟" },
    { label: "شروط الحصول على قرض", query: "ما هي شروط الحصول على قرض؟" },
    { label: "التمويل العقاري", query: "ما هي خيارات التمويل العقاري؟" },
  ],
};

/* ------------------------------------------------------------------ */
/* Inline SVG icons (no extra dependencies)                            */
/* ------------------------------------------------------------------ */

type IconProps = { size?: number };

function Icon({
  size = 18,
  filled = false,
  children,
}: IconProps & { filled?: boolean; children: ReactNode }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke={filled ? "none" : "currentColor"}
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
    >
      {children}
    </svg>
  );
}

const IconSearch: FC<IconProps> = (p) => (
  <Icon {...p}>
    <circle cx="11" cy="11" r="7.5" />
    <path d="m20.5 20.5-4.55-4.55" />
  </Icon>
);

const IconSparkle: FC<IconProps> = (p) => (
  <Icon {...p} filled>
    <path d="M12 3l1.9 5.8a2 2 0 0 0 1.3 1.3L21 12l-5.8 1.9a2 2 0 0 0-1.3 1.3L12 21l-1.9-5.8a2 2 0 0 0-1.3-1.3L3 12l5.8-1.9a2 2 0 0 0 1.3-1.3L12 3z" />
    <path d="M19 15.4l.75 2 2 .75-2 .75-.75 2-.75-2-2-.75 2-.75.75-2z" opacity=".6" />
  </Icon>
);

const IconClock: FC<IconProps> = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9.5" />
    <path d="M12 6.5V12l3.5 2" />
  </Icon>
);

const IconShield: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M12 22s8-3.6 8-10V5.2L12 2 4 5.2V12c0 6.4 8 10 8 10z" />
    <path d="m8.8 11.6 2.2 2.2 4.2-4.4" />
  </Icon>
);

const IconUser: FC<IconProps> = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="7.5" r="3.8" />
    <path d="M4.5 21a7.5 7.5 0 0 1 15 0" />
  </Icon>
);

const IconCard: FC<IconProps> = (p) => (
  <Icon {...p}>
    <rect x="2.5" y="5" width="19" height="14" rx="2.5" />
    <path d="M2.5 10h19" />
    <path d="M6.5 15h4" />
  </Icon>
);

const IconPercent: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M19 5 5 19" />
    <circle cx="6.8" cy="6.8" r="2.3" />
    <circle cx="17.2" cy="17.2" r="2.3" />
  </Icon>
);

const IconPhone: FC<IconProps> = (p) => (
  <Icon {...p}>
    <rect x="6" y="2.5" width="12" height="19" rx="2.5" />
    <path d="M12 18.2h.01" />
  </Icon>
);

const IconBuilding: FC<IconProps> = (p) => (
  <Icon {...p}>
    <rect x="4.5" y="3" width="15" height="18" rx="1.5" />
    <path d="M9.5 21v-4.5h5V21" />
    <path d="M8.5 7.5h2M13.5 7.5h2M8.5 11h2M13.5 11h2" />
  </Icon>
);

const IconThumbUp: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3z" />
    <path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3" />
  </Icon>
);

const IconThumbDown: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3z" />
    <path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17" />
  </Icon>
);

const IconArrowUpRight: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M7 17 17 7" />
    <path d="M8.5 7H17v8.5" />
  </Icon>
);

const IconInfo: FC<IconProps> = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9.5" />
    <path d="M12 11v5" />
    <path d="M12 7.5h.01" />
  </Icon>
);

const IconAlert: FC<IconProps> = (p) => (
  <Icon {...p}>
    <circle cx="12" cy="12" r="9.5" />
    <path d="M12 7.5V13" />
    <path d="M12 16.5h.01" />
  </Icon>
);

const IconRetry: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M2.5 5.5V10H7" />
    <path d="M4.6 14.5a8 8 0 1 0 .6-6.9L2.5 10" />
  </Icon>
);

const IconMessage: FC<IconProps> = (p) => (
  <Icon {...p}>
    <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
  </Icon>
);

/* ------------------------------------------------------------------ */
/* Quick access (visual suggestions — prefill the search input only)   */
/* ------------------------------------------------------------------ */

type QuickItem = {
  key: string;
  en: string;
  ar: string;
  queryEn: string;
  queryAr: string;
  Icon: FC<IconProps>;
};

const QUICK_ACCESS: QuickItem[] = [
  { key: "accounts", en: "Accounts", ar: "الحسابات", queryEn: "current accounts", queryAr: "فتح حساب جاري", Icon: IconUser },
  { key: "cards", en: "Cards", ar: "البطاقات", queryEn: "debit cards", queryAr: "بطاقة الخصم المباشر", Icon: IconCard },
  { key: "loans", en: "Loans", ar: "القروض", queryEn: "loans", queryAr: "شروط الحصول على قرض", Icon: IconPercent },
  { key: "digital", en: "Digital Banking", ar: "الخدمات الرقمية", queryEn: "digital banking", queryAr: "الخدمات المصرفية الرقمية", Icon: IconPhone },
  { key: "branches", en: "Branches", ar: "الفروع", queryEn: "branches", queryAr: "الفروع", Icon: IconBuilding },
];

const NBE_WEBSITE = "https://www.nbe.com.eg";

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

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

  // UI language follows the typed query automatically (existing behaviour),
  // with an explicit override via the English / عربي toggle.
  const [langOverride, setLangOverride] = useState<"en" | "ar" | null>(null);
  const isArabic = useMemo(() => /[\u0600-\u06FF]/.test(query), [query]);
  const uiLang: "en" | "ar" = langOverride ?? (isArabic ? "ar" : "en");
  const ar = uiLang === "ar";

  useEffect(() => {
    document.documentElement.lang = uiLang;
  }, [uiLang]);

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
      setError(isArabic ? "يرجى إعادة المحاولة." : "Please try your search again.");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch(event: React.FormEvent) {
    event.preventDefault();
    if (!query.trim()) return;

    if (mode === "traditional") {
      setResult(null);
      setError(null);
      return;
    }

    await runSearch(query);
  }

  function selectSuggestion(item: SearchSuggestion) {
    setQuery(item.query);
    setShowAutocomplete(false);
    void runSearch(item.query);
  }

  function handleQuickAccess(item: QuickItem) {
    setQuery(ar ? item.queryAr : item.queryEn);
    setActiveSuggestion(-1);
    inputRef.current?.focus();
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
    <>
      <div className="backdrop" aria-hidden="true" />
      <main className="page" dir={ar ? "rtl" : "ltr"}>
        <div className="layout">
          {/* ---------------- LEFT · AI assistant ---------------- */}
          <aside className="assistant-panel" aria-label={ar ? "مساعدك الذكي" : "AI assistant"}>
            <div className="assistant-head">
              <span className="assistant-badge" aria-hidden="true">
                <IconMessage size={20} />
              </span>
              <h2>{ar ? "مساعدك الذكي من البنك الأهلي" : "Your NBE AI Assistant"}</h2>
              <p>
                {ar
                  ? "اسأل عن الحسابات والبطاقات والقروض والخدمات المصرفية والمزيد."
                  : "Ask questions about accounts, cards, loans, services and more."}
              </p>
            </div>
            <ul className="assistant-features">
              <li>
                <span className="feature-icon" aria-hidden="true"><IconSparkle size={17} /></span>
                <span className="feature-copy">
                  <strong>{ar ? "مدعوم بالذكاء الاصطناعي" : "Powered by AI"}</strong>
                  <span>{ar ? "إجابات سريعة ودقيقة" : "Get fast and accurate answers"}</span>
                </span>
              </li>
              <li>
                <span className="feature-icon" aria-hidden="true"><IconShield size={17} /></span>
                <span className="feature-copy">
                  <strong>{ar ? "معلومات رسمية من البنك الأهلي" : "Official NBE Information"}</strong>
                  <span>{ar ? "إجابات مستندة إلى مصادر موثوقة" : "Answers are grounded in trusted sources"}</span>
                </span>
              </li>
              <li>
                <span className="feature-icon" aria-hidden="true"><IconClock size={17} /></span>
                <span className="feature-copy">
                  <strong>{ar ? "متاح على مدار الساعة" : "Available 24/7"}</strong>
                  <span>{ar ? "في أي وقت ومن أي مكان" : "Anytime, anywhere"}</span>
                </span>
              </li>
            </ul>
          </aside>

          {/* ---------------- CENTER · main AI search ---------------- */}
          <section className="center-col" aria-labelledby="ai-search-title">
            <div className="search-card">
              <span className="card-corner card-corner-tl" aria-hidden="true" />
              <span className="card-corner card-corner-br" aria-hidden="true" />

              <header className="card-header">
                <img
                  className="brand-logo"
                  src="/nbe-logo.svg"
                  alt={ar ? "البنك الأهلي المصري" : "National Bank of Egypt"}
                />
                <div className="header-side">
                  <div className="header-title">
                    <span className="header-spark" aria-hidden="true"><IconSparkle size={24} /></span>
                    <div>
                      <h1 id="ai-search-title">{ar ? "البحث الذكي" : "AI Search"}</h1>
                      <p>
                        {ar
                          ? "اعثر على إجابات من المعلومات الرسمية للبنك الأهلي المصري"
                          : "Find answers from NBE's official information"}
                      </p>
                    </div>
                  </div>
                  <div className="header-controls">
                    <div className="lang-toggle" role="group" aria-label={ar ? "اللغة" : "Language"}>
                      <button
                        type="button"
                        className={uiLang === "en" ? "active" : ""}
                        aria-pressed={uiLang === "en"}
                        onClick={() => setLangOverride("en")}
                      >
                        English
                      </button>
                      <button
                        type="button"
                        className={uiLang === "ar" ? "active" : ""}
                        aria-pressed={uiLang === "ar"}
                        onClick={() => setLangOverride("ar")}
                      >
                        عربي
                      </button>
                    </div>
                    <span className="trusted-badge">
                      <span className="trusted-dot" aria-hidden="true" />
                      <span>{ar ? "مصدر رسمي" : "Official source"}</span>
                    </span>
                  </div>
                </div>
              </header>

              <div className="mode-toggle" role="tablist" aria-label={ar ? "وضع البحث" : "Search mode"}>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "ai"}
                  className={mode === "ai" ? "active" : ""}
                  onClick={() => setMode("ai")}
                >
                  <span className="mode-icon" aria-hidden="true"><IconSparkle size={16} /></span>
                  <span>{ar ? "البحث الذكي" : "AI Search Mode"}</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={mode === "traditional"}
                  className={mode === "traditional" ? "active" : ""}
                  onClick={() => setMode("traditional")}
                >
                  <span className="mode-icon" aria-hidden="true"><IconSearch size={16} /></span>
                  <span>{ar ? "البحث التقليدي" : "Traditional Search"}</span>
                </button>
              </div>

              <form onSubmit={handleSearch} className="search-form">
                <div className="search-box">
                  <span className="input-icon" aria-hidden="true">
                    {mode === "ai" ? <IconSparkle size={22} /> : <IconSearch size={22} />}
                  </span>
                  <input
                    ref={inputRef}
                    value={query}
                    dir={isArabic || ar ? "rtl" : "ltr"}
                    onChange={(event) => {
                      setQuery(event.target.value);
                      setShowAutocomplete(true);
                    }}
                    onFocus={handleInputFocus}
                    onBlur={handleInputBlur}
                    onKeyDown={handleInputKeyDown}
                    placeholder={ar || isArabic
                      ? "اسأل عن منتجات وخدمات وحسابات وبطاقات وقروض البنك الأهلي المصري..."
                      : "Ask about NBE products, services, accounts, cards, loans..."}
                    autoComplete="off"
                    aria-label={ar || isArabic ? "اسأل عن البنك الأهلي المصري" : "Ask about NBE"}
                    aria-autocomplete="list"
                    aria-expanded={showAutocomplete && autocompleteItems.length > 0}
                    aria-controls="search-autocomplete"
                  />
                  <button className="search-submit" type="submit" disabled={loading || !query.trim()}>
                    <IconSearch size={18} />
                    <span>{loading ? (ar ? "جارٍ البحث" : "Searching") : (ar ? "بحث" : "Search")}</span>
                  </button>
                  {mode === "ai" && showAutocomplete && autocompleteItems.length > 0 && (
                    <ul id="search-autocomplete" className="autocomplete-list" role="listbox">
                      {autocompleteItems.map((item, index) => (
                        <li key={`${item.query}-${item.label}`} role="option" aria-selected={index === activeSuggestion}>
                          <button
                            type="button"
                            dir="auto"
                            className={index === activeSuggestion ? "active" : ""}
                            onMouseDown={(event) => event.preventDefault()}
                            onClick={() => selectSuggestion(item)}
                          >
                            <span className="autocomplete-leading" aria-hidden="true"><IconSearch size={15} /></span>
                            <span className="autocomplete-label">{item.label}</span>
                            {item.reason === "popular" && (
                              <span className="autocomplete-badge">
                                {ar ? "شائع" : "Popular"}
                              </span>
                            )}
                          </button>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              </form>

              {mode === "ai" && (
                <div className="popular-row">
                  <span className="popular-label">
                    <IconClock size={16} />
                    <span>{ar ? "عمليات البحث الشائعة:" : "Popular searches:"}</span>
                  </span>
                  {POPULAR_SEARCHES[uiLang].map((item) => (
                    <button
                      key={item.label}
                      type="button"
                      dir="auto"
                      className="chip"
                      onClick={() => selectSuggestion(item)}
                    >
                      <span>{item.label}</span>
                      <span className="chip-arrow" aria-hidden="true"><IconArrowUpRight size={13} /></span>
                    </button>
                  ))}
                </div>
              )}

              {mode === "ai" && (
                <div className="quick-row" role="group" aria-label={ar ? "وصول سريع" : "Quick access"}>
                  <span className="quick-label">{ar ? "وصول سريع" : "Quick access"}</span>
                  {QUICK_ACCESS.map((item) => (
                    <button
                      key={item.key}
                      type="button"
                      className="quick-item"
                      onClick={() => handleQuickAccess(item)}
                    >
                      <span className="quick-icon" aria-hidden="true"><item.Icon size={16} /></span>
                      <span>{ar ? item.ar : item.en}</span>
                    </button>
                  ))}
                </div>
              )}

              {loading && (
                <div className="loading-state" role="status" aria-live="polite">
                  <span className="loading-orb" aria-hidden="true"><IconSparkle size={20} /></span>
                  <strong>
                    {ar ? "جارٍ البحث في معلومات البنك الأهلي المصري..." : "Searching NBE information..."}
                  </strong>
                  <span className="loading-dots" aria-hidden="true"><i /><i /><i /></span>
                  <span className="loading-shimmer" aria-hidden="true" />
                </div>
              )}

              {error && !loading && (
                <div className="error-state" role="alert">
                  <span className="state-icon" aria-hidden="true"><IconAlert size={18} /></span>
                  <div className="state-copy">
                    <strong>{ar ? "حدث خطأ ما" : "Something went wrong"}</strong>
                    <p>{error}</p>
                  </div>
                  <button
                    type="button"
                    className="retry-button"
                    onClick={() => void runSearch(lastSubmittedQuery || query)}
                  >
                    <IconRetry size={15} />
                    <span>{ar ? "إعادة المحاولة" : "Try again"}</span>
                  </button>
                </div>
              )}

              {mode === "traditional" && !loading && !result && !error && (
                <div className="notice-state" role="status">
                  <span className="state-icon" aria-hidden="true"><IconInfo size={18} /></span>
                  <div className="state-copy">
                    <strong>{ar ? "البحث التقليدي" : "Traditional Search"}</strong>
                    <p>
                      {ar
                        ? "البحث بالكلمات المفتاحية متاح على موقع البنك الأهلي المصري الرسمي."
                        : "Traditional keyword search remains on the existing NBE website."}
                    </p>
                  </div>
                </div>
              )}

              {result && !loading && (
                <article className="result" aria-live="polite">
                  {result.answered ? (
                    <>
                      <div className="result-heading">
                        <div>
                          <span className="result-kicker">{ar ? "إجابة الذكاء الاصطناعي" : "AI Answer"}</span>
                          <h2>{ar ? "إليك ما وجدناه" : "Here is what we found"}</h2>
                        </div>
                        <div className="result-meta" aria-label="Answer details">
                          <span className="confidence-pill">
                            {Math.round(result.confidence * 100)}% {ar ? "ثقة" : "confidence"}
                          </span>
                          {result.language && <span>{result.language.toUpperCase()}</span>}
                          {result.cache_hit && <span>{ar ? "من الذاكرة" : "Cached"}</span>}
                        </div>
                      </div>
                      {result.confidence_reason && (
                        <p className="confidence-reason">{result.confidence_reason}</p>
                      )}
                      <div
                        className="answer"
                        dir={result.language === "ar" || isArabic ? "rtl" : "ltr"}
                      >
                        {result.answer}
                      </div>
                      {result.citations.length > 0 && (
                        <section className="citations" aria-labelledby="sources-title">
                          <div className="section-heading">
                            <span className="section-icon" aria-hidden="true"><IconShield size={13} /></span>
                            <h3 id="sources-title">{ar ? "المصادر" : "Sources"}</h3>
                          </div>
                          <div className="citation-list">
                            {result.citations.map((citation, index) => (
                              <a
                                className="citation-card"
                                key={citation.url}
                                href={citation.url}
                                target="_blank"
                                rel="noreferrer"
                              >
                                <span className="citation-index" aria-hidden="true">{index + 1}</span>
                                <span className="citation-copy" dir="auto">
                                  <strong>{citation.title}</strong>
                                  <span>{ar ? "الموقع الرسمي للبنك الأهلي المصري" : "Official NBE website"}</span>
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
                        <span>{ar ? "هل كانت الإجابة مفيدة؟" : "Was this answer helpful?"}</span>
                        <button
                          type="button"
                          className={feedbackSent === "helpful" ? "active" : ""}
                          disabled={!!feedbackSent}
                          onClick={() => sendFeedback("helpful")}
                          aria-label="Helpful"
                        >
                          <IconThumbUp size={15} />
                          <span>{ar ? "مفيدة" : "Helpful"}</span>
                        </button>
                        <button
                          type="button"
                          className={feedbackSent === "not_helpful" ? "active" : ""}
                          disabled={!!feedbackSent}
                          onClick={() => sendFeedback("not_helpful")}
                          aria-label="Not helpful"
                        >
                          <IconThumbDown size={15} />
                          <span>{ar ? "غير مفيدة" : "Not helpful"}</span>
                        </button>
                        {feedbackSent && (
                          <span className="feedback-thanks">
                            {ar ? "شكراً لملاحظتك" : "Thanks for your feedback"}
                          </span>
                        )}
                      </div>
                    </>
                  ) : (
                    <div className="no-answer">
                      <div className="state-icon" aria-hidden="true"><IconAlert size={18} /></div>
                      <div className="state-copy">
                        <h2>
                          {result.guidance && /عائد|فائدة|فائده|yield|لا تتوفر نسبة|غير متوفرة كرقم/i.test(result.guidance)
                            ? (ar ? "النسبة غير متوفرة في الفهرس" : "Rate not in indexed content")
                            : (ar ? "لم نجد إجابة مؤكدة" : "No confident answer found")}
                        </h2>
                        <p>{result.guidance || (ar
                          ? "جرّب أحد الاقتراحات التالية أو أعد صياغة سؤالك."
                          : "Try one of the suggestions below or rephrase your question.")}</p>
                        {result.abstention_reason &&
                          !/^(insufficient_context|rate_not_in_index)$/.test(result.abstention_reason) && (
                          <p className="reason">{ar ? "السبب: " : "Reason: "}{result.abstention_reason}</p>
                        )}
                        {result.suggestions && result.suggestions.length > 0 && (
                          <div className="suggestions">
                            <div className="section-heading">
                              <span className="section-icon" aria-hidden="true"><IconSearch size={13} /></span>
                              <h3>{ar ? "صفحات ومواضيع مقترحة" : "Suggested pages and topics"}</h3>
                            </div>
                            <div className="suggestion-list">
                              {result.suggestions.map((item) => (
                                item.url ? (
                                  <a
                                    key={`${item.query}-${item.label}`}
                                    className="suggestion-chip suggestion-link"
                                    dir="auto"
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
                                    dir="auto"
                                    className="suggestion-chip"
                                    onClick={() => void runSearch(item.query)}
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

              {mode === "ai" && !query.trim() && !result && !loading && !error && (
                <section className="empty-state" aria-label={ar ? "أسئلة مقترحة" : "Example questions"}>
                  <h2>{ar ? "جرّب أن تسأل" : "Try asking"}</h2>
                  <p>{ar ? "اختر سؤالاً للبدء أو اكتب سؤالك بنفسك." : "Pick a question to get started, or type your own."}</p>
                  <div className="empty-list">
                    {(ar ? DEFAULT_SUGGESTIONS_AR : DEFAULT_SUGGESTIONS_EN).slice(0, 3).map((item) => (
                      <button
                        key={item.label}
                        type="button"
                        dir="auto"
                        className="empty-question"
                        onClick={() => selectSuggestion(item)}
                      >
                        <span className="empty-q-icon" aria-hidden="true"><IconSearch size={15} /></span>
                        <span className="empty-q-text">{item.label}</span>
                        <span className="empty-q-arrow" aria-hidden="true"><IconArrowUpRight size={13} /></span>
                      </button>
                    ))}
                  </div>
                </section>
              )}
            </div>
          </section>

          {/* ---------------- RIGHT · trust cards ---------------- */}
          <aside className="trust-panel" aria-label={ar ? "معلومات مفيدة" : "Helpful information"}>
            <article className="trust-card">
              <span className="trust-icon" aria-hidden="true"><IconInfo size={18} /></span>
              <h3>{ar ? "هل تعلم؟" : "Did you know?"}</h3>
              <p>
                {ar
                  ? "يمكنك البحث في منتجات وخدمات البنك الأهلي المصري، بما في ذلك الحسابات والبطاقات والقروض والمزيد."
                  : "You can search across NBE products and services, including accounts, cards, loans and more."}
              </p>
              <a className="trust-link" href={NBE_WEBSITE} target="_blank" rel="noreferrer">
                <span>{ar ? "استكشف خدمات البنك الأهلي" : "Explore NBE Services"}</span>
                <IconArrowUpRight size={14} />
              </a>
            </article>
            <article className="trust-card">
              <span className="trust-icon trust-icon-gold" aria-hidden="true"><IconShield size={18} /></span>
              <h3>{ar ? "معلومات موثوقة" : "Trusted Information"}</h3>
              <p>
                {ar
                  ? "تعتمد الإجابات على المحتوى الرسمي لموقع البنك الأهلي المصري."
                  : "Answers are based on official NBE website content."}
              </p>
              <a className="trust-link" href={NBE_WEBSITE} target="_blank" rel="noreferrer">
                <span>{ar ? "اعرف المزيد" : "Learn more"}</span>
                <IconArrowUpRight size={14} />
              </a>
            </article>
          </aside>
        </div>
      </main>
    </>
  );
}
