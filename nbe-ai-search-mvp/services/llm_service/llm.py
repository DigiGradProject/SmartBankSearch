import re

import httpx

from shared.config import settings
from shared.logging import get_logger
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an Enterprise AI Search Assistant for the National Bank of Egypt (NBE).

Your role is to answer questions ONLY from the retrieved documents.

You are NOT a chatbot.
You are NOT a financial advisor.
You are a Retrieval-Augmented Generation (RAG) search engine."""

RAG_USER_PROMPT = """========================================================
RETRIEVED DOCUMENTS
========================================================

{context}

========================================================
USER QUESTION
========================================================

Language: {language}

Question:
{query}

========================================================
INTERNAL REASONING (DO NOT OUTPUT)
========================================================

Internally perform these steps:

Step 1.
Determine the user's intent.

Possible intents:
- Exchange Rates
- Certificates
- Savings Accounts
- Current Accounts
- Personal Loans
- Credit Cards
- Corporate Banking
- SME Banking
- Digital Services
- Branches & ATMs
- News
- Offers
- Reports
- FAQ
- Other

Step 2.
Identify which retrieved document(s) are relevant.

Step 3.
Ignore unrelated documents completely.

Step 4.
Extract only explicit facts.

Step 5.
Never infer or guess missing information.

Step 6.
If the retrieved documents are insufficient,
return exactly:

NO_ANSWER

========================================================
OUTPUT FORMAT
========================================================

Always return your answer using this structure:

### Answer

<direct answer>

### Key Information

- ...
- ...
- ...

### Source

- <Document Title>

========================================================
RULES
========================================================

- Use ONLY the retrieved documents.
- Never use external knowledge.
- Never invent facts.
- Never hallucinate.
- Preserve numbers exactly.
- Preserve percentages exactly.
- Preserve currency values exactly.
- Preserve product names exactly.
- Keep dates exactly as written.
- Ignore conflicting low-relevance documents.
- If two documents conflict, prefer the highest ranked one.
- Respond in the same language as the user's question.
- Do NOT say:
    "According to the context"
    "Based on the provided information"
    "As an AI"
    "I cannot provide financial advice"

========================================================
SPECIAL CASES
========================================================

If the question asks for exchange rates:

Return

### Currency
...

### Buying Rate
...

### Selling Rate
...

### Last Updated
...

### Source
...

--------------------------------------------

If the question asks about a banking product:

Return

### Product
...

### Summary
...

### Features

- ...
- ...

### Eligibility

- ...

### Required Documents

- ...

### Fees / Interest

...

### Source

...

--------------------------------------------

If the question asks about branches:

Return

### Branch

...

### Address

...

### Working Hours

...

### Services

...

### Source

...

========================================================
FINAL ANSWER
========================================================"""


INFO_QUERY = re.compile(
    r"(انواع|ما هي|types of|what are|قائمة|تصنيف|اشرح|شرح)",
    re.IGNORECASE,
)

REFUSAL_PATTERNS = re.compile(
    r"("
    r"NO_ANSWER|"
    r"لا أجد|لا يوجد|غير موجود|"
    r"لا أستطيع|لا استطيع|لا يمكنني|لا يمكننى|"
    r"معلومات شخصية|نصيحة مالية|استشارة مالية|"
    r"does not contain|cannot find|can't find|not mentioned|no information|"
    r"cannot provide|can't provide|personal (or|and)? financial|"
    r"financial advice|I (cannot|can't|am unable to)"
    r")",
    re.IGNORECASE,
)

COMPLEX_QUERY = re.compile(
    r"("
    r"قارن|مقارنة|الفرق|لماذا|ازاي|إزاي|اشرح|شرح|"
    r"شروط|متطلبات|خطوات|كيف|"
    r"compare|difference|why|explain|how to|requirements|steps"
    r")",
    re.IGNORECASE,
)

THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)


class LLMService:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.tier1_model = settings.ollama_model
        self.tier2_model = settings.ollama_model_tier2

    async def health(self) -> str:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    return "ok"
        except Exception:
            pass
        return "degraded" if settings.llm_fallback_enabled else "down"

    def _select_model(self, query: str) -> str:
        if settings.llm_tier2_enabled and (COMPLEX_QUERY.search(query) or len(query.strip()) > 120):
            return self.tier2_model
        return self.tier1_model

    async def generate_answer(self, query: str, context: str, language: str) -> tuple[str | None, float]:
        primary = self._select_model(query)
        answer, confidence = await self._generate_with_model(primary, query, context, language)
        if answer:
            return answer, confidence

        # Escalate to tier-2 when tier-1 abstains/refuses and models differ.
        if (
            settings.llm_tier2_enabled
            and primary == self.tier1_model
            and self.tier2_model != self.tier1_model
        ):
            logger.info("llm_escalate_tier2", from_model=primary, to_model=self.tier2_model)
            answer, confidence = await self._generate_with_model(
                self.tier2_model, query, context, language
            )
            if answer:
                return answer, min(0.85, confidence + 0.05)

        if settings.llm_fallback_enabled:
            return self._fallback_answer(query, context, language), 0.55
        return None, 0.0

    def _build_prompt(self, query: str, context: str, language: str) -> str:
        return RAG_USER_PROMPT.format(context=context, language=language, query=query)

    async def _generate_with_model(
        self,
        model: str,
        query: str,
        context: str,
        language: str,
    ) -> tuple[str | None, float]:
        prompt = self._build_prompt(query, context, language)
        try:
            async with httpx.AsyncClient(timeout=settings.llm_timeout_seconds) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model,
                        "system": SYSTEM_PROMPT,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.1},
                    },
                )
                response.raise_for_status()
                payload = response.json()
                answer = self._clean_answer((payload.get("response") or "").strip())
                if not answer or self._is_refusal(answer):
                    return None, 0.0
                return answer, 0.78 if model == self.tier1_model else 0.82
        except Exception as exc:
            logger.warning("ollama_generate_failed", model=model, error=str(exc))
            return None, 0.0

    def _clean_answer(self, answer: str) -> str:
        cleaned = THINK_BLOCK.sub("", answer).strip()
        return cleaned

    def _is_refusal(self, answer: str) -> bool:
        return bool(REFUSAL_PATTERNS.search(answer))

    def _fallback_answer(self, query: str, context: str, language: str) -> str | None:
        from shared.document_quality import is_menu_heavy_text

        snippets = re.findall(r"\[\d+\]\s*(.+?)(?=\n\n\[|\Z)", context, re.DOTALL)
        if not snippets:
            return None

        query_terms = extract_query_terms(query, language)
        if not query_terms and language == "ar":
            query_terms = extract_query_terms(query, "ar")
        elif not query_terms:
            query_terms = [term.lower() for term in re.findall(r"\w+", query) if len(term) > 2]

        ranked = sorted(
            snippets,
            key=lambda snippet: keyword_overlap_score(snippet, query_terms, language),
            reverse=True,
        )
        for candidate in ranked[:3]:
            best = candidate.strip()
            if not best or is_menu_heavy_text(best):
                continue
            best_overlap = keyword_overlap_score(best, query_terms, language)
            min_overlap = 0.2 if INFO_QUERY.search(query) else 0.34
            if best_overlap < min_overlap:
                continue
            # Rate/yield questions need an actual number in evidence — never dump menus.
            if re.search(r"(عائد|فايده|فائدة|interest|yield|rate)", query, re.I):
                if not re.search(r"\d+\s*[٪%]", best):
                    continue
            prefix = (
                "بناءً على محتوى موقع البنك الأهلي المصري:"
                if language == "ar"
                else "Based on NBE website content:"
            )
            return f"{prefix} {best[:700]}"
        return None
