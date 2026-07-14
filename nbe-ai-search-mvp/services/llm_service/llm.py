import re

import httpx

from shared.config import settings
from shared.logging import get_logger
from services.search_service.keyword_rank import extract_query_terms, keyword_overlap_score

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are NBE AI Search assistant for the National Bank of Egypt public website.
Your job is to summarize PUBLIC product and service information from the provided context only.
This is NOT personal financial advice and NOT account-specific guidance.
Do NOT refuse questions about certificates, accounts, cards, loans, rates, or bank services
when the context contains relevant public website content — answer from that context.
Answer ONLY using the provided context snippets.
If the context does not contain enough information, reply with exactly: NO_ANSWER
Never use outside knowledge.
Never invent rates, terms, or eligibility rules that are not in the context.
Respond in the same language as the user question (Arabic or English).
Keep answers concise and factual.
When citing facts, reference snippet numbers like [1], [2].
Do not include chain-of-thought or <think> sections in the final answer."""


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

    async def _generate_with_model(
        self,
        model: str,
        query: str,
        context: str,
        language: str,
    ) -> tuple[str | None, float]:
        prompt = (
            f"Context from NBE public website pages:\n{context}\n\n"
            f"Question ({language}): {query}\n\n"
            "Instructions: Summarize only the public product/service facts found in the context. "
            "Do not refuse as personal/financial advice. If context is insufficient, reply NO_ANSWER.\n"
            "Answer:"
        )
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
