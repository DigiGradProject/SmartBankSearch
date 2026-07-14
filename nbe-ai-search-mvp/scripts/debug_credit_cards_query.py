import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from services.search_service.card_catalog import build_card_types_answer, build_credit_cards_answer, is_card_types_query, is_credit_cards_overview_query
from services.search_service.intent_classifier import classify_query
from services.search_service.search import SearchService
from services.context_builder.builder import ContextBuilder

q = "بطاقات الائتمان"
svc = SearchService()
intent = classify_query(q, "ar")
print("intent:", intent.intent, intent.confidence)
print("is_card_types:", is_card_types_query(q, "ar"))
print("is_credit_overview:", is_credit_cards_overview_query(q, "ar"))
r = svc.retrieve(q, "ar")
print("should_answer:", r.should_answer, "confidence:", r.confidence)
for i, c in enumerate(r.chunks[:6]):
    print(f"{i+1}. {c.title[:70]} | score={c.score:.3f} stub={getattr(c,'is_stub',False)}")
    print(f"   url tail: ...{c.url[-55:]}")
built = ContextBuilder().build(q, r.chunks)
print("\ncontext preview:", built.context_text[:500])
print("\n--- credit answer ---")
ans = build_credit_cards_answer(r.chunks, "ar")
print(ans[:1200] if ans else None)
