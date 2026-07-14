import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from services.search_service.card_catalog import build_card_types_answer, is_card_types_query
from services.search_service.certificate_catalog import build_certificate_types_answer, is_certificate_types_query
from services.search_service.intent_classifier import classify_query
from services.search_service.search import SearchService

q = "انواع البطاقات البنكيه"
svc = SearchService()
intent = classify_query(q, "ar")
print("intent:", intent.intent, intent.confidence)
print("is_card_types:", is_card_types_query(q, "ar"))
r = svc.retrieve(q, "ar")
print("top chunks:")
for i, c in enumerate(r.chunks[:4]):
    print(f"{i+1}. {c.title[:70]} | {c.score:.3f}")
print("\n--- catalog answer ---")
print(build_card_types_answer(r.chunks, "ar"))
