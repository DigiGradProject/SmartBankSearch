import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from services.search_service.product_detail import try_product_detail_answer
from services.search_service.search import SearchService

q = "ما هي البنك الأهلى المصرى - شهادة استثمار ' أ '؟"
svc = SearchService()
r = svc.retrieve(q, "ar")
print("intent:", r.intent, "confidence:", r.confidence)
for i, c in enumerate(r.chunks[:3]):
    print(f"{i+1}. score={c.score:.3f} title={c.title}")
detail = try_product_detail_answer(q, "ar", r.chunks)
if detail:
    print("\n--- extractive answer ---")
    print(detail[0][:800])
