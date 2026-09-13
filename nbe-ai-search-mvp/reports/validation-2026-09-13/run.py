import hashlib, json, os, platform, subprocess, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.eval_retrieval_modes import load_jsonl, evaluate_cases, compare_modes, print_summary
from scripts.eval_metrics import aggregate_metrics
from shared.config import settings
from shared.retrieval_mode import RetrievalMode
from services.search_service.search import SearchService
from ingestion.lexical.bm25_index import get_bm25_index
from ingestion.embedding.bge_m3 import get_embedder
from services.search_service.reranker import get_reranker

def main():
 out=Path(__file__).resolve().parent
 dataset=ROOT/'tests/retrieval/validation_set.jsonl'
 cases=load_jsonl(dataset)
 assert len(cases)==220
 service=SearchService()
 meta={'started_utc':datetime.now(timezone.utc).isoformat(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'dataset_sha256':hashlib.sha256(dataset.read_bytes()).hexdigest(),'settings':settings.model_dump(mode='json'),'python':platform.python_version(),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'vector_chunks':service.hybrid_retriever.vector_store.count(),'bm25_chunks':get_bm25_index().size,'protocol':'Warm each mode with an excluded probe; unchanged evaluate_cases per query; alternate mode order per query; no LLM/API/planner evaluation.'}
 assert meta['vector_chunks'] and meta['bm25_chunks']
 for mode in RetrievalMode:
  service.retrieve('ما هي خدمات البنك الأهلي المصري؟','ar',retrieval_mode=mode,business_rules=mode==RetrievalMode.ENTERPRISE)
 meta['embedding_backend']=get_embedder()._backend
 meta['reranker_backend']=get_reranker()._backend
 (out/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
 results={m.value:[] for m in RetrievalMode}
 with (out/'per-query.jsonl').open('w') as f:
  for i,case in enumerate(cases):
   modes=list(RetrievalMode)
   if i%2: modes.reverse()
   for mode in modes:
    row=evaluate_cases([case],mode=mode)['results'][0]
    row.update(query_index=i+1,mode=mode.value,language=case['language'])
    results[mode.value].append(row)
    f.write(json.dumps(row,ensure_ascii=False)+'\n'); f.flush()
   print(f'PROGRESS {i+1}/{len(cases)}',flush=True)
 summaries={name:{**aggregate_metrics(rows),'total':len(rows),'mode':name} for name,rows in results.items()}
 payload={'dataset':str(dataset),'summaries':summaries,'comparison':compare_modes(summaries['PURE_SEMANTIC'],summaries['ENTERPRISE'])}
 (out/'comparison.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 for summary in summaries.values(): print_summary(summary)
 print('COMPLETE',flush=True)
if __name__=='__main__': main()
