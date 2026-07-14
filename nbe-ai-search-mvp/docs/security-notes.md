# Security Notes — On-Prem AI Search (Phase 4)

## Controls implemented / required

1. **No egress** from inference hosts to public LLM APIs  
2. **Audit log** (`data/logs/search_audit.jsonl`) stores query hash, intent, chunk IDs, model versions — not raw PII by default  
3. **Abstention** below confidence threshold (`NO_ANSWER` / unanswered)  
4. **Citations** required on answered paths  
5. **Feature flags** for heavy/optional ML (GLiNER, LLM rewrite, MiniLM) default off or conservative  

## Hardening checklist

- [ ] Block outbound internet on model servers  
- [ ] Rotate / restrict admin ingest endpoints (`/v1/admin/ingest`)  
- [ ] Restrict CORS to bank portals only  
- [ ] Secret scan CI on repo  
- [ ] Log retention policy for audit JSONL  
- [ ] Red-team prompt injection tests on RAG context  

## Threat notes

- Prompt injection via scraped web content → mitigate with structured formatter + extractive bypass for high-risk intents  
- Synonym poisoning → human-owned JSON ontology with validation  
- Model supply chain → pin hashes / internal mirror of HuggingFace/Ollama models  
