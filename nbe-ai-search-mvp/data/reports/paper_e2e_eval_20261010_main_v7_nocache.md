# NBE RAG End-to-End Evaluation

**Run:** `20261010_main_v7_nocache`  
**Date (UTC):** 2026-10-10 10:03  
**Cases:** 26 total; 26 completed; 0 errors  
**Generation / judge model:** `qwen3:8b` via Ollama  
**Dataset SHA-256:** `e3167d24a2e00deed6d6cfc11975684ad953c31ef806ef3bfa2670e2910421b8`  
**Dataset:** `tests/retrieval/paper_e2e_cases.jsonl`

## Purpose and protocol

Each case is sent to the running `/v1/search` AI endpoint with debug enabled. The system performs its configured retrieval, builds context, generates an answer when it decides evidence is sufficient, and returns citations or an abstention. The debug response records the exact context assembled for generation and the ranked retrieved documents. Each non-empty answer is then independently labeled by a Qwen3 faithfulness check against that captured context, using the production rubric: `SUPPORTED`, `PARTIALLY_SUPPORTED`, or `UNSUPPORTED`.

The dataset combines the repository's 22-query retrieval golden set with four explicit abstention cases. Expected source URL patterns are inherited from the golden annotations. `gold citation hit` means at least one cited URL contains one of the annotated patterns; `gold retrieval hit` applies the same rule to retrieved URLs.

## Results

| Metric | Result |
|---|---:|
| Cases completed | 26 / 26 |
| Answerability accuracy | 100.0% |
| Strict end-to-end success (answer/abstain + supported answer + source) | 69.2% |
| Answer rate on expected-answerable cases | 100.0% |
| Correct abstention rate | 100.0% |
| Abstention precision / recall / F1 | 100.0% / 100.0% / 100.0% |
| Gold-source retrieval hit rate | 100.0% |
| Gold-source citation hit rate | 88.2% |
| Gold-source citation precision (micro) | 69.4% |
| Citation coverage among answered queries | 100.0% |
| Qwen3 faithfulness: supported / partial / unsupported | 72.7% / 22.7% / 4.5% (n=22) |
| Forbidden-source citation rate | 0.0% |
| Mean / median response latency | 17.07s / 17.12s |
| Answerability accuracy, English / Arabic | 100.0% / 100.0% |

## Per-case results

| ID | Lang. | Expected | Answered | Faithfulness | Gold citation | Correct | Strict E2E | Latency | Query |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| banking_001 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 26.64s | أسعار العملات بما يعادل الجنيه المصري |
| banking_002 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 26.69s | أسعار العملات |
| banking_003 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 23.51s | سعر الصرف |
| banking_004 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 28.59s | تحويل العملات |
| banking_005 | en | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 28.86s | exchange rates NBE |
| banking_006 | en | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 26.04s | currency converter |
| banking_007 | ar | Answer | Yes | PARTIALLY_SUPPORTED | Yes | Yes | No | 17.12s | قرض شخصي |
| banking_008 | en | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 14.96s | personal loan NBE |
| banking_009 | ar | Answer | Yes | UNSUPPORTED | Yes | Yes | No | 3.67s | بطاقات ائتمان |
| banking_010 | en | Answer | Yes | PARTIALLY_SUPPORTED | Yes | Yes | No | 2.91s | credit cards NBE |
| banking_011 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 20.70s | الفروع |
| banking_012 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 17.12s | الصراف الآلي |
| banking_013 | ar | Answer | Yes | SUPPORTED | No | Yes | No | 3.51s | شراء شهادة |
| banking_014 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 11.95s | كم فايده شهادة سنة بالجنيه |
| banking_015 | ar | Answer | Yes | PARTIALLY_SUPPORTED | Yes | Yes | No | 17.03s | أسعار الشهادات بالعملة الأجنبية |
| banking_016 | ar | Answer | Yes | SUPPORTED | No | Yes | No | 38.82s | فتح حساب بنكي |
| banking_017 | ar | Answer | Yes | SUPPORTED | Yes | Yes | Yes | 14.15s | لو عايز افتح حساب بنكي اي الاوراق المطلوبة؟ |
| banking_018 | ar | Answer | Yes | PARTIALLY_SUPPORTED | — | Yes | No | 3.59s | انواع البطاقات البنكيه |
| banking_019 | ar | Answer | Yes | SUPPORTED | — | Yes | Yes | 21.67s | عروض البنك |
| banking_020 | ar | Answer | Yes | SUPPORTED | — | Yes | Yes | 29.12s | خدمات الشركات |
| banking_021 | ar | Answer | Yes | SUPPORTED | — | Yes | Yes | 28.43s | المشروعات الصغيرة والمتوسطة |
| banking_022 | ar | Answer | Yes | PARTIALLY_SUPPORTED | — | Yes | No | 20.00s | الأسئلة الشائعة |
| abstain_001 | en | Abstain | No | NOT_APPLICABLE | — | Yes | Yes | 3.33s | asdkjhqwe zzz qwerty |
| abstain_002 | en | Abstain | No | NOT_APPLICABLE | — | Yes | Yes | 3.04s | What is the weather in Cairo tomorrow? |
| abstain_003 | en | Abstain | No | NOT_APPLICABLE | — | Yes | Yes | 9.38s | What is my current NBE account balance? |
| abstain_004 | en | Abstain | No | NOT_APPLICABLE | — | Yes | Yes | 2.88s | Who won Egypt vs Brazil today? |

## Interpretation and limitations

These are automated evaluation results, not human adjudication. The Qwen3 judge is an LLM-as-judge signal and may share model-family biases with generation; it should not be described as ground-truth human faithfulness. Gold-source checks are URL-substring annotations and measure whether a relevant source family appeared, not whether every individual claim is fully supported. The abstention set is small and intentionally includes different failure modes. For publication, retain the CSV evidence and have at least two reviewers independently label answer correctness, claim-level support, citation relevance, and abstention appropriateness; report inter-annotator agreement and adjudication.

The CSV contains each query, expected labels, answer, exact captured generation context, retrieved evidence metadata, cited URLs, judge outputs, and latency. Review cases with `cache_hit=true`, judge errors, missing context, or endpoint errors before calculating final paper numbers.
