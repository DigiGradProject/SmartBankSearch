# NBE Scrape Cleaning Report

- Output corpus: `C:/Users/Admin/Desktop/scarp/nbe-scrape-cleaner/output/documents.jsonl`
- Pages processed: **315** (AR=157, EN=158)
- Average raw text size: **34,328** chars
- Average cleaned content size: **1,818** chars
- Size reduction: **94.7%**
- Pages with tables inlined: **30**
- Thin/SPA shell pages (<120 cleaned chars): **175**
- Near-duplicate pairs flagged (contentful pages only): **16**
- Failed pages: **0**

## Notes

- Boilerplate was derived empirically per language from cross-page frequency.
- Base64/image dumps were stripped from all text fields.
- Near-duplicates are flagged via `metadata.duplicate_of`; nothing was deleted.
- Thin pages usually mean the scrape captured chrome only (JS-rendered body missing).

## Near-duplicate pairs (human review)

| Similarity | Canonical URL | Duplicate URL | duplicate_of |
|---|---|---|---|
| 1.000 | https://www.nbe.com.eg/NBE/E/#/AR/CustomerLogin | https://www.nbe.com.eg/NBE/E/#/AR/CustomerProfile | `fc7b814c27ce25e8` -> `e2a93e97cab557e1` |
| 1.000 | https://www.nbe.com.eg/NBE/E/#/AR/PrizesInfo | https://www.nbe.com.eg/NBE/E/#/AR/CustomerPrizes | `eba2af08e42ddbc5` -> `9c5b3de9b28fb8fc` |
| 1.000 | https://www.nbe.com.eg/NBE/E/#/AR/RADProfile | https://www.nbe.com.eg/NBE/E/#/AR/ProductOrderHandler | `1533b81345811883` -> `22cfcb37ada9b209` |
| 1.000 | https://www.nbe.com.eg/NBE/E/#/EN/CustomerLogin | https://www.nbe.com.eg/NBE/E/#/EN/CustomerProfile | `51353db1c57826b5` -> `6160408b60cdb090` |
| 1.000 | https://www.nbe.com.eg/NBE/E/#/EN/PrizesInfo | https://www.nbe.com.eg/NBE/E/#/EN/CustomerPrizes | `f4f36a115b8fe1a8` -> `192b925f41a16477` |
| 1.000 | https://www.nbe.com.eg/NBE/E/#/EN/Pay | https://www.nbe.com.eg/NBE/E/#/EN/Home | `cb5c72be21483b4a` -> `edf8e98ab84adaf6` |
| 0.993 | https://www.nbe.com.eg/NBE/E/#/AR/Sustainability | https://www.nbe.com.eg/NBE/E/#/AR/NBE_Sustainability | `4ca8ea01a8590a5f` -> `5801bc8fa2d8bc90` |
| 0.985 | https://www.nbe.com.eg/NBE/E/#/EN/Sustainability | https://www.nbe.com.eg/NBE/E/#/EN/OurSustainabilityApproach | `c8195b2cefd926ca` -> `d6227d2debb55781` |
| 0.984 | https://www.nbe.com.eg/NBE/E/#/EN/SocialImpactReport1 | https://www.nbe.com.eg/NBE/E/#/EN/NBESocialImpactReport | `f322bde5e7049e4d` -> `971ce916cc4c02a4` |
| 0.979 | https://www.nbe.com.eg/NBE/E/#/EN/NBE_Sustainability | https://www.nbe.com.eg/NBE/E/#/EN/OurSustainabilityApproach | `c8195b2cefd926ca` -> `34d27e01130274bf` |
| 0.978 | https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams={"CategoryID":"50","ProductID":"16641"} | https://www.nbe.com.eg/NBE/E/#/AR/ProductDetails?inParams={"CategoryID":"50","ProductID":"Ahly Net - Personal_16537"} | `9ce140920de14d06` -> `0aac7d1b91f83d7a` |
| 0.978 | https://www.nbe.com.eg/NBE/E/#/EN/Sustainability | https://www.nbe.com.eg/NBE/E/#/EN/NBE_Sustainability | `34d27e01130274bf` -> `d6227d2debb55781` |
| 0.977 | https://www.nbe.com.eg/NBE/E/#/EN/CategorySubCategory | https://www.nbe.com.eg/NBE/E/#/EN/ExchangeRatesAndCurrencyConverter | `2da6dd93db4b0399` -> `5f1d70225db8434d` |
| 0.976 | https://www.nbe.com.eg/NBE/E/#/AR/CategorySubCategory | https://www.nbe.com.eg/NBE/E/#/AR/ExchangeRatesAndCurrencyConverter | `5b6a1778e8369a12` -> `3e408b860a03d17b` |
| 0.971 | https://www.nbe.com.eg/NBE/E/#/AR/Sustainability | https://www.nbe.com.eg/NBE/E/#/AR/OurSustainabilityApproach | `cb287904ab07684e` -> `5801bc8fa2d8bc90` |
| 0.964 | https://www.nbe.com.eg/NBE/E/#/AR/NBE_Sustainability | https://www.nbe.com.eg/NBE/E/#/AR/OurSustainabilityApproach | `cb287904ab07684e` -> `4ca8ea01a8590a5f` |

## Failures

_None._
