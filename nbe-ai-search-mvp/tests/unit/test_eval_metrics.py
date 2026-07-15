"""Unit tests for evaluation metric helpers."""

from scripts.eval_metrics import mrr, ndcg_at_k, recall_at_k, relevance_labels


def test_recall_mrr_ndcg_basic():
    labels = relevance_labels(
        [
            "https://x/Accounts",
            "https://x/ExchangeRatesAndCurrencyConverter",
            "https://x/Loans",
        ],
        "ExchangeRatesAndCurrencyConverter",
    )
    assert labels == [0, 1, 0]
    assert recall_at_k(labels, k=5) == 1.0
    assert mrr(labels) == 0.5
    assert ndcg_at_k(labels, k=5) > 0.0


def test_miss_metrics_are_zero():
    labels = [0, 0, 0]
    assert recall_at_k(labels, k=5) == 0.0
    assert mrr(labels) == 0.0
    assert ndcg_at_k(labels, k=5) == 0.0
