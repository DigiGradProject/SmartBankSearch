"""Intent classifier façade."""

from services.search_service.intent_classifier import (
    QueryIntent,
    classify_query,
    should_apply_metadata_filter,
)

__all__ = ["QueryIntent", "classify_query", "should_apply_metadata_filter"]
