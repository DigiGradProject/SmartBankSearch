from services.llm_service.llm import LLMService


def test_tier1_default_for_simple_query():
    service = LLMService()
    assert service._select_model("ما هي خدمة فون كاش؟") == service.tier1_model


def test_tier2_for_complex_query():
    service = LLMService()
    model = service._select_model("قارن بين شهادات بلادي سنة وثلاث سنوات واشرح الشروط")
    assert model == service.tier2_model


def test_strips_think_blocks():
    service = LLMService()
    raw = "<think>reasoning here</think>\nشهادات بلادي متاحة بالدولار [1]."
    assert service._clean_answer(raw) == "شهادات بلادي متاحة بالدولار [1]."
