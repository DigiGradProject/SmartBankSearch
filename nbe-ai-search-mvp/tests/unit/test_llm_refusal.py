from services.llm_service.llm import LLMService


def test_detects_arabic_financial_refusal():
    service = LLMService()
    answer = "لا أستطيع تقديم معلومات شخصية أو مالية. هل يمكنني مساعدتك في طلب المعلومات العامة عن الشهادة؟"
    assert service._is_refusal(answer) is True


def test_detects_english_financial_refusal():
    service = LLMService()
    answer = "I cannot provide personal or financial information about certificates."
    assert service._is_refusal(answer) is True


def test_normal_answer_is_not_refusal():
    service = LLMService()
    answer = "شهادات بلادي سنة متاحة بالدولار واليورو وفق صفحة البنك الأهلي [1]."
    assert service._is_refusal(answer) is False
