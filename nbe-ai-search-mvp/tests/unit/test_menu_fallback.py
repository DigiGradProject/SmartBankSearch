from shared.document_quality import is_menu_heavy_text
from services.llm_service.llm import LLMService


def test_flattened_navbar_is_menu_heavy():
    text = (
        "اهلا بك | الخروج القوائم الرئيسيه افراد الشركات المشروعات الصغيره بلاتينم عن البنك "
        "الاستدامه سعر الصرف و تحويل العملات افتح حسابك وانت في مكانك الخدمات الذاتيه "
        "الشهادات الاجنبيه الشهادات المحليه شهادات بلادي حسابات التوفير بطاقات الائتمان "
        "القروض الشخصية الاهلي نت الفروع وماكينات الصرف الالي عروض التقسيط و الخصومات "
        "اتصل بنا محليا: 19623 " + ("منتج خدمة " * 40)
    )
    assert is_menu_heavy_text(text) is True


def test_fallback_rejects_menu_for_rate_query():
    service = LLMService()
    context = (
        "[1] اهلا بك | الخروج القوائم الرئيسيه افراد الشركات الشهادات المحليه شهادات بلادي "
        "حسابات التوفير بطاقات الائتمان القروض الشخصية الاهلي نت الفروع وماكينات الصرف "
        + ("قائمة منتجات " * 50)
    )
    answer = service._fallback_answer(
        "كم فايده شهادة سنة بالجنيه",
        context,
        "ar",
    )
    assert answer is None
