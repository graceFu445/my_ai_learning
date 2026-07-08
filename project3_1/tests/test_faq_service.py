from pathlib import Path

from app.services.faq_service import FaqService
from app.storage.json_store import JsonStore


class RebuildSpy:
    def __init__(self):
        self.calls = 0

    def rebuild(self, faqs):
        self.calls += 1
        self.last_faqs = faqs


def test_import_csv_creates_faqs_and_rebuilds_index(tmp_path: Path):
    store = JsonStore(tmp_path)
    index = RebuildSpy()
    service = FaqService(store, index)
    csv_text = (
        "question,answer,tags\n"
        "如何退货？,签收后7天内可申请退货。,售后;退货\n"
        "配送多久？,普通地区1-3天送达。,配送,时效\n"
    )

    imported = service.import_csv(csv_text)

    assert len(imported) == 2
    assert imported[0]["tags"] == ["售后", "退货"]
    assert imported[1]["tags"] == ["配送", "时效"]
    assert index.calls == 1
    assert len(index.last_faqs) == 2


def test_update_and_delete_rebuild_index(tmp_path: Path):
    store = JsonStore(tmp_path)
    index = RebuildSpy()
    service = FaqService(store, index)
    faq = service.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])

    service.update_faq(faq["id"], answer="签收后7天内可在线申请退货。")
    service.delete_faq(faq["id"])

    assert index.calls == 3
    assert store.list_faqs() == []
