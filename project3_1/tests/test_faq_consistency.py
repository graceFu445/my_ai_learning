from pathlib import Path

import pytest

from app.services.faq_service import FaqService
from app.storage.json_store import JsonStore


class FailingIndex:
    def rebuild(self, faqs):
        raise RuntimeError("index rebuild failed")


class RecordingIndex:
    def __init__(self):
        self.snapshots = []

    def rebuild(self, faqs):
        self.snapshots.append([faq.copy() for faq in faqs])


def test_create_faq_does_not_persist_when_rebuild_fails(tmp_path: Path):
    store = JsonStore(tmp_path)
    service = FaqService(store, FailingIndex())

    with pytest.raises(RuntimeError):
        service.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])

    assert store.list_faqs() == []


def test_update_faq_does_not_persist_when_rebuild_fails(tmp_path: Path):
    store = JsonStore(tmp_path)
    service = FaqService(store, RecordingIndex())
    faq = service.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])

    service.index = FailingIndex()
    with pytest.raises(RuntimeError):
        service.update_faq(faq["id"], answer="新的退货说明")

    assert store.get_faq(faq["id"])["answer"] == "签收后7天内可申请退货。"


def test_delete_faq_does_not_persist_when_rebuild_fails(tmp_path: Path):
    store = JsonStore(tmp_path)
    service = FaqService(store, RecordingIndex())
    faq = service.create_faq("如何退货？", "签收后7天内可申请退货。", ["售后"])

    service.index = FailingIndex()
    with pytest.raises(RuntimeError):
        service.delete_faq(faq["id"])

    assert store.get_faq(faq["id"])["id"] == faq["id"]


def test_import_csv_does_not_persist_partial_rows_when_rebuild_fails(tmp_path: Path):
    store = JsonStore(tmp_path)
    service = FaqService(store, FailingIndex())
    csv_text = (
        "question,answer,tags\n"
        "如何退货？,签收后7天内可申请退货。,售后;退货\n"
        "配送多久？,普通地区1-3天送达。,配送,时效\n"
    )

    with pytest.raises(RuntimeError):
        service.import_csv(csv_text)

    assert store.list_faqs() == []
