"""얼린 파이프라인이 노트북 판정과 성적을 그대로 재현하는가. API 키·이미지 없이."""
from datetime import date

import pytest
from evaluate import FIXTURES, as_json, load, run, score

from app.engine.vlm import crop_regions

SPLITS = [s for s in ("dev", "test") if (FIXTURES / f"eval_{s}.json.gz").exists()]


@pytest.fixture(scope="module", params=SPLITS)
def data(request):
    return load(request.param)


def test_matches_notebook(data):
    today = date.fromisoformat(data["today"])
    mismatched = []
    for doc in data["docs"]:
        result, sent = run(doc, today)
        where = (doc["doc_id"], doc["level"])
        assert result.fields == doc["fields"], where
        assert result.reread == doc["reread"], where
        if doc["fields"] is not None:                     # 크롭 위치 · 확대 · 여백이 노트북과 같은가
            assert [list(b) for b in crop_regions(result.fields).values()] == list(doc["crop_regions"].values()), where
        assert sent == {k: c["size"] for k, c in doc["calls"].items()}, where
        got = as_json(result)
        assert len(got) == len(doc["verdicts"]), where
        mismatched += [(*where, e["field"]) for e, g in zip(doc["verdicts"], got) if e != g]
    assert not mismatched, mismatched[:10]


def test_dev_score():
    """개발용 성적(노트북 V3 분석 ② · 상쇄 가드): 놓친 오답 0 · 의심 10% / 31% / 82%"""
    s = score(load("dev"))
    assert [s[lv]["놓친 오답"] for lv in s] == [0, 0, 0]
    assert [round(s[lv]["의심"] / s[lv]["칸"] * 100) for lv in s] == [10, 31, 82]


def test_test_score():
    """시험용 성적(노트북 9d, 규칙 동결 뒤 한 번): 놓친 오답 0 · 의심 15% / 33% / 79%"""
    s = score(load("test"))
    assert [s[lv]["놓친 오답"] for lv in s] == [0, 0, 0]
    assert [round(s[lv]["의심"] / s[lv]["칸"] * 100) for lv in s] == [15, 33, 79]
