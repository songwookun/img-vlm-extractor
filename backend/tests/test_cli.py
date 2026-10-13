"""CLI: 하루 한도에서 저장하고 멈추기, 다시 실행하면 이어하기, 호출 상한. API 0."""
import csv

import pytest
from PIL import Image

from app.cli import EX_TEMPFAIL, main
from app.engine import vlm

V0 = ('{"doc_no": "Q-1", "date": "2026년 10월 1일", "client": "테스트확인 주식회사", "supplier": "(주)샘플상사",'
      ' "supplier_brn": "000-00-00000", "manager": "홍길동", "items": [{"name": "A4", "qty": "1",'
      ' "unit_price": "1,000", "amount": "1,000"}], "supply": "1,000", "vat": "100", "total": "1,100",'
      ' "total_kor": "일천일백원정"}')


class FakeClient:
    model = "fake"

    def __init__(self, limit_after: int | None = None):
        self.calls: list[dict] = []
        self.limit_after = limit_after

    def __call__(self, img, prompt, max_tokens):
        if self.limit_after is not None and len(self.calls) >= self.limit_after:
            raise vlm.DailyLimit("per day")
        self.calls.append({"prompt": prompt[:8], "tokens": 1, "seconds": 0})
        return V0 if prompt == vlm.PROMPT_V0 else "{}"


def no_ocr(img):
    return [[], []]


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        Image.new("RGB", (900, 1273), "white").save(d / f"q{i}.png")
    return d


def run(folder, out, client, *extra):
    return main([str(folder), "-o", str(out), *extra], client=client, read_ocr=no_ocr, preflight=False)


def test_daily_limit_then_resume(folder, tmp_path):
    out = tmp_path / "out"
    assert run(folder, out, FakeClient(limit_after=4)) == EX_TEMPFAIL     # 1장 끝, 2장째 도중 한도
    assert sorted(p.name for p in (out / "results").iterdir()) == ["q0.json"]

    again = FakeClient()
    assert run(folder, out, again) == 0
    assert len(again.calls) == 6                                         # 남은 2장만
    assert len(list((out / "results").iterdir())) == 3
    with (out / "review.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows and {r["doc_id"] for r in rows} == {"q0", "q1", "q2"}    # OCR 근거가 없으니 모든 문서에 검토 칸
    assert all(r["crop_path"] for r in rows if r["bbox"])


def test_max_calls(folder, tmp_path):
    client = FakeClient()
    assert run(folder, tmp_path / "out", client, "--max-calls", "4") == EX_TEMPFAIL
    assert len(client.calls) == 3


class FailingClient(FakeClient):
    def __init__(self, error):
        super().__init__()
        self.error = error

    def __call__(self, img, prompt, max_tokens):
        raise self.error


def test_transient_failure_skips_doc(folder, tmp_path):
    out = tmp_path / "out"
    assert run(folder, out, FailingClient(vlm.VlmError("503"))) == EX_TEMPFAIL
    assert not list((out / "results").iterdir())


def test_fatal_failure_stops(folder, tmp_path):
    assert run(folder, tmp_path / "out", FailingClient(vlm.VlmFatal("401"))) == 1


def test_leftover_tmp_is_not_done(folder, tmp_path):
    out = tmp_path / "out"
    (out / "results").mkdir(parents=True)
    (out / "results" / "q0.json.tmp").write_text('{"fields": ')        # 쓰다 끊긴 결과
    client = FakeClient()
    assert run(folder, out, client) == 0
    assert len(client.calls) == 9


@pytest.mark.parametrize("extra", [["--max-calls", "-1"]])
def test_rejects_negative_budget(folder, tmp_path, extra):
    with pytest.raises(SystemExit):
        run(folder, tmp_path / "out", FakeClient(), *extra)


def test_rejects_missing_folder_and_duplicate_names(folder, tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path / "nope", tmp_path / "out", FakeClient())
    Image.new("RGB", (900, 1273), "white").save(folder / "q0.jpg")
    with pytest.raises(SystemExit):
        run(folder, tmp_path / "out", FakeClient())


def test_transparent_png_becomes_white(tmp_path):
    from app.cli import open_image
    p = tmp_path / "a.png"
    Image.new("RGBA", (4, 4), (0, 0, 0, 0)).save(p)
    assert open_image(p).getpixel((0, 0)) == (255, 255, 255)


def test_rejected_doc_is_reported_not_retried(folder, tmp_path):
    out = tmp_path / "out"
    assert run(folder, out, FailingClient(vlm.VlmRejected("413"))) == 0       # 사람에게 넘기고 끝 — 75 로 무한 재시도하지 않는다
    assert not list((out / "results").iterdir())


def test_case_insensitive_duplicate_names(folder, tmp_path):
    Image.new("RGB", (900, 1273), "white").save(folder / "Q1.jpg")
    with pytest.raises(SystemExit):
        run(folder, tmp_path / "out", FakeClient())


def test_corrupt_result_is_reprocessed(folder, tmp_path):
    out = tmp_path / "out"
    run(folder, out, FakeClient())
    (out / "results" / "q0.json").write_text("{}")
    client = FakeClient()
    assert run(folder, out, client) == 0
    assert len(client.calls) == 3


def test_open_image_normalizes(tmp_path):
    from app.cli import open_image
    p = tmp_path / "rot.jpg"
    img = Image.new("RGB", (20, 10), "white")
    exif = img.getexif()
    exif[0x0112] = 6                                                          # 90도 회전해 찍힌 사진
    img.save(p, exif=exif)
    assert open_image(p).size == (10, 20)
    p16 = tmp_path / "gray16.png"
    Image.new("I;16", (2, 2), 40000).save(p16)
    assert open_image(p16).getpixel((0, 0)) == (156, 156, 156)


def test_preflight_stops_before_any_call(folder, tmp_path, monkeypatch):
    from app.engine import ocr

    def missing():
        raise RuntimeError("kor 없음")
    monkeypatch.setattr(ocr, "check_ready", missing)
    client = FakeClient()
    assert main([str(folder), "-o", str(tmp_path / "out")], client=client, read_ocr=no_ocr) == 1
    assert not client.calls


def test_result_round_trip_keeps_bbox(tmp_path):
    from app.cli import load_result, save_result
    from app.engine.judge import FieldVerdict
    from app.engine.pipeline import Result
    r = Result({"total": "1"}, {"total": "1", "date": None}, [FieldVerdict("total", "1", "review", ["x"], (1, 2, 3, 4), 0)])
    save_result(tmp_path / "a.json", r, "m", [])
    assert load_result(tmp_path / "a.json") == r
