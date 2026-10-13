"""재현용 평가: 고정 데이터(노트북 9c)를 얼린 process() 에 넣어 판정하고 정답으로 채점한다. API 0.

    python tests/evaluate.py          # backend/ 에서. 있는 split 전부
"""
import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

from PIL import Image

from app.engine.checks import kor_to_int, parse_amount, same_date
from app.engine.judge import kind_of
from app.engine.ocr import norm
from app.engine.pipeline import Result, process

FIXTURES = Path(__file__).parent / "fixtures"
LEVELS = ("L0.5", "L1", "L2")
_BLANK = Image.new("RGB", (900, 1273), "white")   # 픽셀은 쓰지 않는다 — 응답은 프롬프트로 찾고 크롭은 크기로 검증


def load(split: str) -> dict:
    with gzip.open(FIXTURES / f"eval_{split}.json.gz", "rt", encoding="utf-8") as f:
        return json.load(f)


def run(doc: dict, today: date) -> tuple[Result, dict]:
    """얼린 process() 에 기록된 응답·OCR 을 넣는다. 돌려주는 sent = 프롬프트 해시 → 보낸 이미지 크기"""
    sent = {}

    def chat(img, prompt, max_tokens):
        key = hashlib.md5(prompt.encode()).hexdigest()
        sent[key] = list(img.size)
        return doc["calls"][key]["response"] or ""        # 프롬프트가 바뀌면 KeyError · 미실행 = null

    return process(_BLANK, chat, read_ocr=lambda _: doc["ocr"], today=today), sent


def as_json(result: Result) -> list[dict]:
    return json.loads(json.dumps([asdict(v) for v in result.verdicts], ensure_ascii=False))


def correct(field: str, v: str | None, truth: str | None) -> bool:
    """정답과 같은 값인가. 날짜는 표기 무관, 금액은 숫자로. null 과 정답에 없는 칸은 틀림"""
    if not v or truth is None:
        return False
    if field == "date":
        return same_date(v, truth)
    if field == "total_kor":
        return kor_to_int(v) == kor_to_int(truth)
    if kind_of(field) == "text" or field in ("doc_no", "supplier_brn"):
        return norm(v, kind_of(field)) == norm(truth, kind_of(field))
    return parse_amount(v) == parse_amount(truth)


def outcomes(result: Result, truths: dict) -> list[str]:
    """칸마다 정상 · 오탐 · 놓친 오답 · 잡음. 정답에 있는데 추출 안 된 칸은 검토로 센다"""
    judged = {v.field: (v.value, v.verdict) for v in result.verdicts}
    judged |= {f: (None, "review") for f in truths if f not in judged}
    names = {(True, "auto"): "정상", (True, "review"): "오탐", (False, "auto"): "놓친 오답", (False, "review"): "잡음"}
    return [names[(correct(f, v, truths.get(f)), verdict)] for f, (v, verdict) in judged.items()]


def score(data: dict) -> dict[str, dict]:
    """단계별 칸 수 · 놓친 오답 · 의심(검토로 보낸 칸)"""
    today = date.fromisoformat(data["today"])
    out = {lv: {"칸": 0, "놓친 오답": 0, "의심": 0} for lv in LEVELS}
    for doc in data["docs"]:
        s = out[doc["level"]]
        for o in outcomes(run(doc, today)[0], doc["truths"]):
            s["칸"] += 1
            s["놓친 오답"] += o == "놓친 오답"
            s["의심"] += o in ("오탐", "잡음")
    return out


if __name__ == "__main__":
    for split in ("dev", "test"):
        if not (FIXTURES / f"eval_{split}.json.gz").exists():
            continue
        print(split)
        for lv, s in score(load(split)).items():
            print(f"  {lv:5} 칸 {s['칸']:5} · 놓친 오답 {s['놓친 오답']} · 의심 {s['의심'] / s['칸']:.1%}")
