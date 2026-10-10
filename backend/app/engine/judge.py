"""칸마다 자동처리 / 사람 검토. 기본은 검토이고, 통과한 신호가 있어야 자동이다."""
import re
from dataclasses import dataclass

from app.engine.checks import CheckResult, Fields, kor_to_int, parse_amount
from app.engine.ocr import BBox, Token, ground_by_mode

TEXT_FIELDS = {"client", "supplier", "manager", "total_kor"}
MIN_SIGNALS = 1


@dataclass
class FieldVerdict:
    field: str
    value: str | None
    verdict: str            # "auto" | "review"
    reasons: list[str]
    bbox: BBox | None       # OCR 근거 위치
    signals: int


def kind_of(field: str) -> str:
    return "text" if field in TEXT_FIELDS or field.endswith(".name") else "num"


def parsable(field: str, v: str) -> bool:
    if field == "total_kor":
        return kor_to_int(v) is not None
    if field == "supplier_brn":
        return len(re.sub(r"\D", "", v)) == 10
    if field == "date":
        return re.search(r"\d{4}\D+\d{1,2}\D+\d{1,2}", v) is not None
    if field == "doc_no" or kind_of(field) == "text":
        return True
    return parse_amount(v) is not None


def judge(x: Fields, checks: list[CheckResult], token_sets: list[list[Token]]) -> list[FieldVerdict]:
    """신호 = 증거로 남은 검산 통과 + OCR 근거(두 읽기 모드가 일치할 때만).

    실패한 검산은 그 검산에 걸린 칸만 검토로 보낸다. OCR 근거가 없는 것만으로는 빨간불이 아니다.
    """
    out = []
    for field, v in x.items():
        if not v:
            out.append(FieldVerdict(field, v, "review", ["값 없음"], None, 0))
            continue
        mine = [c for c in checks if field in c.fields]
        reasons = [f"검산 실패: {c.name} ({c.detail})" for c in mine if c.passed is False]
        signals = sum(c.passed is True for c in mine)
        if not parsable(field, v):
            reasons.append("해석 불가")
        bbox, agreement = ground_by_mode(v, kind_of(field), token_sets)
        if bbox is not None and agreement == "일치":
            signals += 1
        if signals < MIN_SIGNALS:
            reasons.append(f"통과한 신호 {signals}개")
        out.append(FieldVerdict(field, v, "review" if reasons else "auto", reasons, bbox, signals))
    return out
