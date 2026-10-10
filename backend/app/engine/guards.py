"""어떤 검산을 증거로 칠지 거른다.

같은 VLM 이 읽은 값끼리 맞는 것은 증거가 아니다(잘못 읽은 단가로 금액을 계산해 쓴다).
검산은 독립 출처로 확인된 합계(기준점)에서 내려올 때만 증거가 된다.
"""
from app.engine.checks import CheckResult, Fields, item_count, kor_to_int, parse_amount
from app.engine.ocr import Token, ground

CHAIN = ("단가×수량", "Σ품목=공급가액", "부가세=10%", "공급가+부가세=합계")


def anchor_total(x: Fields, token_sets: list[list[Token]], v3_total: str | None = None) -> int | None:
    """한글 금액 · OCR · V3 합계 크롭 중 하나와 일치하는 합계. 없으면 None."""
    t = parse_amount(x.get("total"))
    if t is None:
        return None
    if kor_to_int(x.get("total_kor")) == t or ground(x.get("total"), "num", token_sets):
        return t
    if v3_total is not None and parse_amount(v3_total) == t:
        return t
    return None


def supply_from_total(t: int) -> int | None:
    """공급가액 + 공급가액//10 = 합계 를 만족하는 공급가액. 해는 많아야 하나다."""
    s = round(t / 1.1)
    return next((c for c in range(s - 2, s + 3) if c + c // 10 == t), None)


def chain_case(x: Fields, token_sets: list[list[Token]], v3_total: str | None = None) -> str:
    """"정상" | "기준점 없음" | "불일치"."""
    t = anchor_total(x, token_sets, v3_total)
    if t is None:
        return "기준점 없음"
    s = supply_from_total(t)
    amounts = [parse_amount(x.get(f"item{i}.amount")) for i in range(item_count(x))]
    if s is None or None in amounts:
        return "기준점 없음"
    return "정상" if sum(amounts) == s else "불일치"


def demote(checks: list[CheckResult], names: tuple[str, ...], why: str) -> list[CheckResult]:
    """통과를 '신호 없음'(None)으로 내린다. 실패는 그대로 둔다."""
    return [CheckResult(c.name, None, f"{c.detail} · {why}", c.fields)
            if c.passed is True and c.name.startswith(names) else c for c in checks]


def guard_checks(x: Fields, checks: list[CheckResult], token_sets: list[list[Token]],
                 v3_total: str | None = None) -> list[CheckResult]:
    case = chain_case(x, token_sets, v3_total)
    n = item_count(x)
    totals = ["supply", "vat", "total"]
    amounts = [f"item{i}.amount" for i in range(n)]

    if case == "정상":
        row_failed = any(c.passed is False and c.name.startswith("단가×수량") for c in checks)
        if row_failed:
            # 틀린 줄이 있는데 Σ 가 맞으면 다른 줄이 상쇄했을 수 있다
            return (demote(checks, ("단가×수량", "Σ품목=공급가액"), "틀린 줄이 있어 Σ 일치는 상쇄일 수 있음")
                    + [CheckResult("사슬 확인", True, "합계 독립 확인 → 공급가액 역산 · 품목은 보증하지 않음", totals)])
        return checks + [CheckResult("사슬 확인", True, "합계 독립 확인 → 공급가액 역산 → Σ품목 일치", totals + amounts)]

    demoted = demote(checks, CHAIN, "기준점 없음: VLM이 계산했을 수 있음")
    if case == "기준점 없음":
        return demoted
    upstream = totals + amounts + [f"item{i}.{k}" for i in range(n) for k in ("unit_price", "qty")]
    return demoted + [CheckResult("사슬 불일치", False, "기준점에서 구한 공급가액과 Σ품목이 다름", upstream)]
