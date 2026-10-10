"""결정론 검사. 입력은 VLM 이 읽은 값(flatten 된 dict)뿐이다."""
import re
from dataclasses import dataclass
from datetime import date

Fields = dict[str, str | None]

_DIGITS = "영일이삼사오육칠팔구"
_SMALL_UNITS = {"십": 10, "백": 100, "천": 1000}
_BIG_UNITS = {"만": 10**4, "억": 10**8, "조": 10**12}
_BRN_WEIGHTS = (1, 3, 7, 1, 3, 7, 1, 3, 5)
_YMD = re.compile(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})")


@dataclass
class CheckResult:
    name: str
    passed: bool | None   # None = 검사 불가. 통과로 치지 않는다
    detail: str
    fields: list[str]


def parse_amount(s) -> int | None:
    """마지막 구분자 뒤 1~2자리는 소수, 3자리는 천 단위. 숫자가 아니면 None."""
    if s is None:
        return None
    t = re.sub(r"[\s원₩]", "", str(s))
    m = re.fullmatch(r"(.*?)(?:[.,](\d{1,2}))?", t)
    if m.group(2) and int(m.group(2)) != 0:   # 원화에 소수는 없다
        return None
    t = re.sub(r"[.,]", "", m.group(1))
    return int(t) if t.isdigit() else None


def kor_to_int(s) -> int | None:
    """한글 금액 → 정수. 모르는 글자가 섞이면 None."""
    if not s:
        return None
    s = re.sub(r"^일금|원정?$|\s", "", str(s))
    total = group = num = 0
    for ch in s:
        if ch in _DIGITS:
            num = _DIGITS.index(ch)
        elif ch in _SMALL_UNITS:
            group += (num or 1) * _SMALL_UNITS[ch]
            num = 0
        elif ch in _BIG_UNITS:
            total += (group + num or 1) * _BIG_UNITS[ch]
            group = num = 0
        else:
            return None
    return total + group + num


def brn_check_digit(d9: list[int]) -> int:
    s = sum(d * w for d, w in zip(d9, _BRN_WEIGHTS)) + (d9[8] * 5) // 10
    return (10 - s % 10) % 10


def to_date(s: str | None) -> date | None:
    m = _YMD.search(s or "")
    if not m:
        return None
    try:
        return date(*map(int, m.groups()))
    except ValueError:
        return None


def _ymd(s: str) -> tuple[int, ...] | None:
    m = _YMD.search(s)
    return tuple(map(int, m.groups())) if m else None


def same_date(a: str | None, b: str | None) -> bool:
    """표기와 무관하게 같은 연·월·일인가. 둘 다 해석 불가면 같다고 본다."""
    if not a or not b:
        return False
    return _ymd(a) == _ymd(b)


def item_count(x: Fields) -> int:
    return len({k.split(".")[0] for k in x if k.startswith("item")})


def check_year(x: Fields) -> CheckResult:
    """날짜 연도 = 문서번호 연도. 반증 전용: 두 칸이 같은 착시로 함께 틀리기 때문."""
    d = re.search(r"(\d{4})\D", x.get("date") or "")
    n = re.search(r"(20\d{2})", x.get("doc_no") or "")
    fields = ["date", "doc_no"]
    if not (d and n):
        return CheckResult("날짜=문서번호 연도", None, f"{x.get('date')} / {x.get('doc_no')}", fields)
    if d.group(1) == n.group(1):
        return CheckResult("날짜=문서번호 연도", None, f"{d.group(1)} vs {n.group(1)} · 일치는 반증 실패일 뿐", fields)
    return CheckResult("날짜=문서번호 연도", False, f"{d.group(1)} vs {n.group(1)}", fields)


def check_quote(x: Fields) -> list[CheckResult]:
    """견적서 검사 8종. 재료가 하나라도 없으면 그 검사는 None."""
    def amt(k: str) -> int | None:
        return parse_amount(x.get(k))

    n = item_count(x)
    out = []
    for i in range(n):
        u, q, a = amt(f"item{i}.unit_price"), amt(f"item{i}.qty"), amt(f"item{i}.amount")
        out.append(CheckResult(f"단가×수량[{i}]", None if None in (u, q, a) else u * q == a,
                               f"{u}×{q} vs {a}", [f"item{i}.unit_price", f"item{i}.qty", f"item{i}.amount"]))

    amounts, s = [amt(f"item{i}.amount") for i in range(n)], amt("supply")
    out.append(CheckResult("Σ품목=공급가액", None if None in amounts + [s] else sum(amounts) == s,
                           f"Σ{sum(a or 0 for a in amounts)} vs {s}", [f"item{i}.amount" for i in range(n)] + ["supply"]))

    v, t = amt("vat"), amt("total")
    out.append(CheckResult("부가세=10%", None if None in (s, v) else abs(s * 0.1 - v) <= 1,
                           f"{s}×10% vs {v}", ["supply", "vat"]))
    out.append(CheckResult("공급가+부가세=합계", None if None in (s, v, t) else s + v == t,
                           f"{s}+{v} vs {t}", ["supply", "vat", "total"]))

    k = kor_to_int(x.get("total_kor"))
    out.append(CheckResult("한글금액=합계", None if None in (k, t) else k == t, f"{k} vs {t}", ["total_kor", "total"]))

    d = re.sub(r"\D", "", x.get("supplier_brn") or "")
    out.append(CheckResult("사업자번호 체크섬",
                           None if len(d) != 10 else brn_check_digit([int(c) for c in d[:9]]) == int(d[9]),
                           d, ["supplier_brn"]))

    has_date = _YMD.search(x.get("date") or "") is not None
    out.append(CheckResult("날짜 실재", to_date(x.get("date")) is not None if has_date else None,
                           x.get("date"), ["date"]))
    out.append(check_year(x))
    return out


def check_recent(x: Fields, today: date, max_months: int = 12, future_days: int = 31) -> CheckResult:
    """처리일 기준으로 너무 오래되었거나 먼 미래인 날짜. 반증 전용."""
    d = to_date(x.get("date"))
    if d is None:
        return CheckResult("처리일 타당성", None, str(x.get("date")), ["date"])
    out_of_range = (today - d).days > max_months * 30.5 or (d - today).days > future_days
    return CheckResult("처리일 타당성", False if out_of_range else None,
                       f"{d} · 처리일 {today}" + (" · 범위 밖" if out_of_range else ""), ["date"])


def check_v3_date(x: Fields, v3_date: str | None) -> CheckResult:
    """날짜 크롭 재독과의 불일치. 반증 전용: 일치해도 둘 다 틀린 경우가 많다."""
    if v3_date is None or not x.get("date"):
        return CheckResult("날짜 크롭 재독", None, str(v3_date), ["date"])
    same = same_date(v3_date, x["date"])
    return CheckResult("날짜 크롭 재독", None if same else False, f"전체 {x['date']} vs 크롭 {v3_date}", ["date"])
