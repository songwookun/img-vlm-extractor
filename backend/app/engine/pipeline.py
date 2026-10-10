"""견적서 한 장: VLM 추출 → 크롭 재독 → OCR → 검사·가드 → 칸별 판정."""
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from PIL import Image

from app.engine import ocr, vlm
from app.engine.checks import Fields, check_quote, check_recent, check_v3_date
from app.engine.guards import guard_checks
from app.engine.judge import FieldVerdict, judge

KST = ZoneInfo("Asia/Seoul")
ReadOcr = Callable[[Image.Image], list[list[ocr.Token]]]


@dataclass
class Result:
    fields: Fields | None                  # None = VLM 출력 형식 위반
    reread: dict[str, str | None] = field(default_factory=dict)
    verdicts: list[FieldVerdict] = field(default_factory=list)

    @property
    def auto_ok(self) -> bool:
        return self.fields is not None and all(v.verdict == "auto" for v in self.verdicts)

    @property
    def review(self) -> list[FieldVerdict]:
        return [v for v in self.verdicts if v.verdict == "review"]


def process(img: Image.Image, chat: vlm.Chat, read_ocr: ReadOcr = ocr.read_all,
            today: date | None = None) -> Result:
    """VLM 호출 3번(전체 1 + 크롭 2). 하루 한도(vlm.DailyLimit)는 호출한 쪽으로 올린다."""
    x = vlm.extract(img, chat)
    if x is None:
        return Result(None)
    v3 = vlm.reread(img, x, chat)
    token_sets = read_ocr(img)
    checks = (guard_checks(x, check_quote(x), token_sets, v3_total=v3["total"])
              + [check_recent(x, today or datetime.now(KST).date()), check_v3_date(x, v3["date"])])
    return Result(x, v3, judge(x, checks, token_sets))
