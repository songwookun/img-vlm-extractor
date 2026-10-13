"""VLM: 전체 추출(V0)과 합계·날짜 크롭 재독(V3)."""
import base64
import hashlib
import io
import json
import math
import re
import time
from collections.abc import Callable

import httpx
from PIL import Image

from app.config import settings
from app.engine.checks import Fields, item_count

# (이미지, 프롬프트, max_tokens) → 응답 본문 텍스트
Chat = Callable[[Image.Image, str, int], str]

PROMPT_V0 = """이 견적서 이미지에서 아래 JSON 을 추출하라. JSON 만 출력하라.
{"doc_no": "", "date": "", "client": "", "supplier": "", "supplier_brn": "", "manager": "",
 "items": [{"name": "", "qty": "", "unit_price": "", "amount": ""}],
 "supply": "", "vat": "", "total": "", "total_kor": ""}"""

PROMPT_V3 = {
    "total": """이 이미지 조각에서 '합계금액' 줄의 숫자만 읽어라. JSON 만 출력하라.
규칙: 보이는 글자 그대로, 계산하지 마라(다른 줄 숫자를 더하지 마라), 확실히 읽을 수 없으면 null.
{"total": ""}""",
    "date": """이 이미지 조각의 날짜를 읽어라. JSON 만 출력하라.
규칙: 보이는 글자 그대로, 추측하지 마라, 확실히 읽을 수 없으면 null.
{"date": ""}""",
}

V0_MAX_TOKENS = 1500
V3_MAX_TOKENS = 100

# 견적서 양식 위치 (900×1273 기준)
_TABLE_TOP, _ROW_H, _DEFAULT_ITEMS = 320, 36, 5
_DATE_BOX = (140, 150, 470, 185)

_HEADER_FIELDS = ("doc_no", "date", "client", "supplier", "supplier_brn", "manager",
                  "supply", "vat", "total", "total_kor")
_ITEM_KEYS = ("name", "qty", "unit_price", "amount")
_QTY_UNIT = re.compile(r"\s*([\d.,]+)\s*([가-힣A-Za-z]+)\s*")


class DailyLimit(Exception):
    """하루 한도. 재시도하지 않는다."""


class VlmError(RuntimeError):
    """재시도를 다 쓴 일시적 실패(5xx · 타임아웃)."""


class VlmRejected(VlmError):
    """이 요청만 거부됨(400 · 413 · 422) — 이미지가 너무 크거나 형식이 맞지 않는다."""


class VlmFatal(RuntimeError):
    """어떤 요청이든 같은 실패(키 · 권한 · 주소)."""


_MAX_WAIT = 600     # 이보다 긴 retry-after 는 사실상 하루 한도


def _redact(text: str) -> str:
    return re.sub(r"org_\w+", "org_***", text[:300])


def _retry_after(r: httpx.Response, default: float = 15) -> float:
    try:
        wait = float(r.headers.get("retry-after", default))
    except ValueError:              # HTTP-date 형식
        return default
    return wait if math.isfinite(wait) and wait >= 0 else default


class GroqClient:
    URL = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, api_key: str = settings.groq_api_key, model: str = settings.groq_model,
                 tries: int = 6, timeout: float = 120):
        if not api_key:
            raise VlmFatal("GROQ_API_KEY 가 없습니다. backend/.env 에 넣으세요.")
        self.model, self.tries = model, tries
        self.calls: list[dict] = []     # 성공한 호출의 프롬프트 해시 · 토큰 · 초
        self._http = httpx.Client(headers={"Authorization": f"Bearer {api_key}"}, timeout=timeout)

    def __call__(self, img: Image.Image, prompt: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_png_b64(img)}"}},
            ]}],
        }
        last, wait = None, 0.0
        for attempt in range(self.tries):
            if attempt:
                time.sleep(min(wait, _MAX_WAIT))
            start = time.monotonic()
            try:
                r = self._http.post(self.URL, json=body)
            except httpx.TransportError as e:       # 타임아웃 · 연결 끊김
                last, wait = e, 2 ** attempt
                continue
            if r.status_code == 200:
                try:
                    data = r.json()
                    text = data["choices"][0]["message"]["content"] or ""
                except (ValueError, KeyError, IndexError, TypeError):
                    last, wait = VlmError(f"200 형식 이상 {_redact(r.text)}"), 2 ** attempt
                    continue
                self.calls.append({"prompt": hashlib.md5(prompt.encode()).hexdigest()[:8],
                                   "tokens": (data.get("usage") or {}).get("total_tokens"),
                                   "seconds": round(time.monotonic() - start, 2)})
                return text
            if r.status_code == 429:
                wait = _retry_after(r)
                if "per day" in r.text or wait > _MAX_WAIT:
                    raise DailyLimit(_redact(r.text))
            elif r.status_code >= 500:
                wait = 2 ** attempt
            elif r.status_code in (400, 413, 422):
                raise VlmRejected(f"{r.status_code} {_redact(r.text)}")
            else:
                raise VlmFatal(f"{r.status_code} {_redact(r.text)}")
            last = VlmError(f"{r.status_code} {_redact(r.text)}")
        raise VlmError(f"재시도 {self.tries}회 초과") from last


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def parse_json(text: str) -> dict | None:
    """<think> 블록과 코드 펜스를 벗기고 첫 JSON 객체. 실패하면 None."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        d = json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


def split_qty(v):
    """'1 대' → ('1', '대'). 스키마에 단위 칸이 없어 VLM 이 수량에 붙여 쓴다."""
    m = _QTY_UNIT.fullmatch(v) if isinstance(v, str) else None
    return (m.group(1), m.group(2)) if m else (v, None)


def flatten(d: dict) -> Fields:
    """items 목록을 item0.name … 으로 편다. 빈 문자열과 null 은 둘 다 None."""
    x: Fields = {k: None if d.get(k) in ("", None) else str(d[k]) for k in _HEADER_FIELDS}
    for i, it in enumerate(d.get("items") or []):
        for k in _ITEM_KEYS:
            v = it.get(k) if isinstance(it, dict) else None
            v = None if v in ("", None) else str(v)
            x[f"item{i}.{k}"] = split_qty(v)[0] if k == "qty" else v
    return x


def extract(img: Image.Image, chat: Chat) -> Fields | None:
    """V0 전체 추출. 형식 위반이면 None."""
    d = parse_json(chat(img, PROMPT_V0, V0_MAX_TOKENS))
    return flatten(d) if d is not None else None


def crop_regions(x: Fields) -> dict[str, tuple[int, int, int, int]]:
    """합계·날짜 칸의 양식 위치. 합계 줄은 VLM 이 읽은 품목 수로 정한다(한 줄 틀려도 영역 안)."""
    top = _TABLE_TOP + _ROW_H * ((item_count(x) or _DEFAULT_ITEMS) + 1)
    return {"total": (540, top - 10, 860, top + 150), "date": _DATE_BOX}


def crop_up(img: Image.Image, box, scale: int = 2) -> Image.Image:
    c = img.crop(box)
    return c.resize((c.width * scale, c.height * scale), Image.LANCZOS)


def pad_to_ratio(img: Image.Image, min_h_ratio: float = 0.5) -> Image.Image:
    """위아래 흰 여백으로 가로:세로 2:1 까지. 가늘고 긴 조각은 토큰이 더 든다."""
    need_h = int(img.width * min_h_ratio)
    if img.height >= need_h:
        return img
    canvas = Image.new("RGB", (img.width, need_h), "white")
    canvas.paste(img, (0, (need_h - img.height) // 2))
    return canvas


def crops(img: Image.Image, x: Fields) -> dict[str, Image.Image]:
    out = {field: crop_up(img, box) for field, box in crop_regions(x).items()}
    out["date"] = pad_to_ratio(out["date"])
    return out


def reread(img: Image.Image, x: Fields, chat: Chat) -> dict[str, str | None]:
    """V3 크롭 재독. 파싱 실패와 null 은 None."""
    return {field: (parse_json(chat(c, PROMPT_V3[field], V3_MAX_TOKENS)) or {}).get(field)
            for field, c in crops(img, x).items()}
