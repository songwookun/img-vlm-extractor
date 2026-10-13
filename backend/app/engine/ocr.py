"""OCR: 글자와 위치를 읽고, VLM 값이 이미지에 실제로 있는지 찾는다."""
import re

import pytesseract
from PIL import Image

Token = dict          # {"text": str, "conf": float, "bbox": (x1, y1, x2, y2)}
BBox = tuple[int, int, int, int]

PSMS = (3, 6)         # 깨끗하면 3, 흐리면 6 이 낫다
LANG = "kor+eng"


def read_tokens(img: Image.Image, psm: int = 3, lang: str = LANG) -> list[Token]:
    d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    out = []
    for i, text in enumerate(d["text"]):
        if text.strip() and float(d["conf"][i]) >= 0:
            x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
            out.append({"text": text, "conf": float(d["conf"][i]), "bbox": (x, y, x + w, y + h)})
    return out


def check_ready(lang: str = LANG) -> None:
    """tesseract 와 언어 데이터가 있는가. VLM 호출 전에 확인한다."""
    missing = set(lang.split("+")) - set(pytesseract.get_languages())
    if missing:
        raise RuntimeError(f"tesseract 언어 데이터 없음: {', '.join(sorted(missing))}")


def read_all(img: Image.Image) -> list[list[Token]]:
    return [read_tokens(img, psm=p) for p in PSMS]


def norm(s: str, kind: str) -> str:
    """비교용 정규화. 숫자형은 구분자만 지우고 문자는 남긴다."""
    s = s.replace("|", "")
    return re.sub(r"[\s,.\-]", "", s) if kind == "num" else re.sub(r"\s", "", s)


def lines(tokens: list[Token], y_tol: int = 8) -> list[list[Token]]:
    """세로 중심이 가까운 토큰끼리 한 줄로. OCR 의 읽기 순서는 믿지 않는다."""
    rows: list[list[Token]] = []
    for t in sorted(tokens, key=lambda t: (t["bbox"][1] + t["bbox"][3]) / 2):
        cy = (t["bbox"][1] + t["bbox"][3]) / 2
        if rows and abs(rows[-1][0]["_cy"] - cy) <= y_tol:
            rows[-1].append({**t, "_cy": cy})
        else:
            rows.append([{**t, "_cy": cy}])
    return [sorted(r, key=lambda t: t["bbox"][0]) for r in rows]


def ground(value: str | None, kind: str, token_sets: list[list[Token]]) -> BBox | None:
    """값이 OCR 토큰에 있으면 그 위치(쪼개진 토큰은 이어 붙여 찾는다), 없으면 None."""
    target = norm(value or "", kind)
    if not target:
        return None
    span = len(target)
    for tokens in token_sets:
        for line in lines(tokens):
            for i in range(len(line)):
                for j in range(i + 1, min(i + span, len(line)) + 1):
                    if norm("".join(t["text"] for t in line[i:j]), kind) == target:
                        bs = [t["bbox"] for t in line[i:j]]
                        return (min(b[0] for b in bs), min(b[1] for b in bs),
                                max(b[2] for b in bs), max(b[3] for b in bs))
    return None


def ground_by_mode(value: str | None, kind: str, token_sets: list[list[Token]]) -> tuple[BBox | None, str]:
    """읽기 모드마다 따로 찾는다. 한쪽에서만 찾히면 "모드 불일치"."""
    found = [b for b in (ground(value, kind, [tokens]) for tokens in token_sets) if b]
    if not found:
        return None, "없음"
    return found[0], "일치" if len(found) == len(token_sets) else "모드 불일치"
