"""사람용 출력: 검토 칸만 review.csv 로, 칸마다 빨간 박스 조각 이미지."""
import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw

from app.engine.checks import Fields, item_count
from app.engine.ocr import BBox
from app.engine.pipeline import Result

COLUMNS = ["doc_id", "field", "value", "reasons", "bbox", "bbox_source", "crop_path"]

# 견적서 양식 위치 (900×1273 기준). 품목·합계 줄은 VLM 이 읽은 품목 수로 계산한다
_HEADER_BOXES = {
    "doc_no": (150, 128, 300, 148), "date": (150, 158, 300, 178),
    "client": (60, 212, 460, 240), "supplier_brn": (570, 126, 760, 148),
    "supplier": (570, 171, 760, 193), "manager": (570, 216, 760, 238),
}
_ITEM_COLUMNS = {"name": (55, 438), "qty": (445, 512), "unit_price": (580, 712), "amount": (720, 846)}
_TABLE_TOP, _ROW_H, _DEFAULT_ITEMS = 320, 36, 5
_TOTALS = ("supply", "vat", "total")


def field_box(name: str, x: Fields) -> BBox | None:
    """그 칸이 양식에서 있어야 할 자리."""
    if name in _HEADER_BOXES:
        return _HEADER_BOXES[name]
    top = _TABLE_TOP + _ROW_H * ((item_count(x) or _DEFAULT_ITEMS) + 1)
    if name in _TOTALS:
        y = top + 20 + 38 * _TOTALS.index(name)
        return (550, y - 4, 846, y + 26)
    if name == "total_kor":
        return (55, top + 16, 540, top + 42)
    m = re.fullmatch(r"item(\d+)\.(\w+)", name)
    if m and m.group(2) in _ITEM_COLUMNS:
        x1, x2 = _ITEM_COLUMNS[m.group(2)]
        y = _TABLE_TOP + _ROW_H * (int(m.group(1)) + 1)
        return (x1, y + 4, x2, y + _ROW_H - 4)
    return None


def save_crop(img: Image.Image, bbox: BBox, path: Path, pad: int = 30) -> None:
    """라벨·옆 칸까지 보이게 주변을 잘라 빨간 박스를 그린다."""
    x1, y1, x2, y2 = bbox
    box = (max(0, x1 - pad), max(0, y1 - pad), min(img.width, x2 + pad), min(img.height, y2 + pad))
    c = img.crop(box).convert("RGB")
    ImageDraw.Draw(c).rectangle((x1 - box[0], y1 - box[1], x2 - box[0], y2 - box[1]), outline=(220, 0, 0), width=2)
    c.save(path)


def write_review(docs: list[tuple[str, Image.Image, Result]], out_dir: Path) -> Path:
    """검토 칸만 쓴다. 위치는 OCR 근거가 있으면 그 자리, 없으면 양식 위치."""
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for doc_id, img, result in docs:
        if result.fields is None:
            rows.append({"doc_id": doc_id, "field": "", "value": "", "reasons": "VLM 출력 형식 위반",
                         "bbox": "", "bbox_source": "", "crop_path": ""})
            continue
        for v in result.review:
            bbox, source = (v.bbox, "OCR") if v.bbox else (field_box(v.field, result.fields), "양식")
            crop = ""
            if bbox:
                crop = str(crop_dir / f"{doc_id}_{v.field}.png")
                save_crop(img, bbox, Path(crop))
            rows.append({"doc_id": doc_id, "field": v.field, "value": v.value or "", "reasons": " | ".join(v.reasons),
                         "bbox": bbox or "", "bbox_source": source if bbox else "", "crop_path": crop})
    path = out_dir / "review.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as f:   # 엑셀에서 한글이 깨지지 않게
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path
