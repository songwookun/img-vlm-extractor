"""img-vlm 이미지폴더 -o out — 칸별 판정 JSON + review.csv. 다시 실행하면 이어서 한다.

종료 코드: 0 전부 처리 · 75 남은 문서 있음 · 1 키·설정 오류
"""
import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageOps

from app.engine import ocr, vlm
from app.engine.judge import FieldVerdict
from app.engine.pipeline import KST, Result, process
from app.engine.report import write_review

IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
CALLS_PER_DOC = 3
FORM_SIZE = (900, 1273)
EX_TEMPFAIL = 75


def open_image(path: Path) -> Image.Image:
    """EXIF 회전을 바로잡고, 투명 배경은 흰색 · 16비트는 8비트로."""
    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode.startswith("I"):
        img = img.convert("I").point(lambda v: v / 256).convert("L")
    if img.mode in ("RGBA", "LA", "P") or "transparency" in img.info:
        img = img.convert("RGBA")
        canvas = Image.new("RGB", img.size, "white")
        canvas.paste(img, mask=img.getchannel("A"))
        return canvas
    return img.convert("RGB")


def save_result(path: Path, result: Result, model: str, calls: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")      # 쓰다 끊겨도 반쪽 파일이 남지 않게
    tmp.write_text(json.dumps({
        "processed_at": datetime.now(KST).isoformat(timespec="seconds"),
        "model": model,
        "calls": calls,
        "auto_ok": result.auto_ok,
        "fields": result.fields,
        "reread": result.reread,
        "verdicts": [asdict(v) for v in result.verdicts],
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def load_result(path: Path) -> Result | None:
    """저장된 결과. 없거나 읽을 수 없으면 None — 다시 처리한다."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        verdicts = [FieldVerdict(**{**v, "bbox": tuple(v["bbox"]) if v["bbox"] else None}) for v in d["verdicts"]]
        return Result(d["fields"], d["reread"], verdicts)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _non_negative(s: str) -> int:
    n = int(s)
    if n < 0:
        raise argparse.ArgumentTypeError("0 이상")
    return n


def main(argv: list[str] | None = None, client=None, read_ocr=ocr.read_all, preflight: bool = True) -> int:
    ap = argparse.ArgumentParser(prog="img-vlm", description="견적서 이미지에서 값을 뽑고, 칸마다 자동처리/사람 검토를 판정한다.")
    ap.add_argument("folder", type=Path, help="견적서 이미지 폴더 (png · jpg)")
    ap.add_argument("-o", "--out", type=Path, default=Path("out"))
    ap.add_argument("--max-calls", type=_non_negative,
                    help=f"이번 실행의 VLM 호출 상한 — 문서당 {CALLS_PER_DOC}호출로 센다(재시도 요청은 세지 않음)")
    args = ap.parse_args(argv)
    if not args.folder.is_dir():
        ap.error(f"폴더가 없다: {args.folder}")

    images = sorted(p for p in args.folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_TYPES)
    if dup := [s for s, n in Counter(p.stem.casefold() for p in images).items() if n > 1]:
        ap.error(f"이름이 같은 이미지가 있다(대소문자 무시, 결과 파일이 겹친다): {', '.join(dup)}")
    results = args.out / "results"
    results.mkdir(parents=True, exist_ok=True)
    for tmp in results.glob("*.tmp"):
        tmp.unlink()
    result_path = {p: results / f"{p.stem}.json" for p in images}
    done = {p: r for p in images if (r := load_result(result_path[p])) is not None}
    todo = [p for p in images if p not in done]
    if args.max_calls is not None:
        todo = todo[: args.max_calls // CALLS_PER_DOC]
    print(f"이미지 {len(images)}장 · 이미 처리 {len(done)}장 · 이번에 {len(todo)}장")

    if todo:
        try:
            if preflight:
                ocr.check_ready()
            client = client or vlm.GroqClient()
        except (RuntimeError, OSError) as e:      # tesseract · 언어 데이터 · 키 없음
            print(f"■ 시작 못 함: {e}")
            return 1
    skipped = {}
    for i, path in enumerate(todo, 1):
        try:
            img = open_image(path)
        except (OSError, Image.DecompressionBombError) as e:
            skipped[path] = f"이미지를 열 수 없다 ({e})"
            continue
        if img.size != FORM_SIZE:
            print(f"  ! {path.name}: {img.size} — 양식 위치는 {FORM_SIZE} 기준이다. 크롭이 빗나가면 검토가 늘어난다")
        start = len(client.calls)
        try:
            result = process(img, client, read_ocr=read_ocr)
        except vlm.DailyLimit:
            print("■ 하루 한도 — 여기까지 저장했다. 같은 명령을 나중에 다시 실행하면 이어진다")
            break
        except vlm.VlmFatal as e:
            print(f"■ 다시 해도 같은 실패: {e}")
            return 1
        except vlm.VlmRejected as e:
            skipped[path] = f"VLM 이 요청을 거부 ({e})"
            continue
        except vlm.VlmError as e:                 # 다음 실행에 다시 시도
            print(f"  실패 {path.name}: {e}")
            continue
        save_result(result_path[path], result, client.model, client.calls[start:])
        done[path] = result
        state = "형식 위반 → 전체 검토" if result.fields is None else "자동" if result.auto_ok else f"검토 {len(result.review)}칸"
        print(f"  {i}/{len(todo)} {path.name} · {state}")

    csv_path = write_review(((p.stem, open_image(p) if r.review else None, r) for p, r in sorted(done.items())), args.out)
    cells = [v for r in done.values() for v in r.verdicts]
    review = sum(v.verdict == "review" for v in cells)
    print(f"\n문서 {len(done)}/{len(images)} · 자동 {len(cells) - review}칸 · 검토 {review}칸 · "
          f"전부 자동 {sum(r.auto_ok for r in done.values())}장 · 형식 위반 {sum(r.fields is None for r in done.values())}장")
    print(f"검토표 {csv_path}")
    for path, why in skipped.items():
        print(f"  사람이 직접 볼 것 {path.name}: {why}")
    left = len(images) - len(done) - len(skipped)
    if left:
        hint = f" (--max-calls 는 {CALLS_PER_DOC} 이상이어야 한 장이 처리된다)" if args.max_calls is not None and args.max_calls < CALLS_PER_DOC else ""
        print(f"남은 {left}장 — 같은 명령을 다시 실행하면 이어진다{hint}")
    return EX_TEMPFAIL if left else 0


if __name__ == "__main__":
    sys.exit(main())
