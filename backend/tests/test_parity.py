"""노트북(02_role_split 9c)의 확정 판정과 파이프라인 판정이 칸 단위로 같은가. API 호출 없음."""
import hashlib
import io
import pickle
from dataclasses import asdict
from pathlib import Path

import pytest
from PIL import Image

from app.engine.pipeline import process

PARITY = Path(__file__).parents[2] / "notebooks" / "data" / "parity_dev.pkl"

pytestmark = pytest.mark.skipif(not PARITY.exists(), reason="노트북 9c 셀을 먼저 실행")


def _md5(b: bytes) -> str:
    return hashlib.md5(b).hexdigest()


@pytest.fixture(scope="module")
def parity():
    return pickle.loads(PARITY.read_bytes())


def _cached_chat(responses):
    def chat(img, prompt, max_tokens):
        return responses[(_md5(img.tobytes()), _md5(prompt.encode()))]
    return chat


def test_matches_notebook(parity):
    chat = _cached_chat(parity["responses"])
    mismatched = []
    for doc in parity["docs"]:
        img = Image.open(io.BytesIO(doc["png"]))
        result = process(img, chat, read_ocr=lambda _, tokens=doc["ocr"]: tokens, today=parity["today"])
        assert result.fields == doc["fields"], doc["doc_id"]
        assert result.reread == doc["reread"], doc["doc_id"]
        got = [asdict(v) for v in result.verdicts]
        mismatched += [(doc["doc_id"], e["field"]) for e, g in zip(doc["verdicts"], got) if e != g]
        assert len(got) == len(doc["verdicts"]), doc["doc_id"]
    assert not mismatched, mismatched[:10]
