"""GroqClient 재시도 규칙 — 가짜 HTTP 응답으로. 실제 대기 없이."""
import httpx
import pytest
from PIL import Image

from app.engine import vlm

IMG = Image.new("RGB", (8, 8), "white")
OK = {"choices": [{"message": {"content": "{}"}}], "usage": {"total_tokens": 7}}


@pytest.fixture
def sleeps(monkeypatch):
    slept = []
    monkeypatch.setattr(vlm.time, "sleep", slept.append)
    return slept


def client_with(*responses):
    """응답을 차례로 돌려주는 클라이언트. 요청 수는 sent 로 센다"""
    queue, sent = list(responses), []

    def handler(request):
        sent.append(request)
        r = queue.pop(0)
        return r if isinstance(r, httpx.Response) else r(request)

    c = vlm.GroqClient(api_key="test")
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c, sent


def test_success_records_usage(sleeps):
    c, _ = client_with(httpx.Response(200, json=OK))
    assert c(IMG, "p", 10) == "{}"
    assert c.calls[0]["tokens"] == 7 and not sleeps


def test_5xx_backs_off_then_gives_up(sleeps):
    c, sent = client_with(*[httpx.Response(503, text="busy")] * 6)
    with pytest.raises(vlm.VlmError):
        c(IMG, "p", 10)
    assert len(sent) == 6 and sleeps == [1, 2, 4, 8, 16]               # 마지막 시도 뒤엔 안 기다린다


def test_timeout_is_retried(sleeps):
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)
    c, sent = client_with(timeout, httpx.Response(200, json=OK))
    assert c(IMG, "p", 10) == "{}" and len(sent) == 2


@pytest.mark.parametrize("response", [
    httpx.Response(429, text="Rate limit ... tokens per day (TPD) org_abc123"),
    httpx.Response(429, headers={"retry-after": "3600"}, text="slow down"),
])
def test_daily_limit_is_not_retried(sleeps, response):
    c, sent = client_with(response)
    with pytest.raises(vlm.DailyLimit) as e:
        c(IMG, "p", 10)
    assert len(sent) == 1 and "org_abc123" not in str(e.value)


@pytest.mark.parametrize("retry_after, waited", [("3", 3.0), ("Wed, 21 Oct 2026 07:28:00 GMT", 15), ("-5", 15), ("nan", 15)])
def test_per_minute_limit_waits(sleeps, retry_after, waited):
    c, _ = client_with(httpx.Response(429, headers={"retry-after": retry_after}, text="per minute"), httpx.Response(200, json=OK))
    assert c(IMG, "p", 10) == "{}" and sleeps == [waited]


@pytest.mark.parametrize("status, error", [(401, vlm.VlmFatal), (413, vlm.VlmRejected), (400, vlm.VlmRejected)])
def test_client_errors_are_not_retried(sleeps, status, error):
    c, sent = client_with(httpx.Response(status, text="no"))
    with pytest.raises(error):
        c(IMG, "p", 10)
    assert len(sent) == 1


@pytest.mark.parametrize("bad", [httpx.Response(200, text="<html>"), httpx.Response(200, json={"error": "x"})])
def test_malformed_200_is_retried(sleeps, bad):
    c, sent = client_with(bad, httpx.Response(200, json=OK))
    assert c(IMG, "p", 10) == "{}" and len(sent) == 2
