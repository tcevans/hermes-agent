"""Vertex Gemini usage_metadata → OpenAI-shaped usage for session accounting."""

import base64
from types import SimpleNamespace

from agent.usage_pricing import normalize_usage
from agent.vertex_gemini import (
    _extra_content_to_thought_signature_bytes,
    _openai_usage_from_gemini_response,
    _thought_signature_to_extra_content,
)


def test_openai_usage_from_gemini_basic():
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            candidates_token_count=42,
        )
    )
    u = _openai_usage_from_gemini_response(resp)
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 42
    c = normalize_usage(u, provider="vertex-gemini", api_mode="gemini_generate")
    assert c.input_tokens == 100
    assert c.output_tokens == 42


def test_openai_usage_from_gemini_with_cached():
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=80,
            candidates_token_count=10,
            cached_content_token_count=20,
        )
    )
    u = _openai_usage_from_gemini_response(resp)
    assert u.completion_tokens == 10
    assert u.prompt_tokens == 100
    assert u.prompt_tokens_details.cached_tokens == 20
    c = normalize_usage(u, provider="vertex-gemini", api_mode="gemini_generate")
    assert c.cache_read_tokens == 20
    assert c.input_tokens == 80


def test_openai_usage_from_gemini_thoughts():
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=50,
            candidates_token_count=12,
            thoughts_token_count=8,
        )
    )
    u = _openai_usage_from_gemini_response(resp)
    assert u.prompt_tokens == 50
    assert u.completion_tokens == 20


def test_thought_signature_extra_content_roundtrip():
    raw = b"\x00\xffopaque"
    extra = _thought_signature_to_extra_content(raw)
    assert extra == {"google": {"thought_signature": "AP/ib3BhcmU="}}
    assert _extra_content_to_thought_signature_bytes(extra) == raw


def test_thought_signature_flat_extra_content():
    """Support extra_content shape used by some adapters/tests."""
    extra = {"thought_signature": base64.b64encode(b"abc").decode("ascii")}
    assert _extra_content_to_thought_signature_bytes(extra) == b"abc"


def test_openai_usage_from_gemini_total_fallback():
    resp = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=5,
            total_token_count=105,
        )
    )
    u = _openai_usage_from_gemini_response(resp)
    assert u.prompt_tokens == 100
    assert u.completion_tokens == 5
