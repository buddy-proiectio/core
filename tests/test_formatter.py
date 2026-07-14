import pytest
from core.src.formatter import inject_frontmatter, get_korean_weekday

def test_get_korean_weekday():
    # 2026-07-14 is Tuesday (화요일)
    assert get_korean_weekday("20260714") == "화"

def test_inject_frontmatter_ko_alpha():
    content = "## Hello World"
    result = inject_frontmatter(content, "20260714", "alpha_signal", "ko")
    assert "title: 2026.07.14. (화) Alpha Signal" in result
    assert "category: alpha_signal" in result
    assert "lang: ko" in result
    assert result.endswith(content)
