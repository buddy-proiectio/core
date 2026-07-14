import pytest
import re
import os
import yaml
import tempfile
import datetime as dt_module
from core.src.formatter import inject_frontmatter, get_korean_weekday, run_formatter

def test_get_korean_weekday():
    # 2026-07-14 is Tuesday (화요일)
    assert get_korean_weekday("20260714") == "화"

def test_get_korean_weekday_invalid_fallback():
    # Verify that an invalid date returns a valid weekday character rather than crashing
    weekday = get_korean_weekday("20269999")
    assert weekday in ["월", "화", "수", "목", "금", "토", "일"]

def test_inject_frontmatter_ko_alpha():
    content = "## Hello World"
    result = inject_frontmatter(content, "20260714", "alpha_signal", "ko")
    assert 'title: "2026.07.14. (화) Alpha Signal"' in result
    assert "category: alpha_signal" in result
    assert "lang: ko" in result
    assert result.endswith(content)

def test_inject_frontmatter_timezone_and_yaml():
    content = "## Test Content"
    result = inject_frontmatter(content, "20260714", "alpha_signal", "ko")
    
    # Verify it starts with the YAML frontmatter boundary
    assert result.startswith("---\n")
    parts = result.split("---\n")
    assert len(parts) >= 3
    yaml_content = parts[1]
    
    # Verify valid YAML parsing
    parsed = yaml.safe_load(yaml_content)
    assert parsed["title"] == "2026.07.14. (화) Alpha Signal"
    assert parsed["category"] == "alpha_signal"
    assert parsed["lang"] == "ko"
    
    # Verify timezone calculations: check KST / UTC+09:00 offset on the parsed datetime object
    date_val = parsed["date"]
    assert isinstance(date_val, dt_module.datetime)
    assert date_val.tzinfo is not None
    assert date_val.tzinfo.utcoffset(date_val) == dt_module.timedelta(hours=9)
    
    # Verify the raw string output format matches the KST ISO format exactly
    pattern = r"date: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00"
    assert re.search(pattern, yaml_content) is not None

def test_run_formatter_invalid_date_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, "alpha_signal_20269999.md") # Invalid date in filename
        output_file = os.path.join(tmpdir, "formatted_20269999.md")
        
        # Write dummy content with an invalid header date that will be handled safely
        with open(input_file, "w", encoding="utf-8") as f:
            f.write("## 20269999\nGood day.\nThis is a report.")
            
        # This must run successfully and fallback to today's date KST instead of crashing
        success = run_formatter(input_file, output_file, lang="en")
        assert success is True
        
        # Check that output file exists and parses as valid YAML frontmatter
        assert os.path.exists(output_file)
        with open(output_file, "r", encoding="utf-8") as f:
            output_content = f.read()
            
        assert output_content.startswith("---\n")
        parts = output_content.split("---\n")
        yaml_content = parts[1]
        
        parsed = yaml.safe_load(yaml_content)
        assert "title" in parsed
        assert parsed["category"] == "alpha_signal"
        
        # Verify timezone standard is preserved in fallback
        date_val = parsed["date"]
        assert isinstance(date_val, dt_module.datetime)
        assert date_val.tzinfo is not None
        assert date_val.tzinfo.utcoffset(date_val) == dt_module.timedelta(hours=9)
        
        pattern = r"date: \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+09:00"
        assert re.search(pattern, yaml_content) is not None
