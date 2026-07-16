"""
Translation cleaner module for Buddy Core.

This module provides post-processing cleanup and correction for Korean translations,
ensuring publication-ready quality.
"""

import re


class TranslationCleaner:
    @staticmethod
    def normalize_numbers(text: str) -> str:
        """
        Convert English/literal currency notations ($12.5B, 35M) to proper Korean units.
        E.g.:
          $12.5B / 12.5B 달러 / 12.5십억 달러 -> 125억 달러
          $35M / 35M 달러 -> 3,500만 달러
          $350M -> 3억 5,000만 달러
        """
        if not text:
            return ""

        # Billion pattern: captures optionally prefixed $, value, unit, and optional "달러/달러화"
        billion_pattern = re.compile(
            r"(?:\$\s*)?(\d+(?:\.\d+)?)\s*(?:Billion|billion|B|십억)(?![a-zA-Z])\s*(?:달러|달러화)?",
            re.IGNORECASE,
        )

        def convert_billion(match: re.Match) -> str:
            val_str = match.group(1)
            val = float(val_str)
            billion_val = val * 10.0  # 1 Billion = 10억

            if billion_val.is_integer():
                return f"{int(billion_val)}억 달러"
            else:
                ok_val = int(billion_val)
                rem_val = round((billion_val - ok_val) * 10000)
                if ok_val > 0:
                    return f"{ok_val}억 {rem_val:,}만 달러"
                else:
                    return f"{rem_val:,}만 달러"

        # Million pattern
        million_pattern = re.compile(
            r"(?:\$\s*)?(\d+(?:\.\d+)?)\s*(?:Million|million|M|백만)(?![a-zA-Z])\s*(?:달러|달러화)?",
            re.IGNORECASE,
        )

        def convert_million(match: re.Match) -> str:
            val_str = match.group(1)
            val = float(val_str)
            man_val = val * 100.0  # 1 Million = 100만
            man_val_int = round(man_val)

            if man_val_int >= 10000:
                ok_val = man_val_int // 10000
                rem_val = man_val_int % 10000
                if rem_val > 0:
                    return f"{ok_val}억 {rem_val:,}만 달러"
                else:
                    return f"{ok_val}억 달러"
            else:
                return f"{man_val_int:,}만 달러"

        text = billion_pattern.sub(convert_billion, text)
        text = million_pattern.sub(convert_million, text)
        return text

    @staticmethod
    def remove_duplicates(text: str) -> str:
        """
        Clean up duplicate phrases and repeated characters at word boundaries.
        E.g.:
          "제품을 제품을" -> "제품을"
          "완만한한" -> "완만한"
          "지출출" -> "지출"
        """
        if not text:
            return ""

        # 1. Clean duplicated words separated by spaces (e.g. "제품을 제품을")
        text = re.sub(r"\b(\w+)(?:\s+\1)+\b", r"\1", text)

        # 2. Clean duplicated syllable at the end of word of length >= 3 (e.g. "완만한한" -> "완만한")
        # Prevents matching 2-syllable valid words like "매매", "부부", "지지"
        text = re.sub(r"\b([가-힣]+)([가-힣])\2\b", r"\1\2", text)

        return text

    @classmethod
    def clean(cls, text: str) -> str:
        """Apply all filters sequentially to sanitize the translation."""
        if not text:
            return ""

        # Normalize numbers
        text = cls.normalize_numbers(text)

        # Remove duplicates
        text = cls.remove_duplicates(text)

        # Normalize HTML line breaks
        text = re.sub(r"<\s*br\s*/?\s*>", "<br />", text, flags=re.IGNORECASE)

        return text
