"""
The Chief Translator (English to Korean)

Translates English text to professional Korean using litellm and ollama/translategemma.
Implements a robust retry logic for transient errors.
"""

import os
import re
import sys
import glob
import logging
import argparse
import translators as ts

LOG_FILE = "logs/translator.log"

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.shared_logger import setup_logger

logger = setup_logger(LOG_FILE, __name__)

# Suppress noisy INFO logs from external libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)


def chunk_text(text: str, limit: int = 4000) -> list[str]:
    """Simple text chunker to ensure no piece exceeds the limit."""
    chunks = []
    while len(text) > limit:
        split_index = text.rfind(" ", 0, limit)
        if split_index == -1:
            split_index = limit

        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()

    if text:
        chunks.append(text)
    return chunks


def make_polite(text: str) -> str:
    """
    Post-process Korean translation to enforce polite/formal endings (~입니다, ~합니다).
    DeepL API currently does not support the 'formality' parameter for Korean.
    """
    if not text:
        return text

    # Negative lookahead: only replace if NOT followed by another Korean character.
    # This safely prevents modifying connectors like ~한다고, ~한다면, ~했다며, etc.
    suffix_pattern = r"(?![가-힣ㄱ-ㅎㅏ-ㅣ])"

    # 1. Noun exceptions and explicit special cases
    explicit_replacements = [
        (r"수치다", "수치입니다"),
        (r"결과다", "결과입니다"),
        (r"목표다", "목표입니다"),
        (r"규모다", "규모입니다"),
        (r"추세다", "추세입니다"),
        (r"상태다", "상태입니다"),
        (r"이유다", "이유입니다"),
        (r"예상치다", "예상치입니다"),
        (r"전망치다", "전망치입니다"),
        (r"최고치다", "최고치입니다"),
        (r"최저치다", "최저치입니다"),
        (r"기록이다", "기록입니다"),
        (r"상황이다", "상황입니다"),
        (r"수준이다", "수준입니다"),
        (r"예정이다", "예정입니다"),
        (r"것이다", "것입니다"),
        (r"전망이다", "전망입니다"),
        (r"예상이다", "예상입니다"),
        (r"예측이다", "예측입니다"),
        (r"중이다", "중입니다"),
        (r"때문이다", "때문입니다"),
        (r"이다", "입니다"),
        (r"아니다", "아닙니다"),
    ]
    for old, new in explicit_replacements:
        text = re.sub(old + suffix_pattern, new, text)

    # 2. Convert ~는다 to ~습니다 (for consonant-ending present tense verbs like 받는다)
    text = re.sub(r"([가-힣])는다" + suffix_pattern, r"\1습니다", text)

    # 3. Present tense, Adjectives, and common verbs
    replacements = [
        (r"한다", "합니다"),
        (r"하다", "합니다"),
        (r"된다", "됩니다"),
        (r"되다", "됩니다"),
        (r"있다", "있습니다"),
        (r"없다", "없습니다"),
        (r"않다", "않습니다"),
        (r"크다", "큽니다"),
        (r"많다", "많습니다"),
        (r"적다", "적습니다"),
        (r"높다", "높습니다"),
        (r"낮다", "낮습니다"),
        (r"같다", "같습니다"),
        (r"다르다", "다릅니다"),
        (r"작다", "작습니다"),
        (r"가깝다", "가깝습니다"),
        (r"멀다", "멉니다"),
        (r"빠르다", "빠릅니다"),
        (r"느리다", "느립니다"),
        (r"어렵다", "어렵습니다"),
        (r"쉽다", "쉽습니다"),
        (r"어려워진다", "어려워집니다"),
        (r"쉬워진다", "쉬워집니다"),
        (r"좋다", "좋습니다"),
        (r"나쁘다", "나쁩니다"),
        (r"강하다", "강합니다"),
        (r"약하다", "약합니다"),
        (r"새롭다", "새롭습니다"),
        (r"필요하다", "필요합니다"),
        (r"중요하다", "중요합니다"),
        (r"가능하다", "가능합니다"),
        (r"불가능하다", "불가능합니다"),
        (r"심하다", "심합니다"),
        (r"비슷하다", "비슷합니다"),
        (r"충분하다", "충분합니다"),
        (r"다양하다", "다양합니다"),
        (r"유사하다", "유사합니다"),
        (r"겠다", "겠습니다"),
        (r"진다", "집니다"),
        (r"시킨다", "시킵니다"),
        (r"나온다", "나옵니다"),
        (r"보인다", "보입니다"),
        (r"준다", "줍니다"),
        (r"나타난다", "나타납니다"),
        (r"간다", "갑니다"),
        (r"온다", "옵니다"),
        (r"늘어난다", "늘어납니다"),
        (r"가져온다", "가져옵니다"),
        (r"줄어든다", "줄어듭니다"),
        (r"떨어진다", "떨어집니다"),
        (r"오른다", "오릅니다"),
        (r"이어진다", "이어집니다"),
        (r"커진다", "커집니다"),
        (r"작아진다", "작아집니다"),
        (r"강조한다", "강조합니다"),
        (r"성장한다", "성장합니다"),
        (r"하락한다", "하락합니다"),
        (r"증가한다", "증가합니다"),
        (r"감소한다", "감소합니다"),
        (r"상승한다", "상승합니다"),
    ]
    for old, new in replacements:
        text = re.sub(old + suffix_pattern, new, text)

    # 4. Programmatic replacer for Past Tense (any Syllable with ㅆ 받침 + 다)
    # This safely handles 했, 었, 았, 였, 났, 겼, 혔, 샀, 컸, 줬, 갔, 왔, 등등
    def ssang_sios_replacer(match):
        char = match.group(1)
        if "가" <= char <= "힣":
            char_code = ord(char) - 0xAC00
            if char_code % 28 == 20:  # 20 is the index for ㅆ 받침
                return char + "습니다"
        return match.group(0)

    text = re.sub(r"([가-힣])다" + suffix_pattern, ssang_sios_replacer, text)

    return text


def contains_english(t: str) -> bool:
    return bool(re.search(r"[a-zA-Z]", t))


def contains_korean(t: str) -> bool:
    return bool(re.search(r"[가-힣]", t))


def mask_numbers(text: str) -> tuple[str, list[str]]:
    """
    Masks large numbers with commas (e.g. 65,000, 185,000) using placeholders
    to protect them from translation engine distortion. Keeps dollar signs outside.
    """
    if not text:
        return text, []
    # Matches comma-separated digits (at least one comma)
    pattern = r"[0-9]{1,3}(?:,[0-9]{3})+"
    placeholders = []

    def repl(match):
        placeholder = f"__NUM_{len(placeholders)}__"
        placeholders.append(match.group(0))
        return placeholder

    masked_text = re.sub(pattern, repl, text)
    return masked_text, placeholders


def unmask_numbers(text: str, placeholders: list[str]) -> str:
    """Restores original masked numbers from placeholders."""
    if not text or not placeholders:
        return text
    for i, original in enumerate(placeholders):
        placeholder = f"__NUM_{i}__"
        text = text.replace(placeholder, original)
    return text


# If I add a new company name or CEO name, I need to update the entity_maps dictionary.
def unify_entities(text: str) -> str:
    """Standardizes company and CEO names to prevent inconsistency."""
    if not text:
        return text

    entity_maps = {
        # Companies
        r"(?<![a-zA-Z0-9])(?:Nvidia|NVIDIA|NVIDA)(?![a-zA-Z0-9])|엔비디아|엔비다": "엔비디아",
        r"(?<![a-zA-Z0-9])(?:Tesla|TSLA)(?![a-zA-Z0-9])|테슬라": "테슬라",
        r"(?<![a-zA-Z0-9])Apple(?![a-zA-Z0-9])|애플": "애플",
        r"(?<![a-zA-Z0-9])(?:Microsoft|MSFT)(?![a-zA-Z0-9])|마이크로소프트|(?<![a-zA-Z0-9])MS(?![a-zA-Z0-9])": "마이크로소프트",
        r"(?<![a-zA-Z0-9])(?:Alphabet|Google|GOOGL?)(?![a-zA-Z0-9])|구글": "구글",
        r"(?<![a-zA-Z0-9])(?:Meta Platforms|Meta)(?![a-zA-Z0-9])|메타": "메타",
        r"(?<![a-zA-Z0-9])(?:Amazon|AMZN)(?![a-zA-Z0-9])|아마존": "아마존",
        r"(?<![a-zA-Z0-9])OpenAI(?![a-zA-Z0-9])|오픈AI|오픈에이아이": "오픈AI",
        r"(?<![a-zA-Z0-9])Anthropic(?![a-zA-Z0-9])|앤트로픽|앤쓰로픽": "앤트로픽",
        r"(?<![a-zA-Z0-9])AMD(?![a-zA-Z0-9])|에이엠디": "AMD",
        r"(?<![a-zA-Z0-9])Intel(?![a-zA-Z0-9])|인텔": "인텔",
        r"(?<![a-zA-Z0-9])TSMC(?![a-zA-Z0-9])|티에스엠씨": "TSMC",
        r"(?<![a-zA-Z0-9])(?:Super Micro Computer|Supermicro|SMCI)(?![a-zA-Z0-9])|슈퍼마이크로컴퓨터|슈퍼마이크로": "슈퍼마이크로컴퓨터",
        r"(?<![a-zA-Z0-9])Broadcom(?![a-zA-Z0-9])|브로드컴": "브로드컴",
        r"(?<![a-zA-Z0-9])(?:Arm Holdings|Arm|ARM)(?![a-zA-Z0-9])": "ARM",
        r"(?<![a-zA-Z0-9])ASML(?![a-zA-Z0-9])|에이エス엠엘": "ASML",
        r"(?<![a-zA-Z0-9])Oracle(?![a-zA-Z0-9])|오라클": "오라클",
        r"(?<![a-zA-Z0-9])Palantir(?![a-zA-Z0-9])|팔란티어": "팔란티어",
        r"(?<![a-zA-Z0-9])(?:Rocket Lab|RocketLab)(?![a-zA-Z0-9])|로켓랩": "로켓랩",
        r"(?<![a-zA-Z0-9])(?:AST SpaceMobile|AST Space Mobile)(?![a-zA-Z0-9])|AST\s*스페이스모바일": "AST 스페이스모바일",
        r"(?<![a-zA-Z0-9])Intuitive Machines(?![a-zA-Z0-9])|인튜이티브\s*머신스": "인튜이티브 머신스",
        r"(?<![a-zA-Z0-9])Lockheed Martin(?![a-zA-Z0-9])|록히드\s*마틴": "록히드 마틴",
        r"(?<![a-zA-Z0-9])Northrop Grumman(?![a-zA-Z0-9])|노스롭\s*그루먼": "노스롭 그루먼",
        r"(?<![a-zA-Z0-9])L3Harris(?![a-zA-Z0-9])|L3해리스": "L3해리스",
        r"(?<![a-zA-Z0-9])EchoStar(?![a-zA-Z0-9])|에코스타": "에코스타",
        r"(?<![a-zA-Z0-9])State Street(?![a-zA-Z0-9])|스테이트\s*스트리트": "스테이트 스트리트",
        r"(?<![a-zA-Z0-9])Vanguard(?![a-zA-Z0-9])|뱅가드": "뱅가드",
        r"(?<![a-zA-Z0-9])(?:JPMorgan|J\.P\.\s*Morgan|JP\s*Morgan)(?![a-zA-Z0-9])|JP모건": "JP모건",
        r"(?<![a-zA-Z0-9])(?:Bank of America|BofA|BAC)(?![a-zA-Z0-9])|뱅크\s*오브\s*아메리카|뱅크오브아메리카": "뱅크오브아메리카",
        r"(?<![a-zA-Z0-9])Wells Fargo(?![a-zA-Z0-9])|웰스\s*파고|웰스파고": "웰스파고",
        r"(?<![a-zA-Z0-9])Deutsche Bank(?![a-zA-Z0-9])|도이치\s*뱅크|도이치뱅크": "도이치뱅크",
        r"(?<![a-zA-Z0-9])Salesforce(?![a-zA-Z0-9])|세일즈포스": "세일즈포스",
        r"(?<![a-zA-Z0-9])Snowflake(?![a-zA-Z0-9])|스노우플레이크": "스노우플레이크",
        r"(?<![a-zA-Z0-9])HPQ?(?![a-zA-Z0-9])|에이치피": "HP",
        r"(?<![a-zA-Z0-9])(?:Micron Technology|Micron)(?![a-zA-Z0-9])|마이크론": "마이크론",
        r"(?<![a-zA-Z0-9])Marvell(?![a-zA-Z0-9])|마벨|마벨\s*테크놀로지": "마벨",
        r"(?<![a-zA-Z0-9])Coinbase(?![a-zA-Z0-9])|코인베이스": "코인베이스",
        r"(?<![a-zA-Z0-9])Circle(?![a-zA-Z0-9])|써클|서클": "써클",
        r"(?<![a-zA-Z0-9])(?:MicroStrategy|MSTR)(?![a-zA-Z0-9])|마이크로스트레티지": "마이크로스트레티지",
        r"(?<![a-zA-Z0-9])Roundhill(?![a-zA-Z0-9])|라운드힐": "라운드힐",
        # CEOs / People
        r"Jensen Hwang|젠슨\s*황": "젠슨 황",
        r"Elon Musk|일론\s*머스크": "일론 머스크",
        r"Tim Cook|팀\s*쿡": "팀 쿡",
        r"Satya Nadella|사티아\s*나델라": "사티아 나델라",
        r"Sundar Pichai|순다르\s*피차이": "순다르 피차이",
        r"Mark Zuckerberg|마크\s*저커버그|마크\s*주커버그": "마크 저커버그",
        r"Andy Jassy|앤디\s*재시|앤디\s*제시": "앤디 재시",
        r"Sam Altman|샘\s*올트먼|샘\s*알트만|샘\s*알트먼": "샘 올트먼",
        r"Dario Amodei|다리오\s*아모데이": "다리오 아모데이",
        r"Lisa Su|리사\s*수": "리사 수",
        r"Pat Gelsinger|팻\s*겔싱어|패트릭\s*겔싱어": "팻 겔싱어",
        r"C\.\s*C\.\s*Wei|C\s*C\s*Wei|CC\s*Wei|CC\s*웨이|CC웨이": "CC 웨이",
        r"Charles Liang|찰스\s*리앙|찰스\s*량": "찰스 리앙",
        r"Hock Tan|혹\s*탄": "혹 탄",
        r"Rene Haas|르네\s*하스": "르네 하스",
        r"Christophe Fouquet|크리스토프\s*푸케": "크리스토프 푸케",
        r"Larry Ellison|래리\s*앨리슨": "래리 앨리슨",
        r"Alex Karp|알렉스\s*카프": "알렉스 카프",
        r"Peter Beck|피터\s*벡": "피터 벡",
        r"Chamath Palihapitiya|차마스\s*팔리하피티야": "차마스 팔리하피티야",
        r"Cathie Wood|캐시\s*우드": "캐시 우드",
        r"Bill Ackman|빌\s*애크먼": "빌 애크먼",
        r"Warren Buffett|워런\s*버핏": "워런 버핏",
        r"Greg Abel|그렉\s*아벨": "그렉 아벨",
        r"Alexei Gogolev|알렉시\s*고골레프": "알렉시 고골레프",
        r"Bob Eddy|밥\s*에디": "밥 에디",
        r"Amin Nasser|아민\s*나세르": "아민 나세르",
        r"Bernie Sanders|버니\s*샌더스": "버니 샌더스",
        r"Kevin Warsh|케빈\s*워시": "케빈 워시",
        r"Jerome Powell|제롬\s*파월": "제롬 파월",
        r"Donald Trump|도널드\s*트럼프": "도널드 트럼프",
    }

    for pattern, standard in entity_maps.items():
        text = re.sub(pattern, standard, text, flags=re.IGNORECASE)

    return text


def resolve_korean_particles(text: str) -> str:
    """
    Grammatically resolves double Korean postpositions (particles) like 을(를) or 은(는)
    based on the preceding word ending (consonant vs vowel) for Sino-Korean numbers,
    English acronyms, and Hangul.
    """
    if not text:
        return text

    pattern = r"([a-zA-Z0-9$%,.%가-힣]+)\s*(은\(는\)|는\(은\)|이\(가\)|가\(이\)|을\(를\)|를\(을\)|와\(과\)|과\(와\)|으\(로\)|로\(으\))"

    def check_has_padchim_and_rieul(word_str: str) -> tuple[bool, bool]:
        cleaned = word_str.strip()

        # If it has a dollar sign anywhere, it's read as '달러' -> no padchim
        if "$" in cleaned or "\\$" in cleaned:
            return False, False

        # If it ends with %, read as '퍼센트'/'프로' -> no padchim
        if cleaned.endswith("%"):
            return False, False

        numeric_cleaned = re.sub(r"[$,]", "", cleaned)

        if re.match(r"^\d+(?:\.\d+)?$", numeric_cleaned):
            if "." in numeric_cleaned:
                last_digit = numeric_cleaned[-1]
                has_padchim = last_digit in ["0", "1", "3", "6", "7", "8", "9"]
                is_rieul = last_digit in ["1", "7", "8"]
                return has_padchim, is_rieul
            else:
                if numeric_cleaned == "0":
                    return True, False

                num_str = numeric_cleaned
                trailing_zeros = len(num_str) - len(num_str.rstrip("0"))

                if trailing_zeros == 0:
                    last_digit = num_str[-1]
                    has_padchim = last_digit in ["0", "1", "3", "6", "7", "8", "9"]
                    is_rieul = last_digit in ["1", "7", "8"]
                    return has_padchim, is_rieul
                else:
                    m = trailing_zeros
                    if 12 <= m <= 15:
                        return False, False
                    else:
                        return True, False
        else:
            # Check if the last character is Hangul
            last_char = cleaned[-1]
            if "가" <= last_char <= "힣":
                char_code = ord(last_char) - 0xAC00
                has_padchim = (char_code % 28) != 0
                is_rieul = (char_code % 28) == 8
                return has_padchim, is_rieul

            # English word or acronym
            word = numeric_cleaned.lower()
            if word.endswith("l") or word.endswith("le"):
                return True, True
            if (
                word.endswith("m")
                or word.endswith("me")
                or word.endswith("n")
                or word.endswith("ne")
                or word.endswith("ng")
            ):
                return True, False

            return False, False

    def repl(match):
        word_part = match.group(1)
        particle_part = match.group(2)

        has_padchim, is_rieul = check_has_padchim_and_rieul(word_part)

        if particle_part in ["은(는)", "는(은)"]:
            return word_part + ("은" if has_padchim else "는")
        elif particle_part in ["이(가)", "가(이)"]:
            return word_part + ("이" if has_padchim else "가")
        elif particle_part in ["을(를)", "를(을)"]:
            return word_part + ("을" if has_padchim else "를")
        elif particle_part in ["와(과)", "과(와)"]:
            return word_part + ("과" if has_padchim else "와")
        elif particle_part in ["으(로)", "로(으)"]:
            if has_padchim and not is_rieul:
                return word_part + "으로"
            else:
                return word_part + "로"
        return match.group(0)

    return re.sub(pattern, repl, text)


def translate_text_with_retry(text: str) -> str:
    import time
    import random

    proxies = {"http": "socks5://127.0.0.1:9050", "https": "socks5://127.0.0.1:9050"}

    # Mask large numbers to protect them from distortion by the translation engine
    masked_text, placeholders = mask_numbers(text)

    for attempt in range(15):
        method = attempt % 3
        try:
            if method == 0:
                # Google with Tor
                translated = ts.translate_text(
                    masked_text,
                    from_language="en",
                    to_language="ko",
                    translator="google",
                    proxies=proxies,
                    timeout=10,
                )
            elif method == 1:
                # Google direct (no proxy)
                translated = ts.translate_text(
                    masked_text,
                    from_language="en",
                    to_language="ko",
                    translator="google",
                    timeout=10,
                )
            else:
                # Bing direct (no proxy)
                translated = ts.translate_text(
                    masked_text,
                    from_language="en",
                    to_language="ko",
                    translator="bing",
                    timeout=10,
                )

            translated_str = str(translated).strip()

            # If successful, check if it actually translated to Korean or contains masked placeholders
            if translated_str and (
                contains_korean(translated_str) or "__NUM_" in translated_str
            ):
                # Restore original numbers safely
                restored_text = unmask_numbers(translated_str, placeholders)
                # Unify companies & CEOs
                unified_text = unify_entities(restored_text)
                # Resolve double particles in final Korean text
                resolved_text = resolve_korean_particles(unified_text)
                return make_polite(resolved_text)

        except Exception as e:
            logger.warning(f"Translation attempt {attempt+1} failed: {e}")

        # Exponential backoff with jitter
        sleep_time = 0.5 * (2 ** (attempt % 3)) + random.uniform(0.1, 0.3)
        time.sleep(sleep_time)

    # Absolute fallback: return original text but log error
    logger.error(f"All 15 translation attempts failed for: '{text}'")
    return text


def split_into_sentences(text: str) -> list[str]:
    """
    Splits text into sentences using simple heuristics (delimiters followed by spaces or newlines),
    while preserving spacing and punctuation.
    """
    # Split by period, question mark, or exclamation followed by space/newline, or raw newlines.
    # The capture group keeps the delimiters so we can stitch them back.
    tokens = re.split(r"(\. |\? |\! |\n)", text)
    sentences = []
    current = ""
    for token in tokens:
        if token in [". ", "? ", "! ", "\n"]:
            current += token
            sentences.append(current)
            current = ""
        else:
            current += token
    if current:
        sentences.append(current)
    return [s for s in sentences if s]


def translate_large_text(text: str, max_chunk_len: int = 1200) -> str:
    """
    Splits a large text block into sentence-aware chunks and translates them individually
    to avoid API payload limit errors.
    """
    if len(text) <= max_chunk_len:
        return translate_text_with_retry(text)

    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_len = 0

    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len > max_chunk_len:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_len = 0

            # If a single sentence exceeds the limit, force character-based split
            if sent_len > max_chunk_len:
                sub_text = sentence
                while len(sub_text) > max_chunk_len:
                    # Find last space in the limit
                    split_idx = sub_text.rfind(" ", 0, max_chunk_len)
                    if split_idx == -1:
                        split_idx = max_chunk_len
                    chunks.append(sub_text[:split_idx])
                    sub_text = sub_text[split_idx:].lstrip()
                if sub_text:
                    current_chunk.append(sub_text)
                    current_len = len(sub_text)
            else:
                current_chunk.append(sentence)
                current_len = sent_len
        else:
            current_chunk.append(sentence)
            current_len += sent_len

    if current_chunk:
        chunks.append("".join(current_chunk))

    translated_chunks = []
    for chunk in chunks:
        # Avoid redundant translations for pure formatting
        if contains_english(chunk):
            translated_chunks.append(translate_text_with_retry(chunk))
        else:
            translated_chunks.append(chunk)

    return "".join(translated_chunks)


def translate_text(text: str) -> str:
    if not text.strip():
        return text

    # If the text has no english remaining, return it directly!
    if not contains_english(text):
        return text

    return translate_large_text(text)


def process_line(line: str) -> str:
    """
    Process a single line of markdown text, handling structural symbols and links constraints.
    """
    # 1. Blank lines preserved as blank lines without calling LLM
    if not line.strip():
        return line

    # Strip newline at the right to reconstruct it safely later
    original_endswith_newline = line.endswith("\n")
    line_content = line.rstrip("\n")

    # Check if the line begins with any of our structural symbols (ignoring leading whitespace)
    stripped_line = line_content.lstrip()

    # --- Hard-Tuning (1:1 Matching) ---

    # 1) Exact matches (Headers)
    exact_mappings = {
        "### Daily Point": "### Daily Point",
        "**Topline Signals**": "**Topline Signals**",
        "### Weekly Schedule": "### 주간 일정",
        "### General": "### 경제 일반",
        "### Bitcoin": "### 비트코인",
        "### Semiconductor": "### 반도체",
        "### AI": "### AI",
        "### Bio": "### 바이오",
        "### Aerospace": "### 우주항공",
        "### Software": "### 소프트웨어",
        "### Others": "### 기타",
    }
    if stripped_line in exact_mappings:
        return line_content.replace(stripped_line, exact_mappings[stripped_line]) + (
            "\n" if original_endswith_newline else ""
        )

    # 1.5) Markdown Tables (Do not translate)
    if (
        stripped_line.startswith("|")
        and stripped_line.endswith("|")
        and "|" in stripped_line[1:-1]
    ):
        return line_content + ("\n" if original_endswith_newline else "")

    # 2) Daily Point Indicators (Do not translate Dow Jones, Nasdaq, S&P 500, Bitcoin)
    if stripped_line.startswith("_ "):
        if any(
            indicator in stripped_line
            for indicator in [
                "Dow Jones",
                "S&P 500",
                "Nasdaq",
                "Bitcoin",
            ]
        ):
            return line_content + ("\n" if original_endswith_newline else "")

    # 3) Dates: Report Header (e.g., ## 20260224) - Keep as-is for the formatter to handle
    m_report = re.match(r"^##\s+(\d{4})(\d{2})(\d{2})$", stripped_line)
    if m_report:
        return line_content + ("\n" if original_endswith_newline else "")

    # 2. Check for structural symbols
    structural_symbols = ["## ", "### ", "* ", "★ ", "_ "]
    prefix = ""
    target_text = line_content

    for sym in structural_symbols:
        if stripped_line.startswith(sym):
            # Calculate where the symbol ends to split the line
            prefix_len = len(line_content) - len(stripped_line) + len(sym)
            prefix = line_content[:prefix_len]
            target_text = line_content[prefix_len:]
            break

    # If the rest of the text is empty, no need to query LLM
    if not target_text.strip():
        # Re-attach the newline if it existed
        return prefix + ("\n" if original_endswith_newline else "")

    # 3. Check for Markdown Links
    pattern = r"\[(.*?)\]\((.*?)\)"

    # If there are links, we process them carefully without destroying markdown
    if re.search(pattern, target_text):
        parts = []
        last_idx = 0
        for match in re.finditer(pattern, target_text):
            # Text before the link
            before = target_text[last_idx : match.start()]
            if before.strip():
                parts.append(translate_text(before))
            else:
                parts.append(before)

            # Extract title and url
            title = match.group(1)
            url = match.group(2)

            # Send ONLY the title to translate_text (skip for SEC filings)
            if re.search(r"\b(8-K|10-K|10-Q|SEC Filing)\b", title, re.IGNORECASE):
                translated_title = title
            else:
                translated_title = translate_text(title)

            # Reassemble strictly using Python (without <br /> formatting)
            parts.append(f"[{translated_title}]({url})")

            last_idx = match.end()

        # Text after the last link
        after = target_text[last_idx:]
        if after.strip():
            parts.append(translate_text(after))
        else:
            parts.append(after)

        translated_text = "".join(parts)
    else:
        # 4. For all other plain text
        translated_text = translate_text(target_text)

    # Reattach symbol and newline
    final_line = prefix + translated_text

    final_line += "\n" if original_endswith_newline else ""
    return final_line


def run_translator(report_type: str = "full"):
    workspace_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(workspace_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    if report_type == "premarket":
        search_pattern = os.path.join(data_dir, "premarket_report_*.txt")
    else:
        search_pattern = os.path.join(data_dir, "final_report_*.txt")

    files = glob.glob(search_pattern)

    if not files:
        logger.error(f"No input files matching {search_pattern} found.")
        sys.exit(1)

    # Process the most recent file
    files.sort(reverse=True)
    input_file = files[0]
    filename = os.path.basename(input_file)

    if report_type == "premarket":
        date_part = filename.replace("premarket_report_", "").replace(".txt", "")
        output_filename = f"premarket_report_ko_draft_{date_part}.txt"
    else:
        date_part = filename.replace("final_report_", "").replace(".txt", "")
        output_filename = f"final_report_ko_draft_{date_part}.txt"
    output_file = os.path.join(data_dir, output_filename)

    logger.info(f"Starting translation process...")
    logger.info(f"Input: {input_file}")
    logger.info(f"Output Draft: {output_file}")

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    translated_lines = []
    in_weekly_schedule = False

    for i, line in enumerate(lines):
        try:
            stripped = line.strip()

            # If we hit another section header, exit weekly schedule skip mode
            if stripped.startswith("### ") and stripped != "### Weekly Schedule":
                in_weekly_schedule = False

            if in_weekly_schedule:
                # Bypass translation entirely for weekly schedule contents
                translated_line = line
            else:
                translated_line = process_line(line)

            # If we just hit the Weekly Schedule header, enter skip mode
            if stripped == "### Weekly Schedule":
                in_weekly_schedule = True

        except KeyboardInterrupt:
            logger.warning("Translation process interrupted. Exiting gracefully.")
            return False
        except Exception as e:
            logger.error(
                f"Failed to translate line {i+1}: '{line.strip()}'. Keeping original. Error: {e}"
            )
            translated_line = line

        translated_lines.append(translated_line)

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(translated_lines)

    logger.info(f"Translation complete. Saved draft to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translator Script")
    parser.add_argument(
        "--type",
        choices=["full", "premarket"],
        default="full",
        help="Type of report to translate",
    )
    args = parser.parse_args()
    run_translator(report_type=args.type)
