from datetime import datetime
import pytz


def parse_utc_time(utc_str: str) -> datetime:
    """
    Parses a UTC ISO format time string into a timezone-aware datetime object (UTC).
    Used commonly across both English generation (cio) and Korean formatting (formatter).
    """
    for fmt in (
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(utc_str, fmt)
            if not dt.tzinfo:
                dt = pytz.UTC.localize(dt)
            return dt
        except ValueError:
            pass
    if "+" in utc_str:
        try:
            dt = datetime.fromisoformat(utc_str)
            if not dt.tzinfo:
                dt = pytz.UTC.localize(dt)
            return dt
        except ValueError:
            pass
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        if not dt.tzinfo:
            dt = pytz.UTC.localize(dt)
        return dt
    except ValueError:
        pass
    return datetime.now(pytz.UTC)
