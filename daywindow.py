"""
Shared "coupon day" window: a daily report for target_date covers matches
kicking off from 08:00 local that day through 07:59:59 local the next
morning — not the calendar midnight boundary. This matches how a daily
betting coupon is actually organized (a "day" runs morning-to-morning) and
keeps late-night fixtures (e.g. US/South America matches that land after
midnight Athens time) grouped with the evening's coupon instead of being
split off into the next calendar day.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Athens")
REPORT_WINDOW_START_HOUR = 8


def get_report_window(target_date):
    """Returns (window_start, window_end) as tz-aware Europe/Athens
    datetimes: window_start is 08:00 on target_date, window_end is 08:00
    the following day (exclusive)."""
    start = datetime(
        target_date.year, target_date.month, target_date.day,
        REPORT_WINDOW_START_HOUR, 0, 0, tzinfo=LOCAL_TZ,
    )
    end = start + timedelta(days=1)
    return start, end
