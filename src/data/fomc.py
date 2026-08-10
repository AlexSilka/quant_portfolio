"""Scheduled FOMC announcement dates — the deterministic event calendar for the pre-FOMC drift.

Source: the Federal Reserve's own FOMC calendars (federalreserve.gov/monetarypolicy/
fomccalendars.htm + the per-year fomchistorical<YEAR>.htm archive). Each date below is the
*announcement day* — the second day of a two-day meeting (or the single day of a one-day
meeting), when the policy statement is released at ~14:00 ET. The pre-FOMC drift (Lucca-Moench
2015) is the abnormal equity return accruing in the ~24h *before* this timestamp, so the event
anchor is the announcement, and the tradable window is the day(s) leading into it.

Only *regularly scheduled* meetings are listed: the drift is an anticipation effect and needs the
meeting's timing to be known in advance, so unscheduled/emergency actions (2008 and 2020 conference
calls, inter-meeting cuts) are deliberately excluded. One caveat is stamped inline: in 2020 the
regularly scheduled Mar 17-18 meeting was pre-empted by the emergency Sunday-Mar-15 cut; the scheduled
Mar 18 anchor is kept for completeness (one event of ~170) and the driver's per-year / sub-period
splits surface any COVID-window distortion rather than hiding it.

The calendar runs 2005→ because that is where the ETF price history starts. Eight scheduled meetings
a year is a thin sample for anything measured per-event, so every year of overlap with the prices is
worth having: the 2005-2010 block is +48 events (+38%) and it is also the block the pre-FOMC-drift
literature was written on, which is what makes the post-2015 decay legible instead of ambiguous.
"""
from __future__ import annotations

import pandas as pd

# Announcement day (statement release, ~14:00 ET from 2013; ~14:15 ET before that — immaterial for
# daily bars, and the intraday tests only reach 2020). Verified against the Fed's per-year calendars.
FOMC_ANNOUNCEMENTS: list[str] = [
    # 2005  (equity ETF history starts 2005-01-03, so the calendar starts where the prices do)
    "2005-02-02", "2005-03-22", "2005-05-03", "2005-06-30", "2005-08-09", "2005-09-20",
    "2005-11-01", "2005-12-13",
    # 2006
    "2006-01-31", "2006-03-28", "2006-05-10", "2006-06-29", "2006-08-08", "2006-09-20",
    "2006-10-25", "2006-12-12",
    # 2007  (Aug 10, Aug 16 and Dec 6 were unscheduled conference calls — excluded)
    "2007-01-31", "2007-03-21", "2007-05-09", "2007-06-28", "2007-08-07", "2007-09-18",
    "2007-10-31", "2007-12-11",
    # 2008  (the Jan 22 and Oct 8 emergency cuts came out of conference calls — excluded)
    "2008-01-30", "2008-03-18", "2008-04-30", "2008-06-25", "2008-08-05", "2008-09-16",
    "2008-10-29", "2008-12-16",
    # 2009
    "2009-01-28", "2009-03-18", "2009-04-29", "2009-06-24", "2009-08-12", "2009-09-23",
    "2009-11-04", "2009-12-16",
    # 2010  (May 9 and Oct 15 were unscheduled conference calls — excluded)
    "2010-01-27", "2010-03-16", "2010-04-28", "2010-06-23", "2010-08-10", "2010-09-21",
    "2010-11-03", "2010-12-14",
    # 2011
    "2011-01-26", "2011-03-15", "2011-04-27", "2011-06-22", "2011-08-09", "2011-09-21",
    "2011-11-02", "2011-12-13",
    # 2012
    "2012-01-25", "2012-03-13", "2012-04-25", "2012-06-20", "2012-08-01", "2012-09-13",
    "2012-10-24", "2012-12-12",
    # 2013
    "2013-01-30", "2013-03-20", "2013-05-01", "2013-06-19", "2013-07-31", "2013-09-18",
    "2013-10-30", "2013-12-18",
    # 2014
    "2014-01-29", "2014-03-19", "2014-04-30", "2014-06-18", "2014-07-30", "2014-09-17",
    "2014-10-29", "2014-12-17",
    # 2015
    "2015-01-28", "2015-03-18", "2015-04-29", "2015-06-17", "2015-07-29", "2015-09-17",
    "2015-10-28", "2015-12-16",
    # 2016
    "2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21",
    "2016-11-02", "2016-12-14",
    # 2017
    "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20",
    "2017-11-01", "2017-12-13",
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01", "2018-09-26",
    "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31", "2019-09-18",
    "2019-10-30", "2019-12-11",
    # 2020  (Mar 18 was the scheduled anchor; pre-empted by the emergency Mar 15 cut — see module docstring)
    "2020-01-29", "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29", "2020-09-16",
    "2020-11-05", "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28", "2021-09-22",
    "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21",
    "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20",
    "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18",
    "2024-11-07", "2024-12-18",
    # 2025
    "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
    "2025-10-29", "2025-12-10",
    # 2026  (Sep 16, Oct 28, Dec 9 are future — beyond the data window, harmless)
    "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16",
    "2026-10-28", "2026-12-09",
]


def announce_days(tz: str | None = "UTC") -> pd.DatetimeIndex:
    """Announcement *dates* (midnight-stamped) as a DatetimeIndex, optionally tz-localised.

    For daily-bar studies this is all that is needed: the event is pinned to a calendar day and
    the tradable window is defined in trading-day offsets around it.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(FOMC_ANNOUNCEMENTS))
    return idx.tz_localize(tz) if tz else idx


def announce_timestamps_utc(hour_et: int = 14, minute_et: int = 0) -> pd.DatetimeIndex:
    """Announcement *timestamps* in UTC, localising 14:00 America/New_York per date (DST-correct).

    The precise Lucca-Moench window is the 24h ending at the ~14:00-ET statement release, so an
    intraday (or 24/7 crypto) test needs the exact instant. Eastern→UTC is +5h in winter (EST) and
    +4h in summer (EDT); localising to America/New_York and converting handles the switch per date,
    which a fixed UTC offset would get wrong on the ~half the calendar in the opposite season.
    """
    naive = pd.to_datetime(FOMC_ANNOUNCEMENTS) + pd.Timedelta(hours=hour_et, minutes=minute_et)
    return naive.tz_localize("America/New_York", nonexistent="shift_forward",
                             ambiguous=True).tz_convert("UTC")
