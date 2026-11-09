"""
Tests for timeutils.py.

The key dates in 2026 (checked against Python's time-zone database):
  * US clocks go forward  Sun 8 Mar 2026;  UK/EU go forward Sun 29 Mar 2026.
  * UK/EU clocks go back  Sun 25 Oct 2026; US goes back    Sun 1 Nov 2026.
So 9-28 Mar and 25-31 Oct 2026 are the "out of step" weeks, when London and
New York are 4 hours apart instead of 5 (Section 6.3).

Run with:  python -m pytest
"""

from datetime import date, datetime, time, timedelta, timezone

import pytest

import config
import timeutils as tu

UTC = timezone.utc

# Representative days for each season.
WINTER = date(2026, 1, 14)          # everyone on standard time
SUMMER = date(2026, 7, 15)          # everyone on daylight saving
SPRING_GAP = date(2026, 3, 16)      # US on DST, UK/EU not yet
AUTUMN_GAP = date(2026, 10, 28)     # UK/EU back on standard, US still on DST


# --------------------------------------------------------------------------
# Section 6.4 table: market events in UTC and MUT, winter vs summer
# --------------------------------------------------------------------------
# (event, local time, zone, UTC winter, MUT winter, UTC summer, MUT summer)
SECTION_6_4 = [
    ("Tokyo open",        time(9, 0),   config.TZ_TOKYO,     "00:00", "04:00", "00:00", "04:00"),
    ("London open",       time(8, 0),   config.TZ_LONDON,    "08:00", "12:00", "07:00", "11:00"),
    ("Frankfurt open",    time(9, 0),   config.TZ_FRANKFURT, "08:00", "12:00", "07:00", "11:00"),
    ("New York open",     time(9, 30),  config.TZ_NEW_YORK,  "14:30", "18:30", "13:30", "17:30"),
    ("New York close",    time(16, 0),  config.TZ_NEW_YORK,  "21:00", "01:00", "20:00", "00:00"),
    ("Forex rollover",    time(17, 0),  config.TZ_NEW_YORK,  "22:00", "02:00", "21:00", "01:00"),
]


@pytest.mark.parametrize("event,local_t,tz,utc_w,mut_w,utc_s,mut_s", SECTION_6_4)
def test_section_6_4_table(event, local_t, tz, utc_w, mut_w, utc_s, mut_s):
    for day, exp_utc, exp_mut in [(WINTER, utc_w, mut_w), (SUMMER, utc_s, mut_s)]:
        utc = tu.local_to_utc(day, local_t, tz)
        assert utc.strftime("%H:%M") == exp_utc, f"{event} UTC on {day}"
        assert tu.to_mut(utc).strftime("%H:%M") == exp_mut, f"{event} MUT on {day}"


# --------------------------------------------------------------------------
# Out-of-step weeks
# --------------------------------------------------------------------------
def _london_ny_gap_hours(day: date) -> float:
    """Hours between the same wall-clock time in London and in New York."""
    london = tu.local_to_utc(day, time(12, 0), config.TZ_LONDON)
    new_york = tu.local_to_utc(day, time(12, 0), config.TZ_NEW_YORK)
    return (new_york - london).total_seconds() / 3600


@pytest.mark.parametrize(
    "day,expected_gap",
    [
        (WINTER, 5),
        (SUMMER, 5),
        (SPRING_GAP, 4),
        (AUTUMN_GAP, 4),
        (date(2026, 3, 9), 4),    # first weekday after US change
        (date(2026, 3, 27), 4),   # last weekday before UK change
        (date(2026, 3, 30), 5),   # first weekday after UK change
        (date(2026, 10, 26), 4),  # first weekday after UK change back
        (date(2026, 11, 2), 5),   # first weekday after US change back
    ],
)
def test_london_new_york_gap(day, expected_gap):
    assert _london_ny_gap_hours(day) == expected_gap


def test_spring_gap_week_opens():
    # London still on GMT -> London open 08:00 UTC; New York already on EDT -> 13:30 UTC.
    assert tu.local_to_utc(SPRING_GAP, time(8, 0), config.TZ_LONDON).strftime("%H:%M") == "08:00"
    assert tu.local_to_utc(SPRING_GAP, time(9, 30), config.TZ_NEW_YORK).strftime("%H:%M") == "13:30"
    # Frankfurt follows the EU change (29 Mar), so still 08:00 UTC.
    assert tu.local_to_utc(SPRING_GAP, time(9, 0), config.TZ_FRANKFURT).strftime("%H:%M") == "08:00"


def test_autumn_gap_week_opens():
    # London back on GMT -> 08:00 UTC; New York still on EDT -> 13:30 UTC.
    assert tu.local_to_utc(AUTUMN_GAP, time(8, 0), config.TZ_LONDON).strftime("%H:%M") == "08:00"
    assert tu.local_to_utc(AUTUMN_GAP, time(9, 30), config.TZ_NEW_YORK).strftime("%H:%M") == "13:30"


def test_market_local_hour_follows_daylight_saving():
    # The same UTC hour is a different London hour in winter and summer.
    winter = datetime(2026, 1, 14, 8, 0, tzinfo=UTC)
    summer = datetime(2026, 7, 15, 8, 0, tzinfo=UTC)
    assert tu.to_market_local(winter, config.TZ_LONDON).hour == 8
    assert tu.to_market_local(summer, config.TZ_LONDON).hour == 9


# --------------------------------------------------------------------------
# Mauritius time and display
# --------------------------------------------------------------------------
@pytest.mark.parametrize("day", [WINTER, SUMMER, SPRING_GAP, AUTUMN_GAP])
def test_mauritius_is_always_utc_plus_4(day):
    ts = datetime.combine(day, time(12, 0), tzinfo=UTC)
    assert tu.to_mut(ts).utcoffset() == timedelta(hours=4)


def test_fmt_dual_trade_card_style():
    ts = datetime(2026, 11, 9, 8, 45, tzinfo=UTC)
    assert tu.fmt_dual(ts) == "08:45 UTC (12:45 MUT)"


def test_fmt_dual_across_midnight_shows_both_dates():
    ts = datetime(2026, 11, 9, 22, 30, tzinfo=UTC)
    assert tu.fmt_dual(ts, with_date=True) == "2026-11-09 22:30 UTC (2026-11-10 02:30 MUT)"


def test_fmt_dual_accepts_non_utc_aware_input():
    # 12:45 in Mauritius is the same instant as 08:45 UTC.
    ts = datetime(2026, 11, 9, 12, 45, tzinfo=tu.MUT)
    assert tu.fmt_dual(ts) == "08:45 UTC (12:45 MUT)"


def test_naive_datetime_is_rejected():
    with pytest.raises(ValueError):
        tu.ensure_utc(datetime(2026, 11, 9, 8, 45))


def test_from_epoch_ms():
    assert tu.from_epoch_ms(1767225600000) == datetime(2026, 1, 1, tzinfo=UTC)
    assert tu.from_epoch_ms("1767225600000") == datetime(2026, 1, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# Trading day: resets at 17:00 New York
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "ts,expected_start,expected_label",
    [
        # Winter: 17:00 NY = 22:00 UTC.
        (datetime(2026, 1, 14, 21, 59, tzinfo=UTC), datetime(2026, 1, 13, 22, 0, tzinfo=UTC), date(2026, 1, 14)),
        (datetime(2026, 1, 14, 22, 0, tzinfo=UTC),  datetime(2026, 1, 14, 22, 0, tzinfo=UTC), date(2026, 1, 15)),
        # Summer: 17:00 NY = 21:00 UTC.
        (datetime(2026, 7, 15, 20, 59, tzinfo=UTC), datetime(2026, 7, 14, 21, 0, tzinfo=UTC), date(2026, 7, 15)),
        (datetime(2026, 7, 15, 21, 0, tzinfo=UTC),  datetime(2026, 7, 15, 21, 0, tzinfo=UTC), date(2026, 7, 16)),
        # Spring out-of-step week: follows New York (EDT), not London.
        (datetime(2026, 3, 16, 21, 0, tzinfo=UTC),  datetime(2026, 3, 16, 21, 0, tzinfo=UTC), date(2026, 3, 17)),
        # Autumn out-of-step week: New York still on EDT.
        (datetime(2026, 10, 28, 21, 30, tzinfo=UTC), datetime(2026, 10, 28, 21, 0, tzinfo=UTC), date(2026, 10, 29)),
        # Day of the US change (Sun 8 Mar 2026): reset is at 21:00 UTC that evening.
        (datetime(2026, 3, 8, 21, 30, tzinfo=UTC),  datetime(2026, 3, 8, 21, 0, tzinfo=UTC), date(2026, 3, 9)),
        # Morning in Mauritius belongs to the trading day that started the previous evening.
        (datetime(2026, 11, 9, 4, 0, tzinfo=UTC),   datetime(2026, 11, 8, 22, 0, tzinfo=UTC), date(2026, 11, 9)),
    ],
)
def test_trading_day(ts, expected_start, expected_label):
    assert tu.trading_day_start_utc(ts) == expected_start
    assert tu.trading_day_label(ts) == expected_label
