"""Shared constants for the time-share estimation pipeline."""

DAYS_PER_YEAR = 260
# From https://www.onetcenter.org/dictionary/30.1/excel/task_categories.html
# Maps an O*NET frequency bin (1-7, plus 0 = "Never") to an expected number of task
# instances per day. Overridden per-SOC when --use_cps_constants is set.
FREQ_TO_TIME_PER_DAY = {
    0: 0,
    1: 1/(DAYS_PER_YEAR*2),  # yearly or less
    2: 1/DAYS_PER_YEAR,  # more than yearly (this is expected number of days worked as assumed by ATUS)
    3: 1/20,  # more than monthly
    4: 1/5,  # more than weekly
    5: 1,  # daily
    6: 2,  # several times daily (computed as twice daily)
    7: 7  # hourly or more (computed as 7 times daily); average workday is 8 hours, but here we subtract an hour because this is spread out over O*NET tasks and doesn't account for non-work related acitivities during work (i.e., eating lunch, taking breaks, etc.)
}

# Hours worked per day, used as the LP daily-time budget.
# Doesn't account for time spent on tasks not in O*NET, such as meetings or eating lunch.
HOURS_PER_DAY = 7

TOTAL_HOURS_PER_DAY = 8

# assumes 8 hours a day, 260 days a year; notice that above we only account for 7 hours of work-related activities
HOURS_WORKED_PER_YEAR = 2080

HOURS_COVERED_PER_YEAR = HOURS_PER_DAY * DAYS_PER_YEAR

# If task i is more important than task k, then we assume the time spent on task i is at least IMPORTANCE_TOLERANCE hours greater than the time spent on task k (the epsilon margin in Equation 1).
IMPORTANCE_TOLERANCE = 1e-1

# minimum time per task that is marked as relevant to an occupation; this value is also used for the feasible-set exploration in validation/.
MIN_TIME_PER_TASK = IMPORTANCE_TOLERANCE
