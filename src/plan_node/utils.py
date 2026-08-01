# utils.py
import numpy as np
import math

BUCKETS = [
    "Severe Underestimation",
    "Slight Underestimation",
    "Approximately Accurate",
    "Slight Overestimation",
    "Severe Overestimation",
]
bucket2id = {b: i for i, b in enumerate(BUCKETS)}
id2bucket = {i: b for b, i in bucket2id.items()}


def qerror_and_bucket(est_f, act_f, t1 = 0.01, t2 = 0.5, t3 = 2.0, t4 = 10.0):
    """Return (bucket_name, qerror_float) using user's bucket logic."""
    est = max(est_f, 1e-9)
    act = max(act_f, 1e-9)
    q = est / act
    if q <= t1:
        return BUCKETS[0], q
    elif q <= t2:
        return BUCKETS[1], q
    elif q <= t3:
        return BUCKETS[2], q
    elif q <= t4:
        return BUCKETS[3], q
    else:
        return BUCKETS[4], q
