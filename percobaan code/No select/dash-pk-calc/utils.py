# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:48:55 2019

@author: shday
"""

import math
from collections import namedtuple
import pandas as pd
import numpy as np
import datetime 

PKParams = namedtuple(
    "PKParams",
    "t_half, rate_const, auc0_t, auc0_inf,"
    "percent_extrap, c_max, t_max, term_slope, term_inter ",
)


def calc_pk(x, y, iv_calc=False, term_points=3):
    xy = list(zip(x, y))
    xy.sort()
    if xy[0][0] > 0 and not iv_calc:
        xy.insert(0, (0, 0))

    x, y = list(zip(*xy))

    c_max = max(y)
    t_max = x[y.index(max(y))]
    auc0_t = np.trapz(y, x)

    try:
        slope, inter = np.polyfit(
            x[-term_points:], [math.log(i) for i in y[-term_points:]], deg=1
        )
    except ValueError:
        return PKParams(None, None, auc0_t, None, None, c_max, t_max, None, None)

    rate_const = -slope

    t1_2 = math.log(2) / rate_const

    auc0_inf = auc0_t + y[-1] / rate_const

    percent_extrap = 100 * (y[-1] / rate_const) / auc0_inf

    return PKParams(
        t1_2, rate_const, auc0_t, auc0_inf, percent_extrap, c_max, t_max, slope, inter
    )


def pkdata2dt(df):
    pivoted = df.pivot(index="Date", values="Close", columns="subject_index")
    todict = pivoted.to_dict("index")

    records = []
    for r in pivoted.index:
        record = todict[r]
        record[pivoted.index.name] = r
        records.append(record)

    return records


def dt2pkdata(dt):
    keys = list(dt[0].keys())
    keys.remove("Date")

    records = []
    for subject in keys:
        for rec in dt:
            try:
                rec["Date"] = pd.to_datetime(rec["Date"])
                records.append(
                    {
                        "Date": rec["Date"],
                        "subject_index": int(subject),
                        "Close": float(rec[subject]),
                    }
                )
            except (ValueError, KeyError):
                continue

    return pd.DataFrame.from_records(records)


def test_calcpk():
    pkdata1 = pd.DataFrame(
        {
            "subject_index": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "Date": [datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1)],
            "Close": [14100, 14100, 14100, 14100, 14100, 14100, 14100, 14100],
        }
    )

    pkdata2 = pd.DataFrame(
        {
            "subject_index": [0, 0, 0, 0, 0, 0, 0, 0],
            "Date": [datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1)],
            "Close": [14100, 14100, 14100, 14100, 14100, 14100, 14100, 14100],
        }
    )

    p1 = calc_pk(pkdata1["Date"], pkdata1["Close"])
    p2 = calc_pk(pkdata2["Date"], pkdata2["Close"])

    assert p1 == p2


def test_pkdata2dt():
    pkdata = pd.DataFrame(
        {
            "subject_index": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
            "Date": [datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1), datetime.date(2016-7-1)],
            "Close": [14100, 14100, 14100, 14100, 14100, 14100, 14100, 14100],
        }
    )

    dt = pkdata2dt(pkdata)

    assert dt == [
        {0: 1.0, 1: 1.0, "Date": datetime.date(2016-7-1)},
        {0: 3.0, 1: 3.2, "Date": datetime.date(2016-7-1)},
        {0: 5.0, 1: 5.1, "Date": datetime.date(2016-7-1)},
        {0: 4.0, 1: 4.1, "Date": datetime.date(2016-7-1)},
        {0: 2.0, 1: 2.2, "Date": datetime.date(2016-7-1)},
        {0: 1.0, 1: 1.0, "Date": datetime.date(2016-7-1)},
        {0: 0.5, 1: 0.55, "Date": datetime.date(2016-7-1)},
        {0: 0.25, 1: 0.3, "Date": datetime.date(2016-7-1)},
    ]
