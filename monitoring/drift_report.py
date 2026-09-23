"""
Generates a data/prediction drift report comparing a "reference" window
of logged predictions (e.g. the first week after deploy) against the
"current" window (e.g. the last 7 days).

Run manually or on a schedule (cron / GitHub Actions scheduled workflow):
    python monitoring/drift_report.py --db monitoring/predictions.db --out monitoring/report.html
"""

import argparse
import sqlite3

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset



