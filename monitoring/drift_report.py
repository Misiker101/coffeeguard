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


def load_predictions(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY ts", conn)
    conn.close()
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="monitoring/predictions.db")
    parser.add_argument("--out", default="monitoring/report.html")
    parser.add_argument("--split", type=float, default=0.3,
                         help="Fraction of earliest rows used as reference window")
    args = parser.parse_args()

    df = load_predictions(args.db)
    if len(df) < 20:
        print(f"Not enough logged predictions yet ({len(df)}). "
              f"Serve more traffic before generating a drift report.")
        return

    split_idx = int(len(df) * args.split)
    reference = df.iloc[:split_idx][["confidence", "latency_ms"]]
    current = df.iloc[split_idx:][["confidence", "latency_ms"]]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(args.out)
    print(f"Drift report written to {args.out}")


if __name__ == "__main__":
    main()
