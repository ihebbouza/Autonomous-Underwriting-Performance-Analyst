"""
DataLoader reads the four weekly extracts, checks they have the shape it expects, checks they agree
with each other on which weeks they cover, and joins them into one table.

This is a structural check, not a data-quality check: it verifies columns are present and weeks are
consistent across files. It does not check for duplicate rows, missing values inside a column, or
whether a number looks sane (e.g. a negative GWP would not be caught). That is a known, stated scope
limit -- see README.
"""
from pathlib import Path
import pandas as pd

import config


class DataValidationError(Exception):
    """Raised when an input file doesn't match what we expect, or the four files disagree on weeks."""


class DataLoader:
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.df = None

    def load(self):
        frames = {}
        filenames = {
            "submissions": "case4_weekly_submissions.csv",
            "premium": "case4_weekly_premium.csv",
            "pipeline": "case4_pipeline.csv",
            "loss": "case4_loss_indicators.csv",
        }
        for key, filename in filenames.items():
            path = self.data_dir / filename
            if not path.exists():
                raise DataValidationError(f"Missing input file: {path}")
            df = pd.read_csv(path)
            missing = set(config.REQUIRED_COLUMNS[key]) - set(df.columns)
            if missing:
                raise DataValidationError(f"{filename} is missing required column(s): {sorted(missing)}")
            df["week_ending"] = pd.to_datetime(df["week_ending"])
            frames[key] = df

        self._check_weeks_match(frames)

        merged = (
            frames["submissions"]
            .merge(frames["premium"], on=["week_ending", "lob"])
            .merge(frames["pipeline"], on=["week_ending", "lob"])
            .merge(frames["loss"], on=["week_ending", "lob"])
        )

        denom = merged["bound_count"] + merged["quoted_count"] + merged["declined_count"] + merged["ntu_count"]
        merged["hit_rate"] = merged["bound_count"] / denom * 100
        merged["gwp_vs_plan_pct"] = merged["actual_gwp"] / merged["plan_gwp"] * 100
        merged["gwp_variance"] = merged["actual_gwp"] - merged["plan_gwp"]

        if merged["attritional_loss_ratio_ytd"].max() <= 1:
            merged["attritional_loss_ratio_ytd"] = merged["attritional_loss_ratio_ytd"] * 100

        self.df = merged.sort_values(["lob", "week_ending"]).reset_index(drop=True)
        return self.df

    @staticmethod
    def _check_weeks_match(frames):
        week_sets = {key: set(df["week_ending"]) for key, df in frames.items()}
        reference_key = "submissions"
        reference = week_sets[reference_key]
        for key, weeks in week_sets.items():
            if weeks != reference:
                raise DataValidationError(
                    f"Week mismatch: '{key}' covers a different set of weeks than '{reference_key}'. "
                    f"This usually means one of the extracts is stale or from a different run."
                )

    def available_weeks(self):
        if self.df is None:
            raise RuntimeError("Call load() first.")
        return sorted(self.df["week_ending"].unique())

    def latest_week(self):
        return self.available_weeks()[-1]

    def portfolio_kpis(self, as_of_week=None):
        if self.df is None:
            raise RuntimeError("Call load() first.")
        as_of_week = as_of_week or self.latest_week()
        wk = self.df[self.df["week_ending"] == as_of_week]
        ytd_actual = float(wk["ytd_actual"].sum())
        ytd_plan = float(wk["ytd_plan"].sum())
        return {
            "as_of_week": str(pd.Timestamp(as_of_week).date()),
            "gwp_actual_this_week": round(float(wk["actual_gwp"].sum()), 0),
            "gwp_plan_this_week": round(float(wk["plan_gwp"].sum()), 0),
            "ytd_gwp_actual": round(ytd_actual, 0),
            "ytd_gwp_plan": round(ytd_plan, 0),
            "ytd_gwp_vs_plan_pct": round(100 * ytd_actual / ytd_plan, 1) if ytd_plan else None,
            "portfolio_hit_rate_pct": round(float(wk["hit_rate"].mean()), 1),
        }
