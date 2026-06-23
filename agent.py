"""
AnalystAgent ties together DataLoader, SignalDetector, and NarrativeWriter. It creates its own
instances of each (composition, not dependency injection) -- there is no interchangeable alternative
implementation for any of the three in this project's scope, so there is nothing to gain from making
them swappable from outside.

analyze() and generate_narrative() are kept as two separate calls rather than one combined step,
specifically because the dashboard needs to call analyze() on every sensitivity-slider movement (cheap,
no network call) without re-triggering an LLM call each time. Only generate_narrative() touches the
network.
"""
import pandas as pd

from data import DataLoader
from signals import SignalDetector
from narrative import NarrativeWriter


class AnalystAgent:
    def __init__(self, data_dir="data", min_sustained_weeks=3):
        self.loader = DataLoader(data_dir)
        self.detector = SignalDetector(min_sustained_weeks=min_sustained_weeks)
        self.writer = NarrativeWriter()
        self.df = None

    def load_data(self):
        self.df = self.loader.load()
        return self.df

    def analyze(self, as_of_week=None, min_sustained_weeks=None):
        if self.df is None:
            self.load_data()
        if min_sustained_weeks is not None:
            self.detector.min_sustained_weeks = min_sustained_weeks
        as_of_week = as_of_week or self.loader.latest_week()
        result = self.detector.find_all(self.df, as_of_week)
        result["portfolio_kpis"] = self.loader.portfolio_kpis(as_of_week)
        result["trend"] = self._compute_trend(as_of_week, result)
        result["net_materiality_usd"] = self._compute_net_materiality(result)
        return result

    @staticmethod
    def _compute_net_materiality(result):
        """
        Sum of every known materiality figure across the top concerns and the opportunity. None of the
        individual numbers are new -- they're already computed per-finding -- but nobody ever adds them
        up, so a CUO has no single "what's the net dollar picture this week" figure without doing the
        arithmetic themselves. Returns None (not 0) if no finding this week has a known dollar figure,
        so the narrative can say so honestly rather than implying a net position of exactly zero.
        """
        total, any_known = 0.0, False
        for f in result["top_concerns"] + result["top_opportunities"]:
            if f["materiality_usd"] is not None:
                total += f["materiality_usd"]
                any_known = True
        return round(total, 0) if any_known else None

    def _compute_trend(self, as_of_week, current_result):
        """
        Week-over-week comparison: is each top concern NEW this week, or has it been in the top 3 for
        N consecutive weeks running; did anything that was a top concern last week drop out since.
        This exists because a card-based dashboard shows where things stand today -- it has no way to
        show what changed since the report a CUO read last Monday, which is the one thing a narrative
        can do that a list of cards genuinely can't. Not requested anywhere in the brief; built because
        checking it against the real data first showed the third concern slot has rotated through three
        different lines in the last three weeks alone, which is a real, demo-worthy finding, not a
        theoretical one.
        """
        weeks = self.loader.available_weeks()
        as_of_ts = pd.Timestamp(as_of_week)
        idx = weeks.index(as_of_ts) if as_of_ts in weeks else len(weeks) - 1

        if idx == 0:
            return {
                "concerns": [{"lob": f["lob"], "status": "new", "weeks_running": 1} for f in current_result["top_concerns"]],
                "resolved_concerns": [],
                "opportunity": (
                    {"lob": current_result["top_opportunities"][0]["lob"], "status": "new", "weeks_running": 1}
                    if current_result["top_opportunities"] else None
                ),
            }

        # Memoize per-week findings -- several concerns each walking backward independently would
        # otherwise recompute the same earlier week's findings multiple times.
        cache = {idx: current_result}

        def findings_for(i):
            if i not in cache:
                cache[i] = self.detector.find_all(self.df, weeks[i])
            return cache[i]

        def streak(lob, in_top_fn):
            count = 0
            for i in range(idx, -1, -1):
                if in_top_fn(findings_for(i), lob):
                    count += 1
                else:
                    break
            return count

        prior_result = findings_for(idx - 1)
        prior_concern_lobs = {f["lob"] for f in prior_result["top_concerns"]}
        # Dedupe by LoB, not by finding -- a single line can have two different checks (e.g. both
        # gwp_band and loss_ratio_trend) both land in the same week's top 3, which would otherwise
        # produce the same line twice in the trend list. Order preserved from top_concerns, which is
        # already severity-sorted, so the first occurrence per LoB is its highest-severity finding.
        seen, current_lobs_ordered = set(), []
        for f in current_result["top_concerns"]:
            if f["lob"] not in seen:
                seen.add(f["lob"])
                current_lobs_ordered.append(f["lob"])
        current_concern_lobs = set(current_lobs_ordered)

        concerns_trend = []
        for lob in current_lobs_ordered:
            if lob in prior_concern_lobs:
                weeks_running = streak(lob, lambda r, lob: lob in {x["lob"] for x in r["top_concerns"]})
                concerns_trend.append({"lob": lob, "status": "continuing", "weeks_running": weeks_running})
            else:
                concerns_trend.append({"lob": lob, "status": "new", "weeks_running": 1})

        resolved_concerns = sorted(prior_concern_lobs - current_concern_lobs)

        opportunity_trend = None
        if current_result["top_opportunities"]:
            opp_lob = current_result["top_opportunities"][0]["lob"]
            prior_opp_lobs = {f["lob"] for f in prior_result["top_opportunities"]}
            if opp_lob in prior_opp_lobs:
                weeks_running = streak(
                    opp_lob,
                    lambda r, lob: bool(r["top_opportunities"]) and r["top_opportunities"][0]["lob"] == lob,
                )
                opportunity_trend = {"lob": opp_lob, "status": "continuing", "weeks_running": weeks_running}
            else:
                opportunity_trend = {"lob": opp_lob, "status": "new", "weeks_running": 1}

        return {
            "concerns": concerns_trend,
            "resolved_concerns": resolved_concerns,
            "opportunity": opportunity_trend,
        }

    def generate_narrative(self, summary, api_key=None):
        text, source = self.writer.write(summary, api_key=api_key)
        return {**summary, "narrative": text, "narrative_source": source}

    def run(self, as_of_week=None, min_sustained_weeks=None, api_key=None):
        summary = self.analyze(as_of_week, min_sustained_weeks)
        return self.generate_narrative(summary, api_key=api_key)