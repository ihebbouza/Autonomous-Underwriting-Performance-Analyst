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
        return result

    def generate_narrative(self, summary, api_key=None):
        text, source = self.writer.write(summary, api_key=api_key)
        return {**summary, "narrative": text, "narrative_source": source}

    def run(self, as_of_week=None, min_sustained_weeks=None, api_key=None):
        summary = self.analyze(as_of_week, min_sustained_weeks)
        return self.generate_narrative(summary, api_key=api_key)
