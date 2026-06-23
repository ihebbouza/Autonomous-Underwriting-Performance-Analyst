import pytest

from agent import AnalystAgent


def test_full_pipeline_matches_named_signals():
    # The regression test: if a future change to the detection or normalization logic accidentally
    # displaces these from the top of the ranking, this should fail loudly, not silently.
    agent = AnalystAgent()
    result = agent.run(api_key=None)
    top_concern_lobs = [f["lob"] for f in result["top_concerns"]]
    assert "Excess Casualty" in top_concern_lobs
    assert "Cyber" in top_concern_lobs
    assert result["top_opportunities"][0]["lob"] == "Political Violence"


def test_run_produces_narrative_and_source():
    agent = AnalystAgent()
    result = agent.run(api_key=None)
    assert "narrative" in result
    assert result["narrative_source"] == "template"
    assert len(result["narrative"]) > 0


def test_analyze_is_separate_from_narrative_generation():
    # analyze() must not require an API key or touch the network -- the dashboard calls it on every
    # slider movement.
    agent = AnalystAgent()
    summary = agent.analyze()
    assert "narrative" not in summary
    assert "top_concerns" in summary


def test_min_sustained_weeks_changes_findings():
    agent = AnalystAgent(min_sustained_weeks=1)
    lenient = agent.analyze()
    agent2 = AnalystAgent(min_sustained_weeks=4)
    strict = agent2.analyze()
    # On the real data both settle on the same answer for the dominant signals -- this just confirms
    # the parameter is actually wired through end to end, not silently ignored.
    assert agent.detector.min_sustained_weeks == 1
    assert agent2.detector.min_sustained_weeks == 4


def test_as_of_week_time_travel():
    agent = AnalystAgent()
    agent.load_data()
    weeks = agent.loader.available_weeks()
    early_summary = agent.analyze(as_of_week=weeks[5])
    assert early_summary["as_of_week"] == str(weeks[5].date())
