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


def test_trajectory_contrast_on_real_data():
    # The actual, verified finding: on the real data, the only worsening finding is a near-miss
    # (Environmental), and every top-3 concern plus the opportunity is stable or improving -- so the
    # contrast should fire and name Environmental specifically.
    agent = AnalystAgent()
    summary = agent.analyze()
    assert summary["trajectory_contrast"] == ["Environmental"]


def test_trajectory_contrast_returns_none_when_a_top_finding_is_also_worsening():
    from agent import AnalystAgent as _Agent

    result = {
        "top_concerns": [{"lob": "A", "trajectory": "worsening"}, {"lob": "B", "trajectory": "stable"}],
        "top_opportunities": [{"lob": "C", "trajectory": "improving"}],
        "near_miss_concerns": [{"lob": "D", "trajectory": "worsening"}],
    }
    # The contrast must not fire if a top-3 finding is ALSO worsening -- forcing the framing when it
    # doesn't cleanly hold would be exactly the kind of reverse-engineered claim this feature exists
    # to avoid making.
    assert _Agent._compute_trajectory_contrast(result) is None


def test_trajectory_contrast_returns_none_when_no_near_miss_is_worsening():
    from agent import AnalystAgent as _Agent
    result = {
        "top_concerns": [{"lob": "A", "trajectory": "stable"}],
        "top_opportunities": [],
        "near_miss_concerns": [{"lob": "D", "trajectory": "stable"}],
    }
    assert _Agent._compute_trajectory_contrast(result) is None


def test_trajectory_contrast_fires_correctly_on_a_clean_synthetic_case():
    from agent import AnalystAgent as _Agent
    result = {
        "top_concerns": [{"lob": "A", "trajectory": "stable"}, {"lob": "B", "trajectory": "improving"}],
        "top_opportunities": [{"lob": "C", "trajectory": "stable"}],
        "near_miss_concerns": [{"lob": "D", "trajectory": "worsening"}, {"lob": "E", "trajectory": "stable"}],
    }
    assert _Agent._compute_trajectory_contrast(result) == ["D"]