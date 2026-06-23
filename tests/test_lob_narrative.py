import pytest

from agent import AnalystAgent
from narrative import NarrativeWriter


@pytest.fixture(scope="module")
def summary():
    return AnalystAgent().analyze()


@pytest.fixture
def writer():
    return NarrativeWriter()


def test_lob_with_finding_explains_the_category_in_plain_language(summary, writer):
    findings = [f for f in summary["all_concerns"] + summary["all_opportunities"] if f["lob"] == "Political Risk"]
    text, source = writer.write_lob_narrative("Political Risk", findings, {"gwp_vs_plan_pct": 103, "hit_rate_pct": 28, "loss_ratio_pct": 44}, api_key=None)
    assert source == "template"
    assert "quotes" in text.lower() and "underwriting" in text.lower()  # the actual explanation text, not just the label
    assert "Distribution Friction Risk" in text


def test_lob_with_no_findings_still_produces_a_confirmation_not_silence(summary, writer):
    text, source = writer.write_lob_narrative("Financial Institutions", [], {"gwp_vs_plan_pct": 100, "hit_rate_pct": 27, "loss_ratio_pct": 45}, api_key=None)
    assert "no flagged concerns" in text.lower()
    assert "100" in text and "27" in text and "45" in text


def test_lob_narrative_mentions_severity_in_plain_language_not_just_a_raw_number(summary, writer):
    findings = [{"lob": "Political Risk", "check": "pipeline_friction", "direction": "concern",
                 "category": "Distribution Friction Risk", "severity": 1.15, "materiality_usd": None,
                 "detail": "Taking 27.6 days vs peer avg 25.7"}]
    text, _ = writer.write_lob_narrative("Political Risk", findings, {}, api_key=None)
    # The raw number alone isn't banned, but a plain-language read must also be present
    assert "minor signal" in text.lower() or "moderate signal" in text.lower() or "high-priority signal" in text.lower()


def test_lob_narrative_states_materiality_unavailable_rather_than_omitting_it(summary, writer):
    findings = [{"lob": "Political Risk", "check": "pipeline_friction", "direction": "concern",
                 "category": "Distribution Friction Risk", "severity": 1.15, "materiality_usd": None,
                 "detail": "Taking 27.6 days vs peer avg 25.7"}]
    text, _ = writer.write_lob_narrative("Political Risk", findings, {}, api_key=None)
    assert "no dollar materiality" in text.lower() or "not directly computable" in text.lower()


def test_lob_narrative_with_materiality_cites_the_figure(summary, writer):
    findings = [f for f in summary["all_concerns"] if f["lob"] == "Excess Casualty"]
    text, _ = writer.write_lob_narrative("Excess Casualty", findings, {"gwp_vs_plan_pct": 58, "hit_rate_pct": 25, "loss_ratio_pct": 50}, api_key=None)
    assert "811,200" in text


def test_lob_narrative_llm_path_falls_back_to_template_on_failure(summary, writer, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated failure")
    monkeypatch.setattr(writer, "_write_lob_via_llm", boom)
    findings = [f for f in summary["all_concerns"] if f["lob"] == "Excess Casualty"]
    text, source = writer.write_lob_narrative("Excess Casualty", findings, {}, api_key="fake-key")
    assert source == "template"
    assert len(text) > 0
