import pytest

import config
from data import DataLoader
from signals import SignalDetector
from narrative import NarrativeWriter
from agent import AnalystAgent


@pytest.fixture(scope="module")
def summary():
    df = DataLoader().load()
    result = SignalDetector().find_all(df)
    loader = DataLoader()
    loader.df = df
    return {**result, "portfolio_kpis": loader.portfolio_kpis()}


def test_template_path_used_when_no_api_key(summary):
    text, source = NarrativeWriter().write(summary, api_key=None)
    assert source == "template"
    assert len(text) > 0


def test_template_has_no_placeholder_text(summary):
    text, _ = NarrativeWriter().write(summary, api_key=None)
    banned_phrases = ["TODO", "[template narrative", "no live Claude call", "re-run with an API key"]
    for phrase in banned_phrases:
        assert phrase not in text


def test_template_word_count_within_brief_range(summary):
    text, _ = NarrativeWriter().write(summary, api_key=None)
    word_count = NarrativeWriter()._count_body_words(text)
    assert config.NARRATIVE_WORD_MIN <= word_count <= config.NARRATIVE_WORD_MAX


def test_template_word_count_in_range_across_all_12_weeks():
    # Regression test: an earlier version of the template had no enforcement at all and silently
    # produced narratives ranging from 107 to 201 words depending on how many findings existed that
    # week -- both ends out of the brief's 150-200 range. Word count is one of the brief's four
    # explicit grading criteria; this checks every week in the dataset, not just the latest one.
    agent = AnalystAgent()
    agent.load_data()
    writer = NarrativeWriter()
    for wk in agent.loader.available_weeks():
        result = agent.run(as_of_week=wk, api_key=None)
        word_count = writer._count_body_words(result["narrative"])
        assert config.NARRATIVE_WORD_MIN <= word_count <= config.NARRATIVE_WORD_MAX, (
            f"{wk.date()}: {word_count} words, outside {config.NARRATIVE_WORD_MIN}-{config.NARRATIVE_WORD_MAX}"
        )


def test_template_never_has_double_periods():
    # Regression test: detail strings that already end in a period, combined naively with a template
    # that always appended its own ".", produced visible "...not a one-off.." double periods.
    agent = AnalystAgent()
    agent.load_data()
    for wk in agent.loader.available_weeks():
        result = agent.run(as_of_week=wk, api_key=None)
        assert ".." not in result["narrative"]


def test_near_miss_concern_is_mentioned_when_present():
    agent = AnalystAgent()
    summary = agent.analyze()  # latest week: Environmental is a real near-miss here
    assert len(summary["near_miss_concerns"]) > 0
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)
    assert "Environmental" in text
    assert "Close Behind" in text


def test_near_miss_not_mentioned_when_absent():
    agent = AnalystAgent()
    agent.load_data()
    summary = agent.analyze(as_of_week=agent.loader.available_weeks()[7])  # 2024-08-25: no near-miss that week
    assert len(summary["near_miss_concerns"]) == 0
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)
    assert "Close Behind" not in text


def test_near_miss_never_implies_it_was_missed_or_excluded():
    agent = AnalystAgent()
    summary = agent.analyze()
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)
    for banned in ["missed", "excluded", "should have made the top 3"]:
        assert banned not in text.lower()


def test_near_miss_does_not_sacrifice_all_materiality_when_avoidable():
    # On a week where both a near-miss mention and at least one materiality figure can coexist within
    # budget, both should survive -- the priority order should only sacrifice materiality when there's
    # genuinely no room left, not by default.
    agent = AnalystAgent()
    agent.load_data()
    writer = NarrativeWriter()
    found_a_week_with_both = False
    for wk in agent.loader.available_weeks():
        summary = agent.analyze(as_of_week=wk)
        if summary["near_miss_concerns"] and any(f["materiality_usd"] is not None for f in summary["top_concerns"]):
            text = writer._write_via_template(summary)
            if "Close Behind" in text and "$" in text.split("### Recommended Actions")[0]:
                found_a_week_with_both = True
    assert found_a_week_with_both, "No week found where both a near-miss mention and a materiality figure survive together"


def test_template_cites_every_top_concern_lob(summary):
    text, _ = NarrativeWriter().write(summary, api_key=None)
    for f in summary["top_concerns"]:
        assert f["lob"] in text


def test_template_has_recommended_action_per_finding(summary):
    text, _ = NarrativeWriter().write(summary, api_key=None)
    actions_section = text.split("### Recommended Actions")[1]
    for f in summary["top_concerns"] + summary["top_opportunities"]:
        assert f["lob"] in actions_section


def test_template_negative_materiality_formats_with_leading_sign(summary):
    text, _ = NarrativeWriter().write(summary, api_key=None)
    assert "-$811,200" in text or "$-811,200" not in text


def test_llm_path_falls_back_to_template_on_failure(summary, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("simulated network failure")

    writer = NarrativeWriter()
    monkeypatch.setattr(writer, "_write_via_llm", boom)
    text, source = writer.write(summary, api_key="fake-key-for-test")
    assert source == "template"
    assert len(text) > 0


def test_count_body_words_excludes_headers_and_title():
    writer = NarrativeWriter()
    text = (
        "Mosaic Insurance — Weekly Underwriting Narrative\n"
        "Week ending 2024-09-22\n"
        "Five real words here now.\n"
        "### Top Concerns\n"
        "One two three four five six seven eight nine ten.\n"
    )
    # Title lines + "### Top Concerns" header excluded; the two real sentences (5 + 10 words) counted.
    assert writer._count_body_words(text) == 15


def test_enforce_narrative_rules_accepts_clean_text_without_extra_call(monkeypatch):
    writer = NarrativeWriter()
    clean_text = "word " * 170  # in range, no banned phrases
    calls = []

    class FakeClient:
        class messages:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                raise AssertionError("should not be called when text already passes both checks")

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", clean_text)
    assert result == clean_text
    assert calls == []


def test_enforce_narrative_rules_requests_a_rewrite_when_too_long(monkeypatch):
    writer = NarrativeWriter()
    too_long_text = "word " * 275
    shortened_text = "shortened " * 180

    class FakeResponse:
        content = [type("Block", (), {"text": shortened_text})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "275" in last_msg
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", too_long_text)
    assert result == shortened_text


def test_enforce_narrative_rules_catches_missed_framing(monkeypatch):
    # Regression test for a real, observed failure: a live LLM call produced "Environmental just
    # missed the cutoff" despite the system prompt explicitly prohibiting that framing. Stating the
    # rule once in the prompt was not reliably enough on its own -- the same lesson the word-count
    # check already encodes -- so this is checked and corrected programmatically, not just requested.
    writer = NarrativeWriter()
    bad_text = ("word " * 170) + "Environmental just missed the cutoff this week."
    fixed_text = ("word " * 170) + "Environmental sits just behind the top 3 this week."

    class FakeResponse:
        content = [type("Block", (), {"text": fixed_text})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "missed" in last_msg.lower()
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", bad_text)
    assert result == fixed_text


def test_enforce_narrative_rules_catches_excluded_framing():
    writer = NarrativeWriter()
    bad_text = ("word " * 170) + "Environmental was excluded from the top 3 this week."

    class FakeResponse:
        content = [type("Block", (), {"text": "fixed version"})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "excluded" in last_msg.lower()
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", bad_text)
    assert result == "fixed version"


def test_enforce_narrative_rules_combines_both_issues_in_one_call():
    # If a narrative is BOTH too long AND uses banned framing, both issues should be raised in the
    # SAME corrective message, not two sequential calls -- fixing one independently risks the rewrite
    # reintroducing the other.
    writer = NarrativeWriter()
    bad_text = ("word " * 275) + "Environmental just missed the cutoff."

    class FakeResponse:
        content = [type("Block", (), {"text": "fixed version"})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "280" in last_msg  # actual computed count: 275 repeats + 5 words in the sentence
            assert "missed" in last_msg.lower()
            return FakeResponse()

    class FakeClient:
        messages = FakeMessages()

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", bad_text)
    assert result == "fixed version"


def test_enforce_narrative_rules_falls_back_to_original_if_correction_call_fails():
    writer = NarrativeWriter()
    bad_text = ("word " * 170) + "Environmental just missed the cutoff."

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("simulated network failure")

    class FakeClient:
        messages = FakeMessages()

    result = writer._enforce_narrative_rules(FakeClient(), "user prompt", bad_text)
    assert result == bad_text  # ships the flawed-but-real narrative rather than losing it entirely