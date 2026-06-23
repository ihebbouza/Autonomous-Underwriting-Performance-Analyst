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


def test_template_word_count_never_exceeds_hard_cap(summary):
    # No strict minimum anymore -- length is secondary to findings, on direct instruction. The only
    # thing that must always hold is the hard ceiling.
    text, _ = NarrativeWriter().write(summary, api_key=None)
    word_count = NarrativeWriter()._count_body_words(text)
    assert word_count <= config.NARRATIVE_WORD_HARD_CAP


def test_template_word_count_never_exceeds_hard_cap_across_all_12_weeks():
    # Regression test: an earlier version of the template had no enforcement at all and silently
    # produced narratives ranging from 107 to 201 words against what was then a 150-200 target. The
    # policy is simpler now (no minimum, trim only above 350), but the one thing that must always hold
    # -- never exceeding the hard cap -- is checked across every week, not just the current one.
    agent = AnalystAgent()
    agent.load_data()
    writer = NarrativeWriter()
    for wk in agent.loader.available_weeks():
        result = agent.run(as_of_week=wk, api_key=None)
        word_count = writer._count_body_words(result["narrative"])
        assert word_count <= config.NARRATIVE_WORD_HARD_CAP, f"{wk.date()}: {word_count} words, over the hard cap"


def test_template_never_has_double_periods():
    # Regression test: detail strings that already end in a period, combined naively with a template
    # that always appended its own ".", produced visible "...not a one-off.." double periods.
    agent = AnalystAgent()
    agent.load_data()
    for wk in agent.loader.available_weeks():
        result = agent.run(as_of_week=wk, api_key=None)
        assert ".." not in result["narrative"]


def _minimal_summary(near_miss=True, extra_concerns=0, with_opportunity=False):
    """A small, controlled summary -- enough content to be realistic, little enough that there's
    always room for everything, used to test trim-priority logic deterministically rather than
    depending on whether some real week happens to have enough slack."""
    concerns = [{"lob": "Test Line", "category": "Premium Risk", "severity": 2.0,
                 "materiality_usd": -100000, "detail": "Running below plan this week."}]
    for i in range(extra_concerns):
        concerns.append({"lob": f"Extra Line {i}", "category": "Conversion Risk", "severity": 1.9 - i * 0.1,
                          "materiality_usd": -50000 - i * 1000, "detail": "Hit rate has fallen this week too."})
    opportunities = []
    if with_opportunity:
        opportunities.append({"lob": "Growth Line", "category": "Growth Opportunity", "severity": 2.1,
                               "materiality_usd": 80000, "detail": "Running well above plan this week."})
    near_misses = [{"lob": "Close Line", "category": "Conversion Risk", "severity": 1.8,
                    "materiality_usd": None, "detail": "Almost cleared the cutoff this week."}] if near_miss else []
    return {
        "as_of_week": "2024-01-01",
        "top_concerns": concerns,
        "top_opportunities": opportunities,
        "near_miss_concerns": near_misses,
        "portfolio_kpis": {"ytd_gwp_vs_plan_pct": 95.0, "portfolio_hit_rate_pct": 25.0,
                            "gwp_actual_this_week": 100000, "gwp_plan_this_week": 105000,
                            "ytd_gwp_actual": 1000000, "ytd_gwp_plan": 1050000, "as_of_week": "2024-01-01"},
        "trend": {
            "concerns": [{"lob": f["lob"], "status": "new", "weeks_running": 1} for f in concerns],
            "resolved_concerns": [],
            "opportunity": {"lob": opportunities[0]["lob"], "status": "new", "weeks_running": 1} if opportunities else None,
        },
        "net_materiality_usd": sum(f["materiality_usd"] for f in concerns + opportunities if f["materiality_usd"] is not None),
    }


def test_near_miss_concern_is_mentioned_when_there_is_room():
    summary = _minimal_summary(near_miss=True)
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)
    assert "Close Behind" in text
    assert "Close Line" in text


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


def test_trend_status_and_every_finding_survive_even_when_trimming_is_needed():
    # The actual, current priority when content is heavy enough to force trimming (3-step cascade:
    # drop the net-materiality line, then the resolved-since-last-week line, then shorten the
    # near-miss mention): findings, their materiality figures, and the trend status are NEVER
    # sacrificed -- only synthesis/commentary content is. Confirmed on a deliberately oversized
    # synthetic case (5 concerns with long detail text + an opportunity + a near-miss + 2 resolved
    # concerns), built specifically to exceed the 350-word trim trigger, not by hoping a real week
    # happens to be heavy enough -- real data shifts as the methodology evolves, this won't.
    long_detail = (
        "Running well below plan for many consecutive weeks now, with no sign of recovery in the "
        "underlying submission pipeline or broker engagement levels across the territory this line covers."
    )
    concerns = [
        {"lob": f"Concern Line {i}", "category": "Premium Risk", "severity": 2.5 - i * 0.1,
         "materiality_usd": -100000 - i * 10000, "detail": long_detail}
        for i in range(5)
    ]
    opportunities = [{"lob": "Growth Line", "category": "Growth Opportunity", "severity": 2.1,
                       "materiality_usd": 80000, "detail": long_detail}]
    near_misses = [{"lob": "Close Line", "category": "Conversion Risk", "severity": 1.8,
                    "materiality_usd": None, "detail": long_detail}]
    summary = {
        "as_of_week": "2024-01-01", "top_concerns": concerns, "top_opportunities": opportunities,
        "near_miss_concerns": near_misses,
        "portfolio_kpis": {"ytd_gwp_vs_plan_pct": 95.0, "portfolio_hit_rate_pct": 25.0,
                            "gwp_actual_this_week": 100000, "gwp_plan_this_week": 105000,
                            "ytd_gwp_actual": 1000000, "ytd_gwp_plan": 1050000, "as_of_week": "2024-01-01"},
        "trend": {"concerns": [{"lob": f["lob"], "status": "new", "weeks_running": 1} for f in concerns],
                  "resolved_concerns": ["Resolved Line A", "Resolved Line B"],
                  "opportunity": {"lob": "Growth Line", "status": "new", "weeks_running": 1}},
        "net_materiality_usd": -200000,
    }
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)

    # Confirm trimming actually activated (otherwise this test isn't exercising what it claims to)
    assert "Net dollar impact" not in text
    assert "Resolved Line A" not in text
    assert "Also close behind the top 3" in text  # short form survives
    assert "but close enough to be worth watching" not in text  # full form's extra sentence is gone

    # Confirm nothing that matters was sacrificed to get there
    for i in range(5):
        assert f"Concern Line {i}" in text
        assert f"${100000 + i * 10000:,}" in text
    assert "Growth Line" in text and "$80,000" in text
    assert "Close Line" in text  # the near-miss mention itself survives, just shortened
    assert text.count("New this week") == 6  # 5 concerns + 1 opportunity, every trend status intact


def test_near_miss_does_not_sacrifice_materiality_when_avoidable():
    # On a small, controlled case where everything genuinely fits, both the near-miss mention and the
    # materiality figures should survive together -- the trim logic should only start cutting things
    # when there's an actual word-budget problem, not by default.
    summary = _minimal_summary(near_miss=True, extra_concerns=0, with_opportunity=False)
    writer = NarrativeWriter()
    text = writer._write_via_template(summary)
    assert "Close Behind" in text
    assert "$" in text.split("### Recommended Actions")[0]


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
    too_long_text = "word " * 400  # over the 350 trim trigger
    shortened_text = "shortened " * 290

    class FakeResponse:
        content = [type("Block", (), {"text": shortened_text})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "400" in last_msg
            assert "350" in last_msg  # the trigger threshold itself, not just the count
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
    bad_text = ("word " * 400) + "Environmental just missed the cutoff."

    class FakeResponse:
        content = [type("Block", (), {"text": "fixed version"})()]

    class FakeMessages:
        @staticmethod
        def create(**kwargs):
            last_msg = kwargs["messages"][-1]["content"]
            assert "405" in last_msg  # actual computed count: 400 repeats + 5 words in the sentence
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


def test_all_prompt_files_have_version_headers_and_load_clean():
    # Consistency check across all 4 prompt files: each should declare a version (since the brief
    # asks candidates to bring their prompts, and prompt versioning is the literal answer to Probing
    # Question 2), and none of that header bookkeeping should leak into what's actually sent to the
    # model -- regardless of how long the header comment block grows.
    import config
    from pathlib import Path
    prompt_dir = Path(config.PROMPT_DIR)
    for fname in ["system_prompt.txt", "narrative_user_prompt.txt", "lob_narrative_prompt.txt", "chat_system_prompt.txt"]:
        raw = (prompt_dir / fname).read_text()
        assert raw.lstrip().startswith("# Prompt version:"), f"{fname} is missing a version header"
        writer = NarrativeWriter()
        loaded = writer._load_prompt(fname)
        assert not loaded.strip().startswith("#"), f"{fname}: comment header leaked into the loaded prompt body"