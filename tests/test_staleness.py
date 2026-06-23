from staleness import fingerprint, is_stale


def make_summary(top_concern_severity=2.47, has_opportunity=True):
    summary = {
        "top_concerns": [{"lob": "Excess Casualty", "check": "gwp_band", "severity": top_concern_severity, "category": "Premium Risk"}],
        "top_opportunities": [{"lob": "Political Violence", "check": "gwp_band", "severity": 2.47, "category": "Growth Opportunity"}] if has_opportunity else [],
    }
    return summary


def test_identical_summary_and_key_is_not_stale():
    summary = make_summary()
    fp = fingerprint(summary, has_api_key=False)
    assert not is_stale(fp, summary, has_api_key=False)


def test_changed_severity_is_stale():
    original = make_summary(top_concern_severity=2.47)
    fp = fingerprint(original, has_api_key=False)
    changed = make_summary(top_concern_severity=1.80)  # e.g. sensitivity slider changed the ranking
    assert is_stale(fp, changed, has_api_key=False)


def test_unchanged_findings_despite_different_slider_value_is_not_stale():
    # The whole point: a slider move that happens not to change the actual findings should not
    # trigger a false "stale" warning. The fingerprint is based on findings, not raw slider state.
    summary = make_summary()
    fp = fingerprint(summary, has_api_key=False)
    same_summary_different_moment = make_summary()  # identical findings, as if slider moved but result didn't change
    assert not is_stale(fp, same_summary_different_moment, has_api_key=False)


def test_adding_api_key_after_generation_marks_stale():
    summary = make_summary()
    fp = fingerprint(summary, has_api_key=False)
    assert is_stale(fp, summary, has_api_key=True)


def test_removing_api_key_after_generation_marks_stale():
    summary = make_summary()
    fp = fingerprint(summary, has_api_key=True)
    assert is_stale(fp, summary, has_api_key=False)


def test_lob_change_in_top_concerns_is_stale():
    original = make_summary()
    fp = fingerprint(original, has_api_key=False)
    changed = {
        "top_concerns": [{"lob": "Cyber", "check": "hit_rate_collapse", "severity": 2.28, "category": "Conversion Risk"}],
        "top_opportunities": original["top_opportunities"],
    }
    assert is_stale(fp, changed, has_api_key=False)
