from formatting import fmt_usd, escape_dollar_signs, severity_band


def test_fmt_usd_negative_value_has_leading_sign():
    assert fmt_usd(-811200) == "-$811,200"


def test_fmt_usd_positive_value_has_leading_plus():
    assert fmt_usd(128200) == "+$128,200"


def test_fmt_usd_none_is_explicit_not_computable():
    assert fmt_usd(None) == "not directly computable"


def test_escape_dollar_signs_prevents_latex_misinterpretation():
    text = "closed the week at $934,600 against a $980,000 plan"
    escaped = escape_dollar_signs(text)
    assert escaped == r"closed the week at \$934,600 against a \$980,000 plan"
    assert escaped.count("$") == 2  # both still present, just escaped, not stripped


def test_severity_band_covers_all_three_tiers():
    assert "minor" in severity_band(0.5)
    assert "moderate" in severity_band(1.7)
    assert "high-priority" in severity_band(2.5)
