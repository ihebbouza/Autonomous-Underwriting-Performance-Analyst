from streamlit.testing.v1 import AppTest


def test_app_loads_without_error():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception


def test_app_shows_top_concerns():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    body = "\n".join(m.value for m in at.markdown)
    assert "Excess Casualty" in body
    assert "Cyber" in body


def test_sensitivity_slider_exists_and_moves():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    sliders = [s for s in at.slider if "Sensitivity" in s.label]
    assert len(sliders) == 1
    sliders[0].set_value(1).run(timeout=30)
    assert not at.exception


def test_time_travel_select_slider_exists():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    select_sliders = [s for s in at.select_slider if "time travel" in s.label.lower()]
    assert len(select_sliders) == 1


def test_lob_drilldown_selectbox_has_all_eight_lobs():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    boxes = [b for b in at.selectbox if "Line of business" in b.label]
    assert len(boxes) == 1
    assert len(boxes[0].options) == 8


def test_no_exception_without_api_key():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    assert not at.exception


def test_overview_has_gwp_trend_by_lob_and_heatmap():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    headers = [h.value for h in at.subheader]
    assert any("12-Week Trend by LoB" in h for h in headers)
    assert any("Heatmap" in h for h in headers)
    assert len(at.get("plotly_chart")) >= 2


def test_heatmap_xaxis_is_forced_categorical_not_autodetected_as_dates():
    # Regression test for the bug where Plotly auto-detected the "MM-DD" string column labels as
    # dates and re-rendered them with an unrelated, incorrect year (2008/2009 instead of 2024).
    from charts import build_hit_rate_heatmap
    from data import DataLoader

    df = DataLoader().load()
    fig = build_hit_rate_heatmap(df)
    assert fig.layout.xaxis.type == "category"
    assert fig.layout.yaxis.type == "category"


def test_chat_tab_has_quick_question_buttons_before_first_message():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    button_labels = [b.label for b in at.button]
    assert any("top concerns" in lbl.lower() for lbl in button_labels)


def test_narrative_staleness_warning_appears_after_time_travel():
    # Note: the sensitivity slider does NOT change the top-3 ranking on the real dataset -- Excess
    # Casualty, Cyber, and Transactional Liability clear threshold regardless of sensitivity 1, 3, or 4,
    # because the signals are unambiguous, not borderline (confirmed directly against AnalystAgent
    # before writing this test). Time-travel (as-of-week) reliably changes findings instead, since
    # earlier weeks have less history for checks like loss-ratio trend and claims anomaly -- that's
    # the parameter this test uses to exercise staleness detection end to end.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    generate_buttons = [b for b in at.button if "Generate narrative" in b.label]
    assert len(generate_buttons) == 1
    generate_buttons[0].click().run(timeout=30)
    assert not any("have changed" in w.value.lower() for w in at.warning)

    time_travel_sliders = [s for s in at.select_slider if "time travel" in s.label.lower()]
    earliest_week = time_travel_sliders[0].options[5]
    time_travel_sliders[0].set_value(earliest_week).run(timeout=30)
    assert any("have changed" in w.value.lower() for w in at.warning)


def test_staleness_warning_disappears_immediately_on_regenerate_click():
    # Regression test for the actual reported bug: the warning was computed from a variable captured
    # BEFORE the button-click block ran, so clicking "Regenerate" while stale still displayed the
    # warning on that same render -- even though the click had just fixed the staleness. The fix
    # re-checks staleness fresh, after the button logic, not from the pre-click snapshot.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    generate_buttons = [b for b in at.button if "Generate narrative" in b.label]
    generate_buttons[0].click().run(timeout=30)

    time_travel_sliders = [s for s in at.select_slider if "time travel" in s.label.lower()]
    time_travel_sliders[0].set_value(time_travel_sliders[0].options[5]).run(timeout=30)
    assert any("have changed" in w.value.lower() for w in at.warning)

    regenerate_buttons = [b for b in at.button if "narrative" in b.label.lower()]
    assert regenerate_buttons[0].label == "Regenerate narrative"
    regenerate_buttons[0].click().run(timeout=30)
    assert not any("have changed" in w.value.lower() for w in at.warning), (
        "Warning is still showing on the SAME run as the regenerate click that should have cleared it"
    )


def test_drilldown_shows_a_plain_language_lob_narrative_not_a_raw_findings_list():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    # Select Political Risk specifically -- the line the bare "Distribution Friction Risk: 27.6 days
    # vs peer avg 25.7 (severity 1.15)" complaint was about.
    drilldown_box = [b for b in at.selectbox if "Line of business" in b.label][0]
    drilldown_box.select("Political Risk").run(timeout=30)
    headers = [h.value for h in at.subheader]
    assert any("What's happening with Political Risk" in h for h in headers)
    body = "\n".join(m.value for m in at.markdown)
    # The category explanation text must actually be present, not just the bare label
    assert "underwriting" in body.lower() and "quotes" in body.lower()


def test_chat_input_with_dollar_sign_is_escaped_before_display():
    # Regression test: the assistant's reply was already escaped, but the user's OWN typed question
    # was being passed straight to .write() unescaped -- a question containing a dollar figure would
    # hit the same Streamlit LaTeX-misinterpretation bug fixed elsewhere on the page.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    at.chat_input[0].set_value("What about the $293,000 spike?").run(timeout=30)
    rendered = []
    for chat_msg in at.get("chat_message"):
        rendered.extend(md.value for md in chat_msg.markdown)
    assert any(r"\$293,000" in text for text in rendered)


def test_app_source_has_no_emoji():
    # Explicit design decision: emoji read as an "AI-generated" tell in a CUO-facing tool. Status and
    # category indicators use st.badge() (colored text labels, the same pattern Jira/GitHub use for
    # status tags) instead, and chat avatars use Streamlit's own built-in "user"/"assistant" defaults
    # rather than custom emoji. This test exists so a future edit doesn't quietly reintroduce them.
    import re
    source = open("app.py", encoding="utf-8").read()
    emoji_pattern = re.compile(
        "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]+",
        flags=re.UNICODE,
    )
    matches = emoji_pattern.findall(source)
    assert matches == [], f"Emoji found in app.py: {matches}"


def test_tab_styling_css_is_injected():
    # Verifies the equal-width tab CSS block is actually present on the page. Doesn't verify rendered
    # appearance (AppTest has no browser/CSS engine), only that the style block reaches the page --
    # the actual selectors were verified separately, directly against the installed Streamlit version's
    # frontend bundle, not assumed from documentation.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    css_blocks = [m.value for m in at.markdown if "stTabs" in m.value and "flex" in m.value]
    assert len(css_blocks) == 1


def test_portfolio_health_banner_present():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    info_blocks = [i.value for i in at.info]
    assert any("This week" in i and "concern" in i for i in info_blocks)


def test_finding_cards_render_with_severity_metrics():
    # st.container() itself isn't a directly queryable element type in AppTest -- it tracks the
    # widgets inside containers, not the container wrapper. This checks for the actual content that
    # only exists inside the new card layout: one "Severity" metric per finding card (3 top concerns +
    # 1 opportunity on the real data this week), confirming the card structure is actually rendering,
    # not silently falling back to flat markdown.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    severity_metrics = [m for m in at.get("metric") if m.label == "Severity"]
    assert len(severity_metrics) >= 4  # 3 top concerns + 1 opportunity on the real data


def test_drilldown_quick_glance_metrics_present():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    metric_labels = [m.label for m in at.get("metric")]
    assert any("GWP vs Plan" in lbl for lbl in metric_labels)
    assert any("Hit Rate" in lbl for lbl in metric_labels)
    assert any("Loss Ratio" in lbl for lbl in metric_labels)


def test_overview_shows_near_miss_concern():
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    body = "\n".join(m.value for m in at.markdown)
    assert "Environmental" in body
    assert "just outside the top 3" in body


def test_chat_exchange_renders_with_both_messages_after_asking():
    # Regression test: chat_input is called LAST in the script (so it's visually below the message
    # history, working around a known Streamlit limitation where chat_input doesn't pin to the bottom
    # inside st.tabs). That means a fresh question can't be rendered inline after capturing it --
    # doing so would put it BELOW the already-rendered chat_input, the exact problem being fixed. The
    # real fix calls st.rerun() so the next pass's history loop (which now includes the new exchange)
    # renders it in its correct position above chat_input. This also depends on DataAssistant.ask()
    # actually appending to history on the no-key path, not just returning a value directly -- a
    # separate regression covered in test_chat.py.
    at = AppTest.from_file("app.py")
    at.run(timeout=30)
    at.chat_input[0].set_value("What are the top concerns?").run(timeout=30)
    rendered = []
    for chat_msg in at.get("chat_message"):
        rendered.extend(md.value for md in chat_msg.markdown)
    assert any("What are the top concerns?" in text for text in rendered)
    assert any("ANTHROPIC_API_KEY" in text for text in rendered)
    # chat_input must still be present and usable for the next question
    assert len(at.chat_input) == 1