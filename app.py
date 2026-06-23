"""
Streamlit dashboard. Five views: KPIs + findings + narrative, sensitivity control, per-LoB drill-down,
chat, and the raw JSON/prompts for anyone who wants to see exactly what produced the narrative.

Streamlit reruns this whole script on every interaction. analyze() is cheap (no network call) so the
sensitivity slider can call it on every movement; generate_narrative() only runs when the button below
is clicked, since that's the one step that can touch a live model.
"""
import os

import pandas as pd
import plotly.express as px
import streamlit as st

import config
from agent import AnalystAgent
from chat import DataAssistant
from charts import build_hit_rate_heatmap
from data import DataValidationError
from formatting import fmt_usd, escape_dollar_signs
from staleness import fingerprint, is_stale

st.set_page_config(page_title="Mosaic Underwriting Performance Pack", layout="wide")

# Equal-width, full-bleed tabs instead of Streamlit's default left-aligned, content-width tabs.
# Selectors verified directly against this environment's installed Streamlit version (1.58) by
# grepping its actual frontend JS bundle for real data-testid/role attributes, rather than assumed
# from documentation that can drift between versions -- both data-testid="stTabs"/"stTab" and the
# underlying role="tablist"/role="tab" ARIA attributes are confirmed present; both are targeted so
# the styling survives if either one changes in a future Streamlit release.
st.markdown("""
<style>
[data-testid="stTabs"] [role="tablist"] {
    display: flex;
    width: 100%;
    gap: 0.25rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
}
[data-testid="stTabs"] button[role="tab"] {
    flex: 1;
    justify-content: center;
    font-size: 1.05rem;
    font-weight: 600;
    padding: 0.85rem 1rem;
    border-radius: 0.5rem 0.5rem 0 0;
    transition: background-color 0.15s ease;
}
[data-testid="stTabs"] button[role="tab"]:hover {
    background-color: rgba(255, 255, 255, 0.05);
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background-color: rgba(255, 75, 75, 0.08);
}
</style>
""", unsafe_allow_html=True)

if "agent" not in st.session_state:
    st.session_state.agent = AnalystAgent()
    try:
        st.session_state.agent.load_data()
    except DataValidationError as exc:
        st.error(f"Data validation failed: {exc}")
        st.stop()

agent = st.session_state.agent
weeks = agent.loader.available_weeks()

st.title("Mosaic Underwriting Performance Pack")

with st.sidebar:
    st.header("Controls")
    as_of_week = st.select_slider("As-of week (time travel)", options=weeks, value=weeks[-1],
                                   format_func=lambda d: pd.Timestamp(d).strftime("%Y-%m-%d"))
    min_sustained_weeks = st.slider(
        "Sensitivity: minimum sustained weeks (hit-rate check)", min_value=1, max_value=4, value=3,
        help="Lower = more sensitive to short-lived dips. Higher = only sustained patterns are flagged. "
             "This directly answers 'what if it was just a one-off week?' (Probing Question 1)."
    )
    st.divider()
    api_key = st.text_input("ANTHROPIC_API_KEY (optional)", type="password",
                             value=os.environ.get("ANTHROPIC_API_KEY", ""))
    st.caption("Without a key: offline template narrative + chat disabled.")

summary = agent.analyze(as_of_week=as_of_week, min_sustained_weeks=min_sustained_weeks)

st.caption(f"Weekly underwriting performance, automated — as of {summary['portfolio_kpis']['as_of_week']}")

_flagged_lobs = {f["lob"] for f in summary["top_concerns"] + summary["top_opportunities"] + summary.get("near_miss_concerns", [])}
_clean_count = len(config.LINES_OF_BUSINESS) - len(_flagged_lobs)
st.info(
    f"**This week:** {len(summary['top_concerns'])} concern(s) flagged, "
    f"{len(summary['top_opportunities'])} opportunity flagged, "
    f"{_clean_count} of {len(config.LINES_OF_BUSINESS)} lines clean."
)

kpi_cols = st.columns(4)
kpis = summary["portfolio_kpis"]
kpi_cols[0].metric("YTD GWP vs Plan", f"{kpis['ytd_gwp_vs_plan_pct']}%")
kpi_cols[1].metric("Portfolio Hit Rate", f"{kpis['portfolio_hit_rate_pct']}%")
kpi_cols[2].metric("This Week's GWP", f"${kpis['gwp_actual_this_week']:,.0f}")
kpi_cols[3].metric("As-of Week", kpis["as_of_week"])

tab_overview, tab_drilldown, tab_chat, tab_raw = st.tabs(
    ["Overview", "LoB Drill-down", "Ask a Question", "JSON / Prompts"]
)

with tab_overview:
    st.subheader("GWP vs. Plan — 12-Week Trend by LoB")
    trend_df = agent.df[agent.df["week_ending"] <= pd.Timestamp(as_of_week)]
    fig_trend = px.line(trend_df, x="week_ending", y="gwp_vs_plan_pct", color="lob", markers=True,
                         title=None, labels={"gwp_vs_plan_pct": "% of Plan", "week_ending": "Week"})
    fig_trend.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Plan (100%)")
    st.plotly_chart(fig_trend, use_container_width=True)

    st.subheader("Hit Rate Heatmap — LoB vs. Week")
    fig_heat = build_hit_rate_heatmap(trend_df)
    st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Top Concerns")
    st.caption("Ranked by statistical severity relative to peers -- not the same as business urgency. "
               "See the materiality figure for dollar impact where it's directly computable.")
    for f in summary["top_concerns"]:
        with st.container(border=True):
            card_cols = st.columns([4, 1])
            with card_cols[0]:
                st.badge("CONCERN", color="red")
                st.markdown(f"#### {f['lob']}")
                st.caption(f["category"])
            card_cols[1].metric("Severity", f"{f['severity']}")
            st.write(escape_dollar_signs(f["detail"]))
            st.caption(f"Materiality: {escape_dollar_signs(fmt_usd(f['materiality_usd']))}")

    if summary.get("near_miss_concerns"):
        st.caption("Close behind the top 3, within a small statistical margin of the cutoff -- worth "
                   "knowing about, not a missed signal:")
        for f in summary["near_miss_concerns"]:
            nm_cols = st.columns([1, 5])
            nm_cols[0].badge("WATCH", color="orange")
            nm_cols[1].markdown(f"_{f['lob']} ({f['category']})_ — severity {f['severity']}, just outside the top 3.")
        st.divider()

    st.subheader("Top Opportunity")
    for f in summary["top_opportunities"]:
        with st.container(border=True):
            card_cols = st.columns([4, 1])
            with card_cols[0]:
                st.badge("OPPORTUNITY", color="green")
                st.markdown(f"#### {f['lob']}")
                st.caption(f["category"])
            card_cols[1].metric("Severity", f"{f['severity']}")
            st.write(escape_dollar_signs(f["detail"]))
            st.caption(f"Materiality: {escape_dollar_signs(fmt_usd(f['materiality_usd']))}")

    st.subheader("Narrative")
    current_fingerprint = fingerprint(summary, bool(api_key))
    had_cached_before_click = "last_narrative" in st.session_state
    stale_before_click = had_cached_before_click and is_stale(
        st.session_state.last_narrative_fingerprint, summary, bool(api_key)
    )

    button_label = "Regenerate narrative" if had_cached_before_click else "Generate narrative"
    if st.button(button_label, type="primary" if stale_before_click else "secondary"):
        with st.spinner("Writing narrative..."):
            result = agent.generate_narrative(summary, api_key=api_key or None)
        st.session_state.last_narrative = result
        st.session_state.last_narrative_fingerprint = current_fingerprint

    # Everything below is computed fresh AFTER the button block, not reused from the pre-click
    # variables above. This matters for two separate reasons:
    #   1. A click just above may have set last_narrative for the very first time in this same
    #      script run -- reusing the pre-click "had_cached_before_click" would mean the narrative
    #      doesn't actually appear until some later, unrelated rerun.
    #   2. The warning banner must reflect whether the narrative IS stale right now, not whether it
    #      WAS stale before the button was clicked -- showing it from the pre-click variable meant
    #      clicking "Regenerate" while stale would still display the warning on that same run, even
    #      though the click had just fixed it. Real bug, found by actually clicking regenerate while
    #      stale and watching the warning fail to clear.
    if "last_narrative" in st.session_state:
        r = st.session_state.last_narrative
        now_stale = is_stale(st.session_state.last_narrative_fingerprint, summary, bool(api_key))
        if now_stale:
            st.warning(
                "The settings shown above (as-of week, sensitivity, or API key) have changed since "
                "this narrative was generated — it may no longer describe the current findings. "
                "Regenerate above."
            )
        st.caption(f"Source: {r['narrative_source']}" + (" — STALE, settings have changed" if now_stale else ""))
        st.markdown(escape_dollar_signs(r["narrative"]))

with tab_drilldown:
    selected_lob = st.selectbox("Line of business", options=sorted(agent.df["lob"].unique()))
    history_df = agent.df[agent.df["week_ending"] <= pd.Timestamp(as_of_week)]
    lob_df = history_df[history_df["lob"] == selected_lob]

    lob_kpis = {
        "gwp_vs_plan_pct": round(lob_df["gwp_vs_plan_pct"].mean(), 0) if len(lob_df) else None,
        "hit_rate_pct": round(lob_df["hit_rate"].iloc[-1], 0) if len(lob_df) else None,
        "loss_ratio_pct": round(lob_df["attritional_loss_ratio_ytd"].iloc[-1], 1) if len(lob_df) else None,
    }
    glance_cols = st.columns(3)
    glance_cols[0].metric("GWP vs Plan (avg)", f"{lob_kpis['gwp_vs_plan_pct']}%" if lob_kpis["gwp_vs_plan_pct"] is not None else "N/A")
    glance_cols[1].metric("Hit Rate (latest)", f"{lob_kpis['hit_rate_pct']}%" if lob_kpis["hit_rate_pct"] is not None else "N/A")
    glance_cols[2].metric("Loss Ratio (latest)", f"{lob_kpis['loss_ratio_pct']}%" if lob_kpis["loss_ratio_pct"] is not None else "N/A")

    # Peer/portfolio-wide reference points, computed the same way the detection checks compute them --
    # shown in red on each chart so a CUO can see distance from "normal" at a glance, not just the
    # line's own shape.
    peer_pipeline_days = history_df.groupby("lob")["avg_days_in_pipeline"].mean().mean()
    lob_hit_rate = lob_df["hit_rate"].reset_index(drop=True)
    own_baseline_hit_rate = (
        lob_hit_rate.iloc[: config.HIT_RATE_BASELINE_WEEKS].mean()
        if len(lob_hit_rate) >= config.HIT_RATE_BASELINE_WEEKS else None
    )

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(lob_df, x="week_ending", y="gwp_vs_plan_pct", markers=True, title="GWP vs Plan (%)")
        fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="Plan (100%)")
        st.plotly_chart(fig, use_container_width=True)

        fig2 = px.line(lob_df, x="week_ending", y="hit_rate", markers=True, title="Hit Rate (%)")
        if own_baseline_hit_rate is not None:
            fig2.add_hline(y=own_baseline_hit_rate, line_dash="dash", line_color="red",
                            annotation_text=f"This line's own baseline ({own_baseline_hit_rate:.0f}%)")
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.line(lob_df, x="week_ending", y="attritional_loss_ratio_ytd", markers=True, title="Loss Ratio YTD (%)")
        fig3.add_hline(y=config.LOSS_RATIO_TARGET, line_dash="dash", line_color="red",
                       annotation_text=f"Target ({config.LOSS_RATIO_TARGET:.0f}%)")
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.line(lob_df, x="week_ending", y="avg_days_in_pipeline", markers=True, title="Avg Days in Pipeline")
        fig4.add_hline(y=peer_pipeline_days, line_dash="dash", line_color="red",
                       annotation_text=f"Peer avg ({peer_pipeline_days:.1f}d)")
        st.plotly_chart(fig4, use_container_width=True)

    relevant = [f for f in summary["all_concerns"] + summary["all_opportunities"] if f["lob"] == selected_lob]
    st.subheader(f"What's happening with {selected_lob}")

    # Cached per (lob, as-of-week, sensitivity, whether a key is set) so switching back and forth
    # between lines doesn't re-fire an LLM call every time -- generated automatically (no button) since
    # this is meant to read immediately when you click into a line, but still avoids redundant calls.
    cache_key = (selected_lob, as_of_week, min_sustained_weeks, bool(api_key))
    if "lob_narrative_cache" not in st.session_state:
        st.session_state.lob_narrative_cache = {}
    if cache_key not in st.session_state.lob_narrative_cache:
        with st.spinner(f"Writing a note on {selected_lob}..."):
            st.session_state.lob_narrative_cache[cache_key] = agent.writer.write_lob_narrative(
                selected_lob, relevant, lob_kpis, api_key=api_key or None
            )
    lob_text, lob_source = st.session_state.lob_narrative_cache[cache_key]
    st.caption(f"Source: {lob_source}")
    st.markdown(escape_dollar_signs(lob_text))

with tab_chat:
    if "assistant" not in st.session_state:
        st.session_state.assistant = DataAssistant()

    if not st.session_state.assistant.history:
        with st.container(border=True):
            st.markdown("**Ask about this week's data.** Scoped to the current findings and KPIs — "
                        "it explains what was found, it never recommends an underwriting action.")
            qcols = st.columns(3)
            quick_questions = ["What are the top concerns this week?", "What does the Cyber finding mean?",
                                "What's the difference between severity and materiality?"]
            for col, q in zip(qcols, quick_questions):
                if col.button(q, use_container_width=True):
                    st.session_state.pending_question = q

    # All existing history renders first ...
    for msg in st.session_state.assistant.history:
        st.chat_message(msg["role"]).write(escape_dollar_signs(msg["content"]))

    # ... and chat_input is called LAST, so it's visually below every message in the script's render
    # order. Note: Streamlit does not pin chat_input to the bottom of the page when it's placed inside
    # st.tabs() (a known, currently-open Streamlit limitation, confirmed against Streamlit's own GitHub
    # issue tracker, not something fixable purely from application code) -- so "below the messages" is
    # the achievable guarantee here, not "fixed to the screen regardless of scroll position."
    question = st.chat_input("Ask about this week's data") or st.session_state.pop("pending_question", None)

    if question:
        # Show the user's message and a visible loading state immediately, THEN rerun once the answer
        # is ready. ask() already appends both messages to assistant.history internally, so the rerun's
        # history loop (above) picks up this exact exchange and renders it in its correct chronological
        # position -- above chat_input, not below it -- rather than appending it inline here, which
        # would otherwise render below the already-called chat_input on this same pass.
        st.chat_message("user").write(escape_dollar_signs(question))
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.session_state.assistant.ask(question, summary, api_key=api_key or None)
        st.rerun()

with tab_raw:
    st.subheader("Structured output (JSON)")
    st.json(summary, expanded=False)
    st.subheader("Prompts")
    for fname in ["system_prompt.txt", "narrative_user_prompt.txt", "chat_system_prompt.txt"]:
        with st.expander(fname):
            st.code((agent.writer.prompt_dir / fname).read_text(), language="text")