"""
Chart construction pulled out of app.py into plain functions that return a Plotly figure object.
This makes the actual bug-prone logic (axis types, reference lines) directly unit-testable without
going through Streamlit's AppTest, which can't easily inspect a rendered figure's properties.
"""
import pandas as pd
import plotly.express as px


def build_hit_rate_heatmap(trend_df):
    heatmap_df = trend_df.pivot(index="lob", columns="week_ending", values="hit_rate")
    heatmap_df.columns = [pd.Timestamp(c).strftime("%m-%d") for c in heatmap_df.columns]
    fig = px.imshow(heatmap_df, color_continuous_scale="RdYlGn", aspect="auto",
                     labels={"color": "Hit Rate %"}, text_auto=".0f")
    # Without this, Plotly auto-detects the "MM-DD" string labels as dates and re-formats them with
    # its own (wrong) inferred year -- forcing categorical type makes it use the literal labels as-is.
    fig.update_xaxes(type="category")
    fig.update_yaxes(type="category")
    return fig
