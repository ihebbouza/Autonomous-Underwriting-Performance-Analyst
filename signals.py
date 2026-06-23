"""
SignalDetector answers two genuinely different questions, kept deliberately separate:

    Layer 1 -- DETECTION: "is this metric unusual, by an independently-defined standard?"
    Layer 2 -- NORMALIZATION: "how does this compare to other findings, on a common scale?"

An earlier version of this project's detection logic (built directly into the analysis notebook before
this module existed) answered both questions with a single hand-weighted formula, tuned by trial and
error until it reproduced a known answer. That is the one thing this design is built to avoid. Every
check below is independent: it has its own threshold, justified in config.py, and produces a raw
magnitude for every line of business -- not just the ones that trip the threshold. Severity is then a
peer z-score (how many standard deviations a line's raw magnitude sits from the cross-sectional average
for that same check, across all 8 lines), which makes findings from different checks comparable without
any hand-picked weight. See README for the full reasoning and the two bugs this approach caught.

Materiality (dollar impact) is computed separately again, and is NOT blended into severity -- they
answer different questions ("how unusual" vs. "how much money"), and forcing them into one number would
just reintroduce the same bias one level up. See AnalystAgent for how the two are presented together.
"""
import numpy as np
import pandas as pd

import config


class SignalDetector:
    def __init__(self, min_sustained_weeks=3):
        # The one slider exposed to the dashboard. It controls how many of the recent hit-rate weeks
        # must individually sit below baseline -- the literal scenario in the brief's first probing
        # question (a one-week dip vs. a sustained collapse). Other checks have their own fixed,
        # non-adjustable "sustained" criteria (see config.py) since the brief's example is specifically
        # about hit rate.
        self.min_sustained_weeks = min_sustained_weeks

    # ------------------------------------------------------------------
    # Layer 1: independent checks. Each returns a pd.Series of raw magnitude, indexed by lob, for
    # EVERY line of business -- not just the ones that will eventually pass threshold. The full
    # cross-section is needed for peer normalization in Layer 2.
    # ------------------------------------------------------------------

    @staticmethod
    def _history(df, lob, as_of_week):
        h = df[(df["lob"] == lob) & (df["week_ending"] <= as_of_week)].sort_values("week_ending")
        return h

    def _gwp_band_raw(self, df, as_of_week):
        concern_mag, opp_mag, below_frac, above_frac, avg = {}, {}, {}, {}, {}
        for lob in config.LINES_OF_BUSINESS:
            h = self._history(df, lob, as_of_week)
            r = h["gwp_vs_plan_pct"]
            avg[lob] = r.mean()
            below_frac[lob] = (r < config.GWP_BAND_LOW).mean()
            above_frac[lob] = (r > config.GWP_BAND_HIGH).mean()
            concern_mag[lob] = max(0.0, config.GWP_BAND_LOW - avg[lob])
            opp_mag[lob] = max(0.0, avg[lob] - config.GWP_BAND_HIGH)
        return (pd.Series(concern_mag), pd.Series(opp_mag), pd.Series(below_frac), pd.Series(above_frac), pd.Series(avg))

    def _hit_rate_raw(self, df, as_of_week):
        # Deliberately NOT clipped at zero: a line with an improving hit rate has a genuinely negative
        # value here, and squashing it to 0 would artificially compress the peer distribution used for
        # normalization in find_all() -- it would shift the cross-sectional mean/std and change every
        # other line's severity, even though nothing about their own numbers changed. The threshold
        # check below only acts on lines that clear HIT_RATE_RELATIVE_DROP, so improving lines never
        # trigger a finding regardless of whether their raw value is negative or floored at 0 -- the
        # clip only matters for the normalization step, which is exactly where it was wrong.
        rel_drop, baseline_s, recent_s, weeks_below = {}, {}, {}, {}
        for lob in config.LINES_OF_BUSINESS:
            h = self._history(df, lob, as_of_week)
            hr = h["hit_rate"]
            baseline = hr.iloc[: config.HIT_RATE_BASELINE_WEEKS].mean() if len(hr) >= config.HIT_RATE_BASELINE_WEEKS else float("nan")
            recent = hr.iloc[-config.HIT_RATE_RECENT_WEEKS:] if len(hr) >= config.HIT_RATE_RECENT_WEEKS else hr.iloc[0:0]
            baseline_s[lob] = baseline
            recent_s[lob] = recent.mean() if len(recent) else float("nan")
            rel_drop[lob] = ((baseline - recent.mean()) / baseline) if baseline and len(recent) else 0.0
            weeks_below[lob] = int((recent < baseline).sum()) if baseline and len(recent) else 0
        return pd.Series(rel_drop), pd.Series(baseline_s), pd.Series(recent_s), pd.Series(weeks_below)

    def _loss_ratio_raw(self, df, as_of_week):
        slope, latest = {}, {}
        for lob in config.LINES_OF_BUSINESS:
            h = self._history(df, lob, as_of_week)
            y = h["attritional_loss_ratio_ytd"].values[-config.LOSS_RATIO_TREND_WINDOW:]
            if len(y) >= 2:
                slope[lob] = float(np.polyfit(np.arange(len(y)), y, 1)[0])
            else:
                slope[lob] = 0.0
            latest[lob] = float(h["attritional_loss_ratio_ytd"].iloc[-1]) if len(h) else float("nan")
        return pd.Series(slope), pd.Series(latest)

    def _claims_raw(self, df, as_of_week):
        z, last_val, baseline_mean = {}, {}, {}
        for lob in config.LINES_OF_BUSINESS:
            h = self._history(df, lob, as_of_week)
            claims = h["new_claims_incurred_est"].values
            if len(claims) < config.CLAIMS_MIN_HISTORY_WEEKS + 1:
                z[lob], last_val[lob], baseline_mean[lob] = 0.0, float("nan"), float("nan")
                continue
            baseline = claims[:-1]
            last = claims[-1]
            std = baseline.std()
            z[lob] = (last - baseline.mean()) / std if std > 0 else 0.0
            last_val[lob] = last
            baseline_mean[lob] = baseline.mean()
        return pd.Series(z), pd.Series(last_val), pd.Series(baseline_mean)

    def _pipeline_raw(self, df, as_of_week):
        days = {}
        for lob in config.LINES_OF_BUSINESS:
            h = self._history(df, lob, as_of_week)
            days[lob] = h["avg_days_in_pipeline"].mean()
        days = pd.Series(days)
        return days - days.mean(), days, days.mean()

    # ------------------------------------------------------------------
    # Trajectory: a SEPARATE axis from severity, never blended into it or into the top-3 ranking --
    # the same reasoning that already keeps materiality separate from severity. Answers "is this
    # finding's underlying metric currently getting worse, holding steady, or getting better," which
    # severity alone cannot, since severity measures how unusual the LEVEL is, not which direction it's
    # moving. Computed as a peer z-score of the SLOPE -- the identical statistical convention already
    # used for severity, just applied to the derivative instead of the level, so this isn't a new kind
    # of judgment call, just the same one applied one level deeper.
    # ------------------------------------------------------------------

    @staticmethod
    def _slope_series(df, as_of_week, column, window=None):
        window = window or config.TREND_WINDOW_WEEKS
        slopes = {}
        for lob in config.LINES_OF_BUSINESS:
            h = SignalDetector._history(df, lob, as_of_week)
            y = h[column].values[-window:]
            slopes[lob] = float(np.polyfit(np.arange(len(y)), y, 1)[0]) if len(y) >= 2 else 0.0
        return pd.Series(slopes)

    @staticmethod
    def _trajectory(slope_series, lob, deteriorating_sign):
        # deteriorating_sign flips the raw slope's peer z-score so that, regardless of which literal
        # direction is bad for THIS metric (a falling ratio is bad for GWP; a climbing one is bad for
        # loss ratio), a positive result always means "worsening relative to peers" and a negative
        # result always means "improving relative to peers" -- one consistent meaning across all checks.
        z = SignalDetector._peer_z(slope_series, lob) * deteriorating_sign
        if z > config.TRAJECTORY_STABLE_MARGIN:
            return "worsening", round(z, 2)
        if z < -config.TRAJECTORY_STABLE_MARGIN:
            return "improving", round(z, 2)
        return "stable", round(z, 2)

    # ------------------------------------------------------------------
    # Layer 2: peer normalization. Severity is always "how many standard deviations from the
    # cross-sectional peer average for this same check" -- the same unit for every check, with no
    # check-specific multiplier to tune.
    # ------------------------------------------------------------------

    @staticmethod
    def _peer_z(series, lob):
        std = series.std()
        return float((series[lob] - series.mean()) / std) if std and std > 0 else 0.0

    # ------------------------------------------------------------------
    # Materiality: computed only where a clean dollar figure is derivable from this data without
    # an unstated assumption. Left as None otherwise -- an honest gap, not a fabricated number.
    # ------------------------------------------------------------------

    @staticmethod
    def _gwp_materiality(df, lob, as_of_week):
        h = SignalDetector._history(df, lob, as_of_week)
        return round(float(h["gwp_variance"].sum()), 0)

    @staticmethod
    def _claims_materiality(last_val, baseline_mean):
        if pd.isna(last_val) or pd.isna(baseline_mean):
            return None
        return round(float(last_val - baseline_mean), 0)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def find_all(self, df, as_of_week=None):
        as_of_week = as_of_week or df["week_ending"].max()
        findings = []

        # CHECK A -- GWP band
        concern_mag, opp_mag, below_frac, above_frac, avg = self._gwp_band_raw(df, as_of_week)
        gwp_slope = self._slope_series(df, as_of_week, "gwp_vs_plan_pct")
        for lob in config.LINES_OF_BUSINESS:
            if below_frac[lob] >= config.GWP_SUSTAINED_FRACTION and concern_mag[lob] > 0:
                traj, traj_z = self._trajectory(gwp_slope, lob, deteriorating_sign=-1)
                findings.append(self._build(lob, "gwp_band", "concern", self._peer_z(concern_mag, lob),
                                             self._gwp_materiality(df, lob, as_of_week),
                                             f"Running at {avg[lob]:.0f}% of plan, with {below_frac[lob]*100:.0f}% of weeks "
                                             f"below the {config.GWP_BAND_LOW:.0f}% line — a sustained shortfall, not a one-off.",
                                             traj, traj_z))
            elif above_frac[lob] >= config.GWP_SUSTAINED_FRACTION and opp_mag[lob] > 0:
                traj, traj_z = self._trajectory(gwp_slope, lob, deteriorating_sign=-1)
                findings.append(self._build(lob, "gwp_band", "opportunity", self._peer_z(opp_mag, lob),
                                             self._gwp_materiality(df, lob, as_of_week),
                                             f"Running at {avg[lob]:.0f}% of plan, with {above_frac[lob]*100:.0f}% of weeks "
                                             f"above the {config.GWP_BAND_HIGH:.0f}% line — sustained overperformance, not a blip.",
                                             traj, traj_z))

        # CHECK B -- Hit rate collapse
        rel_drop, baseline_s, recent_s, weeks_below = self._hit_rate_raw(df, as_of_week)
        # Trajectory uses HIT_RATE_RECENT_WEEKS (4), NOT the generic TREND_WINDOW_WEEKS (6) -- a 6-week
        # window here would straddle the baseline/recent boundary this check itself defines, mixing the
        # magnitude of the original collapse into the slope alongside whatever's happened since. Found
        # by checking the real numbers directly: Cyber's 6-week slope was -2.86 (dominated by the crash
        # itself, weeks 7-8 still at the ~24% baseline), while its actual post-collapse trajectory (the
        # 4 weeks the check's own "recent" window covers) is +1.91 -- recovering, not still declining.
        hit_rate_slope = self._slope_series(df, as_of_week, "hit_rate", window=config.HIT_RATE_RECENT_WEEKS)
        sustained_min = min(self.min_sustained_weeks, config.HIT_RATE_RECENT_WEEKS)
        for lob in config.LINES_OF_BUSINESS:
            if rel_drop[lob] >= config.HIT_RATE_RELATIVE_DROP and weeks_below[lob] >= sustained_min:
                traj, traj_z = self._trajectory(hit_rate_slope, lob, deteriorating_sign=-1)
                findings.append(self._build(lob, "hit_rate_collapse", "concern", self._peer_z(rel_drop, lob), None,
                                             f"Hit rate fell from a {baseline_s[lob]:.0f}% baseline to {recent_s[lob]:.0f}% "
                                             f"recently — a {rel_drop[lob]*100:.0f}% drop, sustained across multiple weeks, not a single bad week.",
                                             traj, traj_z))

        # CHECK C -- Loss ratio trend. Reuses the SAME slope already computed for detection itself
        # (not a separate _slope_series call) -- the check's own trend-detection math and its
        # trajectory classification are, for this one check, measuring the literal same thing.
        slope, latest = self._loss_ratio_raw(df, as_of_week)
        for lob in config.LINES_OF_BUSINESS:
            if slope[lob] > config.LOSS_RATIO_SLOPE_ALERT and latest[lob] > config.LOSS_RATIO_PROXIMITY_FLOOR:
                traj, traj_z = self._trajectory(slope, lob, deteriorating_sign=1)
                findings.append(self._build(lob, "loss_ratio_trend", "concern", self._peer_z(slope, lob), None,
                                             f"Loss ratio climbing {slope[lob]:.1f} points per week, now at {latest[lob]:.1f}% "
                                             f"against the {config.LOSS_RATIO_TARGET:.0f}% target — the trend, not just the level, is the concern.",
                                             traj, traj_z))

        # CHECK D -- Claims anomaly. No trajectory: this check is inherently a single-week-vs-history
        # comparison, not a multi-week trend -- there's no slope concept that meaningfully applies to a
        # one-off shock the same way it does to a sustained level or trend.
        z, last_val, baseline_mean = self._claims_raw(df, as_of_week)
        for lob in config.LINES_OF_BUSINESS:
            if z[lob] > config.CLAIMS_Z_THRESHOLD:
                findings.append(self._build(lob, "claims_anomaly", "concern", self._peer_z(z, lob),
                                             self._claims_materiality(last_val[lob], baseline_mean[lob]),
                                             f"Claims came in at ${last_val[lob]:,.0f} last week, well above the typical "
                                             f"${baseline_mean[lob]:,.0f} — a sharp, one-week spike outside this line's normal range."))

        # CHECK E -- Pipeline friction, peer-relative
        rel_days, days, peer_mean = self._pipeline_raw(df, as_of_week)
        pipeline_slope = self._slope_series(df, as_of_week, "avg_days_in_pipeline")
        peer_std = days.std()
        for lob in config.LINES_OF_BUSINESS:
            if peer_std > 0 and (days[lob] - peer_mean) / peer_std > config.PIPELINE_PEER_STD_THRESHOLD:
                traj, traj_z = self._trajectory(pipeline_slope, lob, deteriorating_sign=1)
                findings.append(self._build(lob, "pipeline_friction", "concern", self._peer_z(rel_days, lob), None,
                                             f"Taking {days[lob]:.1f} days to move through the pipeline, versus a "
                                             f"{peer_mean:.1f}-day average across other lines — a modest gap worth watching, not a priority concern.",
                                             traj, traj_z))

        concerns = sorted([f for f in findings if f["direction"] == "concern"], key=lambda f: f["severity"], reverse=True)
        opportunities = sorted([f for f in findings if f["direction"] == "opportunity"], key=lambda f: f["severity"], reverse=True)

        # A concern just outside the top N, but within NEAR_MISS_SEVERITY_MARGIN of the cutoff, is a
        # genuine near-miss worth naming rather than silently dropping -- the rule is generic (applies
        # to whichever line happens to be close, not a specific one), and concerns are already sorted
        # descending, so the gap only grows once one item fails to qualify; no need to keep checking.
        near_miss_concerns = []
        if len(concerns) > config.TOP_N_CONCERNS:
            cutoff_severity = concerns[config.TOP_N_CONCERNS - 1]["severity"]
            for f in concerns[config.TOP_N_CONCERNS:]:
                if cutoff_severity - f["severity"] <= config.NEAR_MISS_SEVERITY_MARGIN:
                    near_miss_concerns.append(f)
                else:
                    break

        return {
            "as_of_week": str(pd.Timestamp(as_of_week).date()),
            "all_concerns": concerns,
            "all_opportunities": opportunities,
            "top_concerns": concerns[: config.TOP_N_CONCERNS],
            "top_opportunities": opportunities[: config.TOP_N_OPPORTUNITIES],
            "near_miss_concerns": near_miss_concerns,
        }

    @staticmethod
    def _build(lob, check, direction, severity, materiality_usd, detail, trajectory=None, trajectory_z=None):
        return {
            "lob": lob,
            "check": check,
            "direction": direction,
            "category": config.CHECK_CATEGORY.get((check, direction), check),
            "severity": round(severity, 2),
            "materiality_usd": materiality_usd,
            "detail": detail,
            # A separate axis from severity, deliberately -- see the trajectory section above for why.
            # None for claims_anomaly, the one check that's inherently a single-event comparison with
            # no slope concept that applies.
            "trajectory": trajectory,
            "trajectory_z": trajectory_z,
        }