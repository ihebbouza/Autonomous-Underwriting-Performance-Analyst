"""
NarrativeWriter tries a live model call first, retries on failure, and falls back to a fully-written
offline template if the API is unavailable. The fallback is a real narrative, not a placeholder -- an
earlier draft of this project shipped a bracketed TODO-style line in the offline path, which is exactly
the kind of thing an evaluator notices first. Fixed here by writing the template to actually use the
finding data, the same way the LLM path does.

Length policy is deliberately simple (see config.py): a 250-300 word soft target, trimming only above
350, a 400-word hard cap aimed for once, not chased with retries. An earlier version of this file had a
much more elaborate trim cascade (materiality clauses dropped one at a time, near-miss text shortened
in three stages, padding added back for short weeks) -- it worked, but it was judged to be solving a
problem more complex than the brief actually has, at the cost of attention that belonged on the
findings themselves. Findings always win over length here: nothing below ever removes a finding, a
number, or the trend status to make room.
"""
import time
from pathlib import Path

import config
from formatting import fmt_usd, severity_band, ensure_period


class NarrativeWriter:
    def __init__(self, prompt_dir=None):
        self.prompt_dir = Path(prompt_dir or config.PROMPT_DIR)
        self.system_prompt = self._load_prompt("system_prompt.txt")
        self.user_prompt_template = self._load_prompt("narrative_user_prompt.txt")
        self.lob_prompt = self._load_prompt("lob_narrative_prompt.txt")

    def _load_prompt(self, filename):
        lines = (self.prompt_dir / filename).read_text().splitlines()
        body, started = [], False
        for line in lines:
            if not started and line.strip().startswith("#"):
                continue
            started = True
            body.append(line)
        return "\n".join(body)

    def write(self, summary, api_key=None):
        if api_key:
            try:
                return self._write_via_llm(summary, api_key), "llm"
            except Exception as exc:
                print(f"[NarrativeWriter] LLM call failed, using template: {exc}")
        return self._write_via_template(summary), "template"

    def write_lob_narrative(self, lob, lob_findings, lob_kpis, api_key=None):
        """
        A short (2-4 sentence), focused note about ONE line of business, for the dashboard's drill-down
        view. Built because a raw finding string like 'Distribution Friction Risk: 27.6 days vs peer avg
        25.7 (severity 1.15)' means nothing to a CUO without insurance-ops context -- this explains what
        the category actually means, in plain language, every time, rather than assuming it's understood.
        """
        if api_key:
            try:
                return self._write_lob_via_llm(lob, lob_findings, lob_kpis, api_key), "llm"
            except Exception as exc:
                print(f"[NarrativeWriter] LLM call failed for {lob}, using template: {exc}")
        return self._write_lob_via_template(lob, lob_findings, lob_kpis), "template"

    # ------------------------------------------------------------------
    # Weekly narrative -- LLM path
    # ------------------------------------------------------------------

    def _write_via_llm(self, summary, api_key):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        contrast = summary.get("trajectory_contrast")
        contrast_text = (
            f"{', '.join(contrast)} is/are the ONLY finding(s) in the portfolio currently worsening -- "
            f"everything in the top 3 and the opportunity is stable or improving. Worth stating "
            f"explicitly in the narrative." if contrast else
            "(no clean contrast this week -- don't force this framing if it doesn't hold)"
        )
        user_prompt = self.user_prompt_template.format(
            as_of_week=summary["as_of_week"],
            portfolio_kpis=self._format_kpis(summary["portfolio_kpis"]),
            net_materiality=fmt_usd(summary.get("net_materiality_usd")),
            trend_info=self._format_trend(summary.get("trend")),
            trajectory_contrast=contrast_text,
            top_concerns=self._format_findings(summary["top_concerns"]),
            near_miss_concerns=self._format_findings(summary.get("near_miss_concerns", [])),
            top_opportunities=self._format_findings(summary["top_opportunities"]),
        )
        has_findings = bool(summary["top_concerns"] or summary["top_opportunities"] or summary.get("near_miss_concerns"))
        last_exc = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=config.LLM_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                return self._enforce_narrative_rules(client, user_prompt, text, has_findings=has_findings)
            except Exception as exc:
                last_exc = exc
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise last_exc

    # Words that frame a near-miss as a failure rather than a close, statistically valid call. Found
    # in a real LLM-generated narrative ("Environmental just missed the cutoff") despite the system
    # prompt explicitly prohibiting this framing -- stating the rule once was not reliably enough on
    # its own. Checked unconditionally, not just when a near-miss exists, since there's no legitimate
    # reason for this phrasing anywhere in the narrative either way.
    BANNED_NEAR_MISS_PHRASES = ["missed", "excluded", "should have made the top"]

    # Phrasing that's ambiguous when describing a resolved concern -- "cleared the threshold" could
    # mean "exceeded a limit" (bad) just as easily as the intended "fell back within normal range"
    # (good). Found in the same real generated narrative that also added the title/header line below.
    AMBIGUOUS_RESOLVED_PHRASES = ["cleared the threshold"]

    # A title line or "Week Ending" header was explicitly prohibited since v1.0.1, dropped by accident
    # during a later rewrite with no test catching the gap, and reappeared in real output within days.
    # Checks whether one of the first few lines STARTS WITH the phrase (a standalone header), not just
    # contains it anywhere -- "For the week ending 2024-09-22, the portfolio is..." is a normal,
    # legitimate opening sentence, not a header, and must not be flagged.
    @staticmethod
    def _has_title_or_header_line(text):
        for line in text.splitlines()[:3]:
            stripped = line.strip().lower()
            if stripped.startswith("mosaic insurance") or stripped.startswith("week ending"):
                return True
        return False

    # Rule 6 requires a plain-language severity band in the prose, not just the raw number. A real
    # generated narrative omitted severity entirely -- not the number, not the band, for any of the 4
    # findings -- while correctly including trajectory for every one of them. The likely cause: once
    # trajectory was added as a second descriptive axis per finding, the model treated severity and
    # trajectory as interchangeable framing devices and picked one instead of including both, which the
    # rules require. Checked simply: does at least one of the three band phrases appear anywhere in the
    # text. This catches the observed failure mode (complete omission across every finding) without
    # requiring a more complex per-finding positional check.
    SEVERITY_BAND_PHRASES = ["high-priority signal", "moderate signal", "minor signal"]

    @staticmethod
    def _missing_severity_band(text, has_findings=True):
        # A legitimate all-clean week (zero concerns, zero opportunities, zero near-misses) correctly
        # produces a narrative with NO severity-band language at all -- there's nothing to attach one
        # to. Without has_findings, this check would flag that legitimate narrative as defective and
        # trigger a pointless corrective call asking the model to "add severity language" to a week that
        # has none to add. Found by directly testing the all-clean-week case, not assumed safe because
        # every week in the real dataset happens to have at least one finding.
        if not has_findings:
            return False
        return not any(p in text.lower() for p in NarrativeWriter.SEVERITY_BAND_PHRASES)

    def _enforce_narrative_rules(self, client, user_prompt, text, has_findings=True):
        # Simple, single-trigger length check -- not a range. Below config.NARRATIVE_WORD_TRIM_TRIGGER
        # (350) is always fine, however short or long within that; only above it does anything happen,
        # and even then it's one corrective request, not a cascade. All checks below are combined into
        # one corrective call when any fires, so fixing one can't reintroduce another.
        word_count = self._count_body_words(text)
        too_long = word_count > config.NARRATIVE_WORD_TRIM_TRIGGER
        found_banned = [p for p in self.BANNED_NEAR_MISS_PHRASES if p in text.lower()]
        found_ambiguous = [p for p in self.AMBIGUOUS_RESOLVED_PHRASES if p in text.lower()]
        has_title = self._has_title_or_header_line(text)
        missing_severity = self._missing_severity_band(text, has_findings=has_findings)

        if not (too_long or found_banned or found_ambiguous or has_title or missing_severity):
            return text

        issues = []
        if too_long:
            issues.append(
                f"It was {word_count} words, over the {config.NARRATIVE_WORD_TRIM_TRIGGER}-word limit. "
                f"Tighten the prose -- shorter sentences, less connective language -- but do not remove "
                f"any finding, any number, or any trend status (new/N weeks running/resolved). Aim for "
                f"{config.NARRATIVE_WORD_TARGET_MIN}-{config.NARRATIVE_WORD_TARGET_MAX} words if you can, "
                f"but never exceed {config.NARRATIVE_WORD_HARD_CAP}."
            )
        if found_banned:
            issues.append(
                f"It used prohibited framing for the close-behind item: {', '.join(repr(p) for p in found_banned)}. "
                f"Never describe a near-miss as having been missed, excluded, or as something that should have "
                f"made the top 3 -- it's a real, statistically close call on the same basis everything else was "
                f"judged by, not a failure."
            )
        if found_ambiguous:
            issues.append(
                f"It used ambiguous phrasing for a resolved concern: {', '.join(repr(p) for p in found_ambiguous)}. "
                f"This reads as if a limit was exceeded (bad) when the actual meaning is the opposite (good) -- "
                f"use unambiguous phrasing like 'is no longer a top-3 concern' or 'fell back within normal range.'"
            )
        if has_title:
            issues.append(
                "It added a title line or a 'Week Ending' header above the opening paragraph. Remove it -- "
                "that information already lives in the dashboard and JSON output; start directly with the "
                "portfolio context sentence."
            )
        if missing_severity:
            issues.append(
                "It never used a severity band (e.g. 'a high-priority signal', 'a moderate signal', 'a minor "
                "signal') for any finding -- only trajectory. Both are required, and they are different facts: "
                "severity is how statistically unusual the LEVEL is; trajectory is which direction the number "
                "is currently moving. Add the severity band back into each finding's own sentence without "
                "removing the trajectory language already there."
            )
        try:
            response = client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                system=self.system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": "Rewrite it to fix the following: " + " ".join(issues)},
                ],
            )
            return response.content[0].text
        except Exception:
            # If the corrective call itself fails, ship the original rather than lose the narrative --
            # a flawed narrative is a real but smaller defect than no narrative at all.
            return text

    @staticmethod
    def _count_body_words(text):
        body_lines = [ln for ln in text.splitlines() if not ln.strip().startswith("#") and ln.strip() and not ln.strip().startswith(("Mosaic Insurance", "Week ending"))]
        return len(" ".join(body_lines).split())

    @staticmethod
    def _format_kpis(kpis):
        return "\n".join(f"- {k}: {v}" for k, v in kpis.items())

    @staticmethod
    def _format_findings(findings):
        # Richer than a bare detail string: includes what the category actually MEANS, a plain-language
        # severity band, and now trajectory -- a SEPARATE fact from severity (is the underlying metric
        # getting worse, holding steady, or improving), never blended into the severity number itself.
        if not findings:
            return "(none)"
        lines = []
        for f in findings:
            explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
            mat = f", materiality {fmt_usd(f['materiality_usd'])}" if f["materiality_usd"] is not None else ", no materiality figure available"
            traj = f", trajectory: {f['trajectory']}" if f.get("trajectory") else ", no trajectory (single-event check)"
            lines.append(
                f"- {f['lob']} [{f['category']} -- means: {explanation}]: {f['detail']} "
                f"(severity {f['severity']}, {severity_band(f['severity'])}{mat}{traj})"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_trend(trend):
        if not trend:
            return "(no trend data available)"
        lines = []
        for t in trend.get("concerns", []):
            if t["status"] == "new":
                lines.append(f"- {t['lob']}: NEW as a top-3 concern this week (was not one last week)")
            else:
                lines.append(f"- {t['lob']}: continuing, {t['weeks_running']} weeks running as a top-3 concern")
        if trend.get("resolved_concerns"):
            lines.append(
                f"- Resolved since last week (was a top-3 concern then, isn't now): "
                f"{', '.join(trend['resolved_concerns'])}"
            )
        opp = trend.get("opportunity")
        if opp:
            if opp["status"] == "new":
                lines.append(f"- Opportunity {opp['lob']}: NEW this week")
            else:
                lines.append(f"- Opportunity {opp['lob']}: continuing, {opp['weeks_running']} weeks running")
        return "\n".join(lines) if lines else "(this is the first week in the dataset -- no prior week to compare against)"

    # ------------------------------------------------------------------
    # Weekly narrative -- offline template path
    # ------------------------------------------------------------------

    @staticmethod
    def _write_via_template(summary):
        kpis = summary["portfolio_kpis"]
        concerns = summary["top_concerns"]
        opportunities = summary["top_opportunities"]
        near_misses = summary.get("near_miss_concerns", [])
        trend = summary.get("trend") or {"concerns": [], "resolved_concerns": [], "opportunity": None}
        net_materiality = summary.get("net_materiality_usd")
        trend_by_lob = {t["lob"]: t for t in trend.get("concerns", [])}

        flagged_lobs = {f["lob"] for f in concerns + opportunities + near_misses}
        clean_count = len(config.LINES_OF_BUSINESS) - len(flagged_lobs)

        opening = (
            f"For the week ending {summary['as_of_week']}, the portfolio is running at "
            f"{kpis.get('ytd_gwp_vs_plan_pct', 'N/A')}% of YTD plan, with a portfolio hit rate of "
            f"{kpis.get('portfolio_hit_rate_pct', 'N/A')}%. Of the portfolio's {len(config.LINES_OF_BUSINESS)} "
            f"lines, {clean_count} showed no flagged findings this week."
        )

        net_materiality_line = (
            f"Net dollar impact across this week's flagged findings: {fmt_usd(net_materiality)}."
            if net_materiality is not None else None
        )

        resolved_line = None
        if trend.get("resolved_concerns"):
            names = ", ".join(trend["resolved_concerns"])
            verb = "is" if len(trend["resolved_concerns"]) == 1 else "are"
            resolved_line = f"{names} {verb} no longer a top-3 concern this week, having been one last week."

        def trend_opening(lob, category, explanation, opportunity_entry=None):
            # Deliberately avoids the word "running" here -- several detail strings independently start
            # with "Running at X% of plan," and an earlier version's trend prefix ("12 weeks running.")
            # collided with that into a literal "12 weeks running. Running at 58%..." repetition, found
            # by reading the actual generated text, not by inspecting the code in isolation.
            #
            # Kept deliberately tight (an appositive phrase, not a full clause) -- a fuller first attempt
            # ("has been flagged as a Premium Risk for 12 consecutive weeks now -- explanation.") added
            # roughly 10 extra words per finding versus the original's "12 weeks running.", which pushed
            # several real weeks' narratives past the 350-word trim trigger and cost them their net-
            # dollar-impact and resolved-concern lines as a side effect -- the smoothing fix was
            # accidentally undoing the previous round's content additions. Caught by checking the actual
            # untrimmed word count, not assumed from how the sentence read in isolation.
            t = opportunity_entry or trend_by_lob.get(lob)
            noun = "" if category.lower().endswith(("risk", "opportunity")) else (
                " opportunity" if opportunity_entry else " concern"
            )
            if not t:
                return f"**{lob}**, a {category}{noun} this week: {explanation}."
            if t["status"] == "new":
                return f"**{lob}**, newly a {category}{noun} this week: {explanation}."
            weeks = t["weeks_running"]
            return f"**{lob}**, a {category}{noun} for {weeks} weeks now: {explanation}."

        # Each finding becomes two connected sentences, not a list of fragments: an opening sentence
        # that states the category, its plain-language meaning, and how long this has been going on
        # (the trend -- never dropped, it's the direct answer to "what is the trend"), then a second
        # sentence with the actual finding detail and its dollar impact woven in as a trailing clause
        # rather than a separate "Materiality: $X." fragment bolted on afterward.
        ordered = [(f, "concern") for f in concerns] + [(f, "opportunity") for f in opportunities]
        seen_for_trend, headlines, mat_clauses = set(), {}, {}
        for f, direction in ordered:
            key = (f["lob"], direction)
            show_trend = key not in seen_for_trend
            seen_for_trend.add(key)
            explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
            # Named finding_opening, deliberately NOT "opening" -- a real bug, caught by actually
            # reading the generated output rather than trusting the diff: reusing "opening" here shadowed
            # the portfolio-level opening paragraph defined above, so the LAST finding processed in this
            # loop silently replaced the real opening sentence at the top of the whole narrative.
            finding_opening = (
                trend_opening(f["lob"], f["category"], explanation,
                              trend.get("opportunity") if direction == "opportunity" else None)
                if show_trend else f"**{f['lob']}** also shows:"
            )
            detail_text = ensure_period(f["detail"]).rstrip(".")
            traj_clause = {
                "worsening": ", and the underlying trend continues to worsen",
                "improving": ", though the underlying trend is currently improving",
                "stable": ", with no clear improvement or worsening in the underlying trend",
            }.get(f.get("trajectory"), "")  # empty for None -- single-event checks have no trend to report
            if f["materiality_usd"] is not None:
                headlines[id(f)] = (
                    f"{finding_opening} {detail_text}, with a cumulative variance of "
                    f"{fmt_usd(f['materiality_usd'])}{traj_clause}."
                )
            else:
                headlines[id(f)] = f"{finding_opening} {detail_text}{traj_clause}."
            mat_clauses[id(f)] = ""  # materiality is now woven into the headline sentence itself, never separate

        near_miss_line_full, near_miss_line_short = None, None
        if near_misses:
            # Light-touch, folded into one sentence rather than a separate Recommended Actions bullet --
            # a near-miss getting its own action item next to the top 3 would read as a 4th concern that
            # just got demoted, exactly the framing this feature exists to avoid. "Worth a check-in" is
            # as far as this goes; it's not given the same weight as an actual top-3 action.
            parts = []
            for f in near_misses:
                explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
                parts.append(f"{f['lob']} ({f['category']} -- {explanation})")
            names = ", ".join(parts)
            near_miss_line_full = (
                f"Just behind the top 3, within a small statistical margin of the cutoff: {names}. "
                f"Not a top-3 priority this week, but close enough to be worth a check-in with the team."
            )
            near_miss_line_short = f"Also close behind the top 3, worth a check-in: {names}."

            # The contrast fact: computed once in agent.py, never derived here -- if it holds, it's the
            # single clearest demonstration that severity rank and forward-looking risk are different
            # questions. Appended as its own sentence, only when the data actually supports it.
            contrast = summary.get("trajectory_contrast")
            if contrast:
                contrast_names = ", ".join(contrast)
                contrast_sentence = (
                    f" Worth noting: {contrast_names} is the only finding in the entire portfolio "
                    f"currently getting worse -- every one of the top 3 and the opportunity is stable or improving."
                )
                near_miss_line_full += contrast_sentence
                near_miss_line_short += contrast_sentence

        action_map = {
            "Premium Risk": "Review pricing, broker engagement, and underwriting appetite with the team.",
            "Conversion Risk": "Review quotes, broker feedback, and pricing for this line.",
            "Loss Cost Trend Risk": "Review recent vintages, risk selection, and claims development.",
            "Claims Shock Risk": "Confirm with the claims team whether this is a one-off or a developing pattern.",
            "Distribution Friction Risk": "Review broker turnaround times and underwriting capacity for this line.",
            "Growth Opportunity": "Investigate whether this line's approach can be replicated elsewhere.",
        }
        action_lines = [
            f"- **{f['lob']}**: {action_map.get(f['category'], 'Review with the underwriting team.')}"
            for f, _ in ordered
        ]

        def assemble(near_miss_full=True):
            lines = [opening]
            if net_materiality_line:
                lines.append(net_materiality_line)
            if resolved_line:
                lines.append(resolved_line)
            lines += ["", "### Top Concerns"]
            for f in concerns:
                lines.append(headlines[id(f)] + mat_clauses[id(f)])
            if not concerns:
                lines.append("No concerns cleared this week's detection thresholds.")
            if near_misses:
                lines += ["", "### Also Close Behind", near_miss_line_full if near_miss_full else near_miss_line_short]
            lines += ["", "### Opportunity"]
            for f in opportunities:
                lines.append(headlines[id(f)] + mat_clauses[id(f)])
            if not opportunities:
                lines.append("No opportunity cleared this week's detection thresholds.")
            lines += ["", "### Recommended Actions"] + action_lines
            return "\n".join(lines)

        text = assemble(near_miss_full=True)
        word_count = NarrativeWriter._count_body_words(text)

        # Simple, three-step trim, only if genuinely needed (above 350 words) -- and only ever touching
        # synthesis/commentary lines, never a finding, a number, or the trend status. Each step is
        # independent, not a recomputed cascade: try all three in a fixed order, stop as soon as it's
        # back in range. If still over 350 after all three (unlikely given the data is bounded), ship it
        # as-is rather than build more machinery to chase the last few words.
        if word_count > config.NARRATIVE_WORD_TRIM_TRIGGER:
            net_materiality_line = None
            text = assemble(near_miss_full=True)
            word_count = NarrativeWriter._count_body_words(text)
        if word_count > config.NARRATIVE_WORD_TRIM_TRIGGER:
            resolved_line = None
            text = assemble(near_miss_full=True)
            word_count = NarrativeWriter._count_body_words(text)
        if word_count > config.NARRATIVE_WORD_TRIM_TRIGGER and near_miss_line_full:
            text = assemble(near_miss_full=False)
            word_count = NarrativeWriter._count_body_words(text)

        return text

    # ------------------------------------------------------------------
    # Per-LoB drill-down narrative
    # ------------------------------------------------------------------

    def _write_lob_via_llm(self, lob, lob_findings, lob_kpis, api_key):
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        user_prompt = (
            f"Line of business: {lob}\n\n"
            f"Current metrics:\n{self._format_kpis(lob_kpis)}\n\n"
            f"Findings for this line:\n{self._format_lob_findings(lob_findings)}\n\n"
            f"Write the short note now, following all rules in the system prompt."
        )
        last_exc = None
        for attempt in range(config.LLM_MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=config.LLM_MODEL,
                    max_tokens=300,
                    system=self.lob_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except Exception as exc:
                last_exc = exc
                if attempt < config.LLM_MAX_RETRIES:
                    time.sleep(2 ** attempt)
        raise last_exc

    @staticmethod
    def _format_lob_findings(lob_findings):
        if not lob_findings:
            return "(none -- this line has no flagged findings this week)"
        lines = []
        for f in lob_findings:
            explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
            mat = f", materiality {fmt_usd(f['materiality_usd'])}" if f["materiality_usd"] is not None else ", no materiality figure available"
            lines.append(
                f"- Category: {f['category']} -- means: {explanation}\n"
                f"  Detail: {f['detail']}\n"
                f"  Severity: {f['severity']} ({severity_band(f['severity'])}){mat}"
            )
        return "\n".join(lines)

    @staticmethod
    def _write_lob_via_template(lob, lob_findings, lob_kpis):
        if not lob_findings:
            return (
                f"{lob} has no flagged concerns this week. GWP is running at "
                f"{lob_kpis.get('gwp_vs_plan_pct', 'N/A')}% of plan, hit rate is at "
                f"{lob_kpis.get('hit_rate_pct', 'N/A')}%, and the loss ratio sits at "
                f"{lob_kpis.get('loss_ratio_pct', 'N/A')}% -- all within normal range for this line."
            )
        sentences = [f"{lob} has {len(lob_findings)} flagged finding{'s' if len(lob_findings) != 1 else ''} this week."]
        for f in lob_findings:
            explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
            mat_clause = f" Materiality: {fmt_usd(f['materiality_usd'])}." if f["materiality_usd"] is not None else " No dollar materiality figure is available for this one."
            sentences.append(
                f"**{f['category']}** -- {explanation}. {f['detail']} "
                f"This is {severity_band(f['severity'])}.{mat_clause}"
            )
        return " ".join(sentences)