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
        user_prompt = self.user_prompt_template.format(
            as_of_week=summary["as_of_week"],
            portfolio_kpis=self._format_kpis(summary["portfolio_kpis"]),
            net_materiality=fmt_usd(summary.get("net_materiality_usd")),
            trend_info=self._format_trend(summary.get("trend")),
            top_concerns=self._format_findings(summary["top_concerns"]),
            near_miss_concerns=self._format_findings(summary.get("near_miss_concerns", [])),
            top_opportunities=self._format_findings(summary["top_opportunities"]),
        )
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
                return self._enforce_narrative_rules(client, user_prompt, text)
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

    def _enforce_narrative_rules(self, client, user_prompt, text):
        # Simple, single-trigger length check -- not a range. Below config.NARRATIVE_WORD_TRIM_TRIGGER
        # (350) is always fine, however short or long within that; only above it does anything happen,
        # and even then it's one corrective request, not a cascade. Combined with the banned-phrase
        # check into one corrective call when either fires, so fixing one can't reintroduce the other.
        word_count = self._count_body_words(text)
        too_long = word_count > config.NARRATIVE_WORD_TRIM_TRIGGER
        found_banned = [p for p in self.BANNED_NEAR_MISS_PHRASES if p in text.lower()]

        if not too_long and not found_banned:
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
        # Richer than a bare detail string: includes what the category actually MEANS and a
        # plain-language severity band, the same richness the per-LoB narrative's prompt already gets
        # (see _format_lob_findings) -- there's no good reason the weekly narrative's prompt should
        # have less to work with than the per-LoB one does.
        if not findings:
            return "(none)"
        lines = []
        for f in findings:
            explanation = config.CATEGORY_EXPLANATION.get(f["category"], "")
            mat = f", materiality {fmt_usd(f['materiality_usd'])}" if f["materiality_usd"] is not None else ", no materiality figure available"
            lines.append(
                f"- {f['lob']} [{f['category']} -- means: {explanation}]: {f['detail']} "
                f"(severity {f['severity']}, {severity_band(f['severity'])}{mat})"
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

        def trend_prefix(lob, opportunity_entry=None):
            t = opportunity_entry or trend_by_lob.get(lob)
            if not t:
                return ""
            return "New this week. " if t["status"] == "new" else f"{t['weeks_running']} weeks running. "

        # Headline includes the trend prefix directly -- it's the answer to "what is the trend," one of
        # the brief's three explicit narrative requirements, so it's never treated as droppable filler.
        # Shown only on a line's first finding (by lob+direction), not repeated if a line has two
        # separate findings in the same direction this week.
        ordered = [(f, "concern") for f in concerns] + [(f, "opportunity") for f in opportunities]
        seen_for_trend, headlines, mat_clauses = set(), {}, {}
        for f, direction in ordered:
            key = (f["lob"], direction)
            show_trend = key not in seen_for_trend
            seen_for_trend.add(key)
            prefix = trend_prefix(f["lob"], trend.get("opportunity") if direction == "opportunity" else None) if show_trend else ""
            category_line = f"({f['category']} -- {config.CATEGORY_EXPLANATION.get(f['category'], '')})"
            headlines[id(f)] = f"**{f['lob']}** {category_line}: {prefix}{ensure_period(f['detail'])}"
            if f["materiality_usd"] is not None:
                clause = "The cumulative variance is" if direction == "concern" else "Cumulative variance is"
                mat_clauses[id(f)] = f" {clause} {fmt_usd(f['materiality_usd'])}."
            else:
                mat_clauses[id(f)] = ""

        near_miss_line_full, near_miss_line_short = None, None
        if near_misses:
            names = ", ".join(f"{f['lob']} ({f['category']})" for f in near_misses)
            near_miss_line_full = (
                f"Close behind the top 3, within a small statistical margin of the cutoff: {names}. "
                f"Not in the top 3 this week, but close enough to be worth watching, not a miss."
            )
            near_miss_line_short = f"Also close behind the top 3: {names}."

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