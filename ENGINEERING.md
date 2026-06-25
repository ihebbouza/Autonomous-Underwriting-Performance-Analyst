# Mosaic Underwriting Performance Pack — Analyst Agent

An AI-powered agent that replaces the manual weekly CUO pack: it ingests four weekly extracts, finds
what's statistically unusual, separates that from what's financially material, writes a CUO-ready
narrative, and serves all of it through an interactive dashboard — unattended at 6am or live in front
of someone.

This document is the complete reference for the project: what each file does, how the pieces fit
together, the reasoning behind the harder design decisions, and what to read first if you're studying
this codebase rather than just running it.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py                  # unattended path -- writes sample_output/summary.json + narrative.md
streamlit run app.py           # interactive dashboard
python -m pytest               # 64 tests
```

Set `ANTHROPIC_API_KEY` (see `.env.example`) for live-model narratives and the chat assistant. Without
one, narratives fall back to a complete offline template (not a placeholder) and the chat assistant
explains plainly that it needs a key — nothing breaks, nothing is hidden.

---

## If you're studying this codebase, read files in this order

1. **`config.py`** — every threshold and constant, each with its justification. Start here; everything
   else just implements what this file declares.
2. **`data.py`** — the simplest file. Ingest, validate, join. Read this fully before anything else.
3. **`signals.py`** — the core of the project. Two layers: detection (5 independent checks) and
   normalization (peer z-scores). This is the file worth knowing cold.
4. **`narrative.py`** — the longest file, but mechanically repetitive: LLM-with-retry, template
   fallback, and a deterministic word-count enforcement pass, applied twice (once for the weekly
   narrative, once for the per-LoB drill-down note).
5. **`agent.py`** — five lines that matter, ties the above three together.
6. **`app.py`** — the dashboard. Mostly straightforward Streamlit, with two genuinely subtle pieces
   worth understanding: the narrative staleness detection, and why certain `st.markdown()` calls
   escape dollar signs and others don't.
7. **`chat.py`**, **`charts.py`**, **`staleness.py`**, **`formatting.py`** — small, single-purpose
   helper modules, each easy to read end to end in a few minutes.
8. **`run.py`** — the unattended CLI entry point.
9. **`tests/`** — read these alongside the file they test. Several tests exist specifically because
   they caught a real bug during development; those are called out in the file they live in.

---

## Architecture

Five classes, one direction of flow:

```
DataLoader -> SignalDetector -> NarrativeWriter
                                      |
                                      v
                                 AnalystAgent  <-- the only thing run.py and app.py call directly
                                      |
                        -------------------------------
                       |                               |
                    run.py                          app.py --- DataAssistant (dashboard-only)
              (unattended, 6am)                (interactive, owns the chat)
```

`AnalystAgent` creates its own `DataLoader`, `SignalDetector`, `NarrativeWriter` — composition, not
dependency injection. None of the three has an interchangeable alternative implementation in this
project's scope, so nothing is gained by making them swappable.

`DataAssistant` is deliberately **not** owned by `AnalystAgent` and never imported by `run.py`.
Answering free-form questions has no equivalent in an unattended cron job — keeping it structurally
separate means the official report can never be touched by anything chat-related, even by accident.

`analyze()` and `generate_narrative()` are kept as two separate methods on `AnalystAgent`, not one
combined call, because the dashboard needs to re-run detection on every sensitivity-slider or
time-travel movement (cheap, no network call) without re-triggering an LLM call each time. Only
`generate_narrative()` ever touches the network.

---

## The detection methodology: two layers, kept apart on purpose

Earlier in this project's life, detection lived in one weighted formula: normalize several metrics, sum
them with hand-picked weights, take the top 3. That formula was tuned by trial and error until it
reproduced a known answer — circular, since a method built that way has no demonstrated ability to work
on data where the answer isn't already known. `signals.py` exists specifically to avoid that.

**Layer 1 — Detection** (`SignalDetector._check_*` logic, run from `find_all`): *is this metric unusual,
by an independently-defined standard?* Every threshold in `config.py` is sourced from either the brief's
literal wording or a generic statistical/business convention, decided before checking whether it
reproduces anything:

| Check | Threshold | Source |
|---|---|---|
| GWP band | 15% either side of plan, sustained ≥80% of weeks | Common underwriting tolerance convention |
| Hit rate collapse | ≥25% relative drop, ≥3 of last 4 weeks below baseline | Brief's own "final four weeks" wording + standard "meaningful change" convention |
| Loss ratio trend | >1.0pp/week (last 6 weeks, linear fit) AND already >50% | Brief's literal 60% target; slope threshold set well below the brief's own ~3pp/week example, deliberately not fit to it |
| Claims anomaly | z > 2.0 | Standard "statistically unusual" convention |
| Pipeline friction | >1 cross-sectional std above peer average | Standard peer-outlier convention |

No cross-metric weights exist anywhere in this layer.

**Layer 2 — Normalization** (`SignalDetector._peer_z`): findings from different checks are made
comparable by converting each one to a peer z-score — standard deviations from the cross-sectional
average across all 8 lines of business, for that *same* check. Same unit for every check, no hand-picked
multiplier to tune.

**A separate axis — Materiality** (`SignalDetector._gwp_materiality`, `_claims_materiality`): dollar
impact, computed only where this dataset supports a clean figure without an unstated assumption (GWP
cumulative variance, claims shock magnitude). Deliberately **not** blended into severity — "statistically
unusual" and "financially material" are different questions, and forcing them into one number would
reintroduce the same bias one level up. The dashboard and narratives show both, side by side, and the
narrative prompts explicitly instruct against implying one finding is simply "worse" than another just
because it's ranked higher.

**A third, separate axis — Trajectory** (`SignalDetector._trajectory`): is the finding's underlying
metric currently getting worse, holding steady, or improving — answered as a peer z-score of the
*slope*, the identical statistical convention severity already uses for the *level*, just applied one
derivative deeper. Like materiality, **never blended into severity or the ranking**. Built after a
direct question worth taking seriously: does this project distinguish "stable-bad" from
"accelerating-bad"? `loss_ratio_trend` already measured slope for detection purposes, but nothing
compared trajectory *across* findings, and nothing surfaced it as its own fact for the other four
checks. Checking the real data before writing any code turned up something worth saying out loud: every
one of the current top-3 concerns plus the opportunity is stable or improving, and the *only* finding
in the entire portfolio still getting worse is Environmental — which ranks 4th, a near-miss, under pure
severity. `AnalystAgent._compute_trajectory_contrast` states this explicitly when it holds, and
deliberately returns nothing when it doesn't (e.g., if a top-3 finding is also worsening) — a real,
checked fact about a given week, never a framing forced to make a point. `claims_anomaly` has no
trajectory (`None`): it's inherently a single-week-vs-history comparison, not a multi-week trend, and
there's no slope concept that meaningfully applies to a one-off shock.

One real bug caught while building this: `hit_rate_collapse`'s trajectory initially used the same
generic 6-week window as every other check, which silently included the collapse event itself —
Cyber's hit rate sat near baseline for the first 2 of those 6 weeks, then crashed, producing a slope
dominated by the crash rather than by what's happened since. Fixed by using the check's own "recent"
window (`HIT_RATE_RECENT_WEEKS`, 4) for this one check specifically, since that's the period whose
*own* direction is actually the question.

A second, more serious bug, found by directly asking "is this classification robust, or did I get
lucky with the default window?" rather than trusting a single number: Excess Casualty's GWP-vs-plan
trajectory flipped between worsening, stable, and improving depending on whether the slope window was
4, 5, 6, 7, or 8 weeks — its underlying slope is close to flat (~+0.4 points/week), so short-window
noise dominates which direction a single window happens to lean. Environmental's loss-ratio
trajectory, by contrast, stayed "worsening" across every one of those windows, because its slope
(~2.7 points/week) is large enough that noise doesn't flip it. `SignalDetector._trajectory` now
requires a primary window AND a second, shorter window to actually **agree** on direction before
reporting worsening or improving; disagreement is itself the honest answer ("stable" — not enough
signal to call a direction), not a coin-flip dressed up as a fact. This changed Excess Casualty's
reported trajectory from a confident-sounding "improving" to the more honest "stable" — the headline
finding above (Environmental is the only worsening finding) survives this change unaffected, since
"stable" still isn't "worsening."

**Categories** (`config.CHECK_CATEGORY`, `config.CATEGORY_EXPLANATION`): each check maps to a
human-readable risk category (Premium Risk, Conversion Risk, Loss Cost Trend Risk, Claims Shock Risk,
Distribution Friction Risk, Growth Opportunity), and every category has a one-sentence plain-language
explanation. This exists because a label like "Distribution Friction Risk" means nothing to a CUO
without insurance-ops context — the explanation is meant to travel with the label every time it's shown,
never assumed as background knowledge.

### Two real bugs this approach caught, and what they were

1. **Sign-mixing.** An early version of the GWP check put concern and opportunity magnitudes on one
   signed scale before normalizing. The more extreme an opportunity was, the *more negative* its z-score
   became — exactly backwards. Fixed by giving concern and opportunity their own one-directional,
   always-non-negative magnitude series.
2. **A clip-at-zero in the hit-rate check** squashed two genuinely-improving lines (Transactional
   Liability, Professional Lines) to an artificial 0 before normalizing, which skewed the peer
   distribution used to score every other line, including Cyber. Removed the clip; severity now matches
   the project's independently-built analysis notebook exactly.

### On the current top-3 ranking

Under this peer-normalized method, the top 3 concerns are **Excess Casualty, Cyber, Transactional
Liability** — not Environmental, which the assessment brief's own framing names as one of the three.
This is reported as-is, not corrected for. Transactional Liability's one-week claims shock scores
marginally higher than Environmental's six-week loss-ratio climb once both are measured on equal
peer-relative footing. The two carry genuinely different kinds of risk — a magnitude risk whose
persistence is unknown vs. a trend risk that compounds the longer it's ignored — and both the weekly
narrative and the executive framing are written to say so explicitly rather than imply one is simply
worse than the other.

**Near-miss concerns** (`config.NEAR_MISS_SEVERITY_MARGIN`, `SignalDetector.find_all`): a concern just
outside the top 3 is surfaced explicitly if its severity sits within 0.5 (half a standard deviation, a
standard statistical convention) of the cutoff — a generic rule that applies to whichever line happens
to be close, never a hard-coded name. On the real data this currently picks out Environmental (severity
1.76, 0.22 below the cutoff) and correctly excludes Political Risk (severity 1.15, 0.83 below — not a
close call). The narrative mentions a near-miss in one sentence under "### Also Close Behind," explicitly
framed as a statistically close call, never as something that "should have" made the top 3. If both a
near-miss mention and every materiality figure can't fit the word budget in the same week, the near-miss
mention is kept and a lower-priority materiality clause is dropped instead — losing one dollar figure
is a smaller loss than losing the only mention of a real, named finding entirely.

---

## The narrative system

Two narratives exist, both following the same LLM-with-retry-and-template-fallback pattern:

**The weekly narrative** (`NarrativeWriter.write`): the full CUO pack — portfolio context, top 3
concerns, top opportunity, recommended actions. The length policy is deliberately simple, on direct
instruction after an earlier version's trim logic (materiality clauses dropped one at a time, near-miss
text shortened in three stages, padding added back for short weeks) was judged more complex than its
actual value justified: 250-300 words is a soft target, trimming only activates above 350, and 400 is a
hard ceiling aimed for once, not chased with retries. Findings always outrank length — nothing in either
path ever removes a finding, a number, or a trend status to get shorter.
- **LLM path:** checked for two things after the model responds — length over the 350-word trigger, and
  banned near-miss framing ("missed," "excluded") — combined into one corrective follow-up if either
  fires, so fixing one can't reintroduce the other.
- **Template path:** a simple, three-step trim, tried in a fixed order, only above 350 words: drop the
  net-materiality rollup line, then the resolved-since-last-week line, then shorten the near-miss
  mention to its short form. Never touches a finding's own content or materiality. If still over 350
  after all three (rare, given the data is bounded), it ships as-is rather than chasing the last words
  with more machinery.

**What actually makes this a narrative and not a restatement of the dashboard's cards** — the original
version of this file reused each finding's `detail` string almost verbatim, which read as a
reformatted duplicate of the cards already shown above it. Two additions, both computed from data the
system already had and never rolled up before, in `AnalystAgent.analyze()`:
- **Net dollar impact** (`_compute_net_materiality`): the sum of every known materiality figure across
  the top concerns and the opportunity, stated once, early — instead of leaving a CUO to add up each
  finding's number themselves.
- **Week-over-week trend** (`_compute_trend`): is each top concern *new* this week, or has it been in
  the top 3 for N consecutive weeks running; did anything resolve since last week's report. This is the
  one thing a narrative can do that a snapshot-based card view genuinely cannot — checked against the
  real data before being treated as a real feature, not a theoretical one: the third concern slot has
  rotated through three different lines over the last three weeks while the top two held steady, which
  is the kind of fact a CUO who read last Monday's pack would actually want surfaced.

The findings themselves are also fed more context than before: `_format_findings` (used in the LLM
prompt) now includes each category's plain-language explanation and each severity's plain-language band
— the same richness the per-LoB narrative's prompt already had, with no good reason the weekly one
should have less to work with.

**The per-LoB narrative** (`NarrativeWriter.write_lob_narrative`): a focused 2-4 sentence note for the
dashboard's drill-down view, generated automatically when a line is selected (no button — this is meant
to read immediately). Exists because a raw finding string like *"Distribution Friction Risk: 27.6 days
vs peer avg 25.7 (severity 1.15)"* means nothing without context — this version explains what the
category means, frames severity in plain language (`formatting.severity_band`), states materiality or
explicitly says it isn't available, and gives a real confirmation sentence for clean lines instead of
silence. Cached per (line, as-of-week, sensitivity, has-key) so clicking between lines doesn't re-fire
the LLM repeatedly.

**Recommended Actions** are deliberately phrased as things to review or investigate ("review pricing
competitiveness with the underwriting team"), not confident root-cause diagnoses — the data shows what
happened, not why, and the prompts are explicit that the action should point at how to find out, not
assert the answer.

---

## The dashboard (`app.py`)

Four tabs:

- **Overview** — the portfolio-wide GWP-vs-plan trend (all 8 lines), the hit-rate heatmap, top 3
  concerns and top opportunity with severity *and* materiality shown side by side, and the weekly
  narrative with staleness detection (below).
- **LoB Drill-down** — per-line charts (GWP, hit rate, loss ratio, pipeline days), each with a **red**
  reference line for the relevant baseline or target — plan (100%), this line's own 8-week hit-rate
  baseline, the 60% loss-ratio target, and the cross-line peer average for pipeline days — computed
  identically to how `signals.py` computes them internally, so the visual matches the detection logic
  exactly. Below the charts, the automatic per-LoB narrative described above.
- **Ask a Question** — the scoped chat assistant, with quick-question buttons before the first message.
- **JSON / Prompts** — the raw structured output and the actual prompt files, for anyone who wants to
  see exactly what produced the narrative.

**Narrative staleness detection.** Moving the sensitivity slider or time-travel control already
recomputes findings instantly — `analyze()` runs fresh on every Streamlit rerun. But a previously
*generated* narrative just sits in session state and won't auto-update (deliberately — auto-regenerating
on every slider tick would mean firing a live LLM call on every interaction). Instead, `staleness.py`
fingerprints the current findings plus whether an API key is present, compares it to the fingerprint
captured at generation time, and shows an unmissable warning banner if they've diverged — the old
narrative stays visible (so nothing vanishes unexpectedly) but is clearly labeled stale, with the
regenerate button right there. Worth knowing: on the real dataset, the sensitivity slider alone doesn't
actually change the top-3 ranking at all (the dominant signals are unambiguous, not borderline) — only
time-travel reliably does. The synthetic test data in `tests/test_signals.py` is what actually exercises
the sensitivity behavior end to end.

**Why some text is escaped before `st.markdown()` and some isn't.** Streamlit's `st.markdown()`
interprets `$...$` as inline LaTeX by default. Any narrative or finding text containing two or more
dollar signs on one line risks being misread as a math span and rendering with visible artifacts —
this looked like backticks appearing around some dollar figures in early testing, and the cause was
checked directly: the raw narrative file had zero backticks and perfect `$` signs, confirming it was a
rendering bug, not a generation bug. `formatting.escape_dollar_signs()` is applied before every
`st.markdown()` or `.write()` call that renders narrative text, finding details, or chat
messages — including the user's own typed question, which was missed in an earlier pass and only
caught by writing a test that actually typed a dollar amount into the chat box.

---

## The chat assistant (`chat.py`)

Scoped Q&A only. Its system prompt explicitly forbids recommending an underwriting action — it can
explain *what* was flagged and why, never *what to do about it*. That boundary is commented in the
prompt file itself as deliberate and load-bearing, not a style preference, specifically to stop future
edits from quietly widening its scope.

---

## Test coverage (107 tests)

| File | What it covers |
|---|---|
| `test_data.py` | Ingestion, validation failures (missing column, week mismatch, missing file), the hit-rate formula |
| `test_signals.py` | All 5 checks against real data, severity/materiality values cross-checked against the project's independently-built analysis notebook, synthetic single-bad-week and sensitivity-slider tests, category attachment |
| `test_narrative.py` | Template completeness (no placeholder text), word-count enforcement on both the LLM and template paths, the double-period regression, in-range checks across all 12 weeks |
| `test_lob_narrative.py` | Plain-language category explanation, clean-line confirmation, severity framing, materiality citation |
| `test_agent.py` | Full pipeline regression test (named signals stay in the top 3), `analyze()`/`generate_narrative()` separation, time-travel |
| `test_app_smoke.py` | Dashboard loads without exception, both required 5.4 charts present, the heatmap date-axis bug (direct figure-object check, not just "no exception"), staleness warning end to end, chat dollar-sign escaping |
| `test_staleness.py` | Fingerprinting logic in isolation — including that an unchanged result doesn't produce a false warning |
| `test_formatting.py` | Sign formatting, dollar escaping, severity banding |

A recurring pattern worth noticing: several of the most useful tests exist not because they were planned
in advance, but because writing an end-to-end test (clicking a button, then moving a slider, then
checking what's on the page) surfaced a real bug that a narrower unit test wouldn't have — the
narrative-not-displaying-on-first-click bug, the heatmap year bug, and the chat-input escaping gap were
all found this way, not by inspection.

---

## Answers to the brief's three probing questions

**"What if it was just a one-off week?"** Every check requires a sustained pattern, not a single week.
Hit rate specifically: the dashboard's sensitivity slider controls how many of the last 4 weeks must
individually sit below baseline — not just the average. Proven by
`test_no_finding_for_hit_rate_when_only_2_of_4_recent_weeks_are_below_baseline` and
`test_sensitivity_slider_changes_hit_rate_outcome`.

**"How do you track prompt changes?"** Prompts are plain `.txt` files in `prompts/`, each with a version
number and changelog header, versioned in git like code. There's a real v1.0.0 → v1.0.1 edit in this
project's history (the word-count rule), with the reasoning documented in the commit and the file itself.

**"What could go wrong at 6am with nobody watching?"** `DataValidationError` → `run.py` exits 1 (safe to
alert on). The LLM call retries twice with backoff, then falls back to a complete offline template — the
pack always ships. `narrative_source` in the output always records which path actually ran.

---

## Known, stated gaps

- `DataLoader` checks structure (columns present, weeks consistent) — not data quality. Duplicate rows or
  a nonsensical value (a negative GWP) would not be caught.
- Materiality is `None` for hit-rate, loss-ratio, and pipeline findings — this dataset doesn't provide
  premium-per-policy or expense data to translate them into a dollar figure without inventing an
  assumption. Flagged honestly rather than estimated.
- No alerting integration (Slack/email) on the `DataValidationError` exit path — the natural next
  addition, genuinely outside this project's scope.
- Combined Ratio (mentioned in the brief's primer) is not computed — the data has no expense figures.
- The chat assistant's conversation history grows unbounded within a session — fine for a single weekly
  review, would need trimming for a long-running session.
- The per-LoB narrative and the main weekly narrative are generated independently and could, in
  principle, frame the same finding slightly differently in wording (not in numbers — both pull from the
  same `summary` dict). Not observed in practice, but not structurally prevented either.
- ~~"N weeks running" tracks rank persistence, not the underlying metric's trajectory~~ — addressed.
  See "Trajectory: a third, separate axis" above. `SignalDetector` now computes trajectory directly
  (peer z-score of the slope) for every check except `claims_anomaly`, kept structurally separate from
  severity and the ranking itself, the same way materiality already is.