"""
All thresholds and constants used by the detection engine live here, in one place, each with its
justification documented inline. Every threshold traces to one of two sources, decided independently
of this dataset:

    (1) The brief's own literal wording (quoted where applicable)
    (2) A standard, generic statistical or business convention that would be reached for on any
        dataset, not this one specifically

This discipline matters: an earlier draft of this project's analysis notebook combined several metrics
into one weighted score, and the weights were tuned by trial and error until the score reproduced a
known answer. That is circular -- a detection method built that way has no demonstrated ability to work
on data where the answer isn't already known. Every constant below was chosen, and is documented, before
checking whether it reproduces anything.
"""

LINES_OF_BUSINESS = [
    "Cyber",
    "Transactional Liability",
    "Environmental",
    "Political Risk",
    "Political Violence",
    "Financial Institutions",
    "Professional Lines",
    "Excess Casualty",
]

REQUIRED_COLUMNS = {
    "submissions": ["week_ending", "lob", "submissions_count", "quoted_count", "bound_count", "declined_count", "ntu_count"],
    "premium": ["week_ending", "lob", "actual_gwp", "plan_gwp", "ytd_actual", "ytd_plan"],
    "pipeline": ["week_ending", "lob", "open_quotes_count", "open_quotes_gwp_est", "avg_days_in_pipeline"],
    "loss": ["week_ending", "lob", "new_claims_count", "new_claims_incurred_est", "attritional_loss_ratio_ytd"],
}

# ---------------------------------------------------------------------------
# CHECK A -- GWP vs. plan band
# Source: a 15-percentage-point band either side of 100% of plan is a common underwriting tolerance
# for "materially off plan" -- a generic convention, not tuned to this dataset.
# ---------------------------------------------------------------------------
GWP_BAND_LOW = 85.0       # % of plan
GWP_BAND_HIGH = 115.0     # % of plan
GWP_SUSTAINED_FRACTION = 0.8   # must hold for at least 80% of the weeks in view -- a sustained pattern, not a blip

# ---------------------------------------------------------------------------
# CHECK B -- Hit rate collapse
# Source: the brief's own wording describes "the final four weeks" as the window of interest, and a
# baseline drawn from the weeks before it. A 25% relative drop is a standard "meaningful change"
# threshold in monitoring contexts -- not specific to this dataset.
# ---------------------------------------------------------------------------
HIT_RATE_BASELINE_WEEKS = 8
HIT_RATE_RECENT_WEEKS = 4
HIT_RATE_RELATIVE_DROP = 0.25
HIT_RATE_SUSTAINED_MIN_WEEKS = 3   # at least 3 of the 4 recent weeks must individually sit below baseline

# ---------------------------------------------------------------------------
# CHECK C -- Loss ratio trend
# Source: "~60% target" is stated directly in the brief's primer. The slope threshold (1.0pp/week) is
# set well below the brief's own worked example (~3pp/week) precisely so it is not reverse-fitted to
# that example -- it is a lower, generic bar that the example clears comfortably, not a number chosen
# to match it.
# ---------------------------------------------------------------------------
LOSS_RATIO_TARGET = 60.0           # %
LOSS_RATIO_TREND_WINDOW = 6        # weeks, fit with a linear regression, not an endpoint comparison
LOSS_RATIO_SLOPE_ALERT = 1.0       # percentage points per week
LOSS_RATIO_PROXIMITY_FLOOR = 50.0  # only counts if already within real striking distance of the target

# ---------------------------------------------------------------------------
# CHECK D -- Claims anomaly (the one check with no brief-given definition at all)
# Source: z > 2.0 is the standard textbook convention for "statistically unusual" (roughly the 95th
# percentile under normality) -- used here because it is the standard choice for this kind of check
# everywhere, not because of anything specific to this dataset.
# ---------------------------------------------------------------------------
CLAIMS_Z_THRESHOLD = 2.0
CLAIMS_MIN_HISTORY_WEEKS = 6   # need a reasonably stable baseline before a z-score is meaningful

# ---------------------------------------------------------------------------
# CHECK E -- Pipeline friction, measured against peers rather than an absolute day count
# Source: "more than one cross-sectional standard deviation above the peer average" is a standard
# outlier convention for comparing entities against a peer group, not a number chosen for this data.
# ---------------------------------------------------------------------------
PIPELINE_PEER_STD_THRESHOLD = 1.0

# ---------------------------------------------------------------------------
# Output shaping
# ---------------------------------------------------------------------------
TOP_N_CONCERNS = 3
TOP_N_OPPORTUNITIES = 1

# A concern just outside the top N can still be worth naming explicitly if it's a genuine near-miss,
# not a clear gap. 0.5 (half a standard deviation) is a standard statistical convention for "close
# enough to not be a meaningfully different case" -- chosen on its own terms, not tuned to include or
# exclude any specific line of business. This rule is what decides whether a 4th item gets mentioned,
# never a hard-coded name.
NEAR_MISS_SEVERITY_MARGIN = 0.5

# Human-readable risk categories, used in the narrative and dashboard so findings read as different
# KINDS of risk rather than competitors on one artificial scale. See README for the reasoning.
CHECK_CATEGORY = {
    ("gwp_band", "concern"): "Premium Risk",
    ("gwp_band", "opportunity"): "Growth Opportunity",
    ("hit_rate_collapse", "concern"): "Conversion Risk",
    ("loss_ratio_trend", "concern"): "Loss Cost Trend Risk",
    ("claims_anomaly", "concern"): "Claims Shock Risk",
    ("pipeline_friction", "concern"): "Distribution Friction Risk",
}

# Plain-language, one-sentence explanation of what each category actually means in business terms.
# Added because a category label alone ("Distribution Friction Risk") means nothing to a CUO without
# insurance-ops-jargon context -- these are meant to be shown alongside the label every time it appears,
# not assumed as background knowledge.
CATEGORY_EXPLANATION = {
    "Premium Risk": "the line is writing less premium than budgeted, consistently, not just one bad week",
    "Growth Opportunity": "the line is writing more premium than budgeted, consistently",
    "Conversion Risk": "fewer quotes are converting into bound policies than is normal for this line",
    "Loss Cost Trend Risk": "claims activity relative to premium is trending the wrong way, not just sitting high",
    "Claims Shock Risk": "a single week's claims came in well outside this line's own normal range",
    "Distribution Friction Risk": "quotes for this line are taking longer than peer lines to move through underwriting, "
                                   "which can point to broker hesitation or pricing friction",
}

# Prompt / narrative config
LLM_MODEL = "claude-sonnet-4-6"
LLM_MAX_RETRIES = 2
LLM_MAX_TOKENS = 800  # bumped from 600 -- a 400-word hard cap plus headers/structure needs more headroom
# Narrative length policy -- deliberately simple, on direct instruction after an earlier version's
# trim logic (drop materiality clause-by-clause, near-miss full/short/none, padding for short weeks)
# was judged over-engineered relative to its actual value. Findings come first: length is secondary,
# and trimming should never remove a finding or a number, only tighten prose. 250-300 is a soft target,
# not enforced; trimming only activates above 350; 400 is a hard ceiling aimed for with one corrective
# pass, not a guarantee chased with retries.
NARRATIVE_WORD_TARGET_MIN = 250
NARRATIVE_WORD_TARGET_MAX = 300
NARRATIVE_WORD_TRIM_TRIGGER = 350
NARRATIVE_WORD_HARD_CAP = 400
PROMPT_DIR = "prompts"