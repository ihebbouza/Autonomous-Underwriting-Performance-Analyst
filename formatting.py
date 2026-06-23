"""
Small shared formatting helpers. Pulled out into their own module specifically because the
materiality sign-formatting bug was fixed once in narrative.py and shipped unfixed in app.py at the
same time -- a single shared function makes that class of bug harder to reintroduce.
"""


def fmt_usd(value):
    if value is None:
        return "not directly computable"
    sign = "-" if value < 0 else "+"
    return f"{sign}${abs(value):,.0f}"


def severity_band(severity):
    """
    Plain-language framing for a peer z-score severity, so 'severity 1.15' doesn't have to mean
    anything to a reader on its own. Bands follow the common statistical convention that z < 1.5 is
    unremarkable, 1.5-2.0 is a real but moderate signal, and > 2.0 is a clearly extreme outlier --
    not tuned to this dataset, the same convention applies regardless of what the numbers turn out to be.
    """
    if severity >= 2.0:
        return "a high-priority signal -- among the most statistically extreme in the portfolio this week"
    if severity >= 1.5:
        return "a moderate signal worth attention, though not among the most extreme this week"
    return "a minor signal -- real, but low-priority relative to the rest of the portfolio this week"


def ensure_period(text):
    """Avoid 'detail string..' double periods when a detail already ends with its own sentence stop."""
    return text if text.rstrip().endswith((".", "!", "?")) else text + "."


def escape_dollar_signs(text):
    """
    Streamlit's st.markdown() interprets $...$ as inline LaTeX by default. Any text containing two or
    more literal dollar signs on one line risks being misread as a math span, which can render as
    garbled or monospace-looking output. Escaping every '$' as '\\$' makes Streamlit treat it as a
    literal character. Apply this to any narrative or finding text before it reaches st.markdown --
    never to text destined for plain print() or a written file, which don't have this problem.
    """
    return text.replace("$", r"\$")