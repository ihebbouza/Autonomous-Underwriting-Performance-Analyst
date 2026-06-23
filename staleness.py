"""
Detects whether a previously-generated narrative is stale relative to the current findings.

Deliberately based on the FINDINGS themselves (lob/check/severity/category for every top concern and
the top opportunity) plus whether an API key is now present -- not on the raw slider values. Moving the
sensitivity slider from 3 to 4 when it happens not to change which lines get flagged should NOT trigger
a false "stale" warning; only an actual change in what the narrative would need to say should.
"""


def fingerprint(summary, has_api_key):
    findings_key = tuple(
        (f["lob"], f["check"], f["severity"])
        for f in summary["top_concerns"] + summary["top_opportunities"]
    )
    return (findings_key, bool(has_api_key))


def is_stale(cached_fingerprint, summary, has_api_key):
    return cached_fingerprint != fingerprint(summary, has_api_key)
