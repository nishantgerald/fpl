"""Deterministic FPL recommendation engine.

Everything in this package except :mod:`engine.fpl_client` and
:mod:`engine.narrative` is a pure function of plain dicts — no network, no clock,
no randomness — so the whole decision path is testable offline and returns the
same answer for the same inputs.

See ``PRDs/prd-transfer-engine.md`` in the Flutter repo for the specification.
"""
