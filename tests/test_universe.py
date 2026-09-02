from collections import Counter

from alpaca_quant_agent import universe as u


def test_every_symbol_lands_in_exactly_one_named_bucket():
    # A symbol missing from CORRELATION_BUCKETS silently falls into a singleton
    # bucket, which defeats the concentration gate's purpose -- guard against it.
    counts = Counter(sym for syms in u.CORRELATION_BUCKETS.values() for sym in syms)
    missing = [s for s in u.SYMBOLS if s not in counts]
    dupes = [s for s, c in counts.items() if c > 1]
    assert missing == [], f"symbols with no named bucket: {missing}"
    assert dupes == [], f"symbols in more than one bucket: {dupes}"


def test_universe_symbols_unique():
    assert len(u.SYMBOLS) == len(set(u.SYMBOLS))


def test_bucket_for_matches_definitions():
    for bucket, syms in u.CORRELATION_BUCKETS.items():
        for sym in syms:
            assert u.bucket_for(sym) == bucket


def test_unknown_symbol_gets_singleton_bucket():
    assert u.bucket_for("ZZZZ").startswith("_singleton::")
