"""Cross-namespace merge behaviour — where a plausible-looking merge is silently wrong."""

import pytest

from finvec.search import (
    Hit,
    build_filter,
    default_merge,
    merge_by_score,
    merge_rrf,
    score_by_for,
    year_namespaces,
)


def hits(namespace, scores):
    return [
        Hit(id=f"{namespace}-{i}", namespace=namespace, score=s, rank=i + 1)
        for i, s in enumerate(scores)
    ]


def test_year_namespaces_parses_ranges_and_lists():
    assert year_namespaces("2021-2024") == ["2021", "2022", "2023", "2024"]
    assert year_namespaces("2019,2021-2022") == ["2019", "2021", "2022"]
    assert year_namespaces("2024") == ["2024"]
    assert year_namespaces(None, default=["2024"]) == ["2024"]


def test_score_merge_orders_by_absolute_score():
    merged = merge_by_score([hits("2023", [0.9, 0.4]), hits("2024", [0.8, 0.7])], 3)
    assert [h.score for h in merged] == [0.9, 0.8, 0.7]


def test_rrf_ignores_score_scale_across_namespaces():
    """The BM25 trap: one namespace's scores are inflated by rarer terms.

    Raw score merge would hand back all of 2008; RRF must interleave by rank instead.
    """
    inflated = hits("2008", [42.0, 39.0, 37.0])   # rare terms -> high IDF
    normal = hits("2021", [3.1, 2.9, 2.7])        # common terms -> low IDF

    by_score = merge_by_score([inflated, normal], 3)
    assert {h.namespace for h in by_score} == {"2008"}, "score merge is namespace-biased"

    fused = merge_rrf([inflated, normal], 4)
    assert {h.namespace for h in fused} == {"2008", "2021"}
    # Rank 1 from both years outranks rank 2 from either.
    assert [h.rank for h in fused] == [1, 1, 2, 2]


def test_rrf_ties_break_by_original_score_deterministically():
    a = hits("2023", [1.0])
    b = hits("2024", [5.0])
    first = merge_rrf([a, b], 2)
    second = merge_rrf([b, a], 2)
    assert [h.id for h in first] == [h.id for h in second]
    # Same rank tier, so the stronger raw score leads.
    assert first[0].namespace == "2024"


def test_rrf_is_round_robin_because_namespaces_are_disjoint():
    # Worth asserting explicitly: with disjoint lists RRF cannot do real fusion, so it
    # imposes equal representation per namespace. That is a tradeoff, not a free win.
    merged = merge_rrf([hits("a", [9, 8, 7]), hits("b", [1, 1, 1])], 6)
    assert [h.namespace for h in merged[:2]] == ["a", "b"]
    assert [h.rank for h in merged] == [1, 1, 2, 2, 3, 3]


def test_default_merge_picks_scale_free_only_where_needed():
    many = ["2023", "2024"]
    assert default_merge("text", many) == "rrf"      # BM25 is not comparable
    assert default_merge("dense", many) == "score"   # cosine is
    assert default_merge("hybrid", many) == "score"  # ranking is dense
    # A single namespace has nothing to merge, so scale never matters.
    assert default_merge("text", ["2024"]) == "score"


def test_filter_excludes_boilerplate_by_default_and_omits_year():
    filt = build_filter(ticker="nvda")
    assert filt["ticker"] == {"$eq": "NVDA"}
    assert filt["is_boilerplate"] == {"$eq": False}
    # Year is the namespace, not a filter — filtering on it too would burn read units
    # for nothing.
    assert "fiscal_year" not in filt


def test_must_contain_becomes_a_hard_filter_not_a_score_clause():
    filt = build_filter(must_contain="Azure")
    assert filt["text"] == {"$match_all": "Azure"}


def test_include_boilerplate_drops_the_clause_entirely():
    assert "is_boilerplate" not in (build_filter(ticker="F", include_boilerplate=True))


def test_score_by_never_mixes_scoring_types():
    # The server rejects mixed types, so each mode must emit exactly one clause.
    assert len(score_by_for("text", "q", None)) == 1
    assert score_by_for("text", "q", None)[0]["type"] == "text"
    assert score_by_for("dense", "q", [0.1])[0]["type"] == "dense_vector"
    assert score_by_for("hybrid", "q", [0.1])[0]["type"] == "dense_vector"
    with pytest.raises(ValueError, match="needs a query embedding"):
        score_by_for("dense", "q", None)
