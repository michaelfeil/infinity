from infinity_emb.primitives import RerankLimits, ReRankSingle


def test_rerank_single_str_repr_default_is_query_plus_document():
    # the no-limit path must stay byte-for-byte identical to query + document so existing
    # cache entries and length estimates are unaffected.
    single = ReRankSingle(query="where is paris", document="paris is in france")
    assert single.str_repr() == "where is parisparis is in france"


def test_rerank_single_str_repr_differs_per_limits():
    # the cache keys on str_repr, so two requests for the same pair but different limits
    # must produce different keys (otherwise a capped result is served for an uncapped one).
    pair = dict(query="q", document="d")
    default = ReRankSingle(**pair)
    capped = ReRankSingle(**pair, limits=RerankLimits(max_pair_tokens=32))
    capped_more = ReRankSingle(**pair, limits=RerankLimits(max_pair_tokens=64))

    assert capped.str_repr() != default.str_repr()
    assert capped.str_repr() != capped_more.str_repr()
