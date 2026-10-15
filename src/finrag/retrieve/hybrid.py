"""Reciprocal Rank Fusion (Cormack et al., 2009).

score(d) = sum over result lists of 1 / (k + rank of d in that list)

Uses ranks only, so vector cosine scores and keyword ts_rank scores never
need to be put on a common scale. k=60 is the standard constant: it damps the
advantage of being #1 vs #2 so that agreement between lists matters most.
"""

from __future__ import annotations

from dataclasses import replace

from finrag.retrieve.vector import Hit

RRF_K = 60


def rrf(result_lists: list[list[Hit]], k: int = RRF_K, limit: int | None = None) -> list[Hit]:
    scores: dict[str, float] = {}
    first_seen: dict[str, Hit] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, 1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(hit.chunk_id, hit)
    ranked = sorted(scores, key=lambda cid: scores[cid], reverse=True)[:limit]
    return [replace(first_seen[cid], score=scores[cid]) for cid in ranked]
