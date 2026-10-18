"""Atomic Semantic Cards (plan §7-8). Owner: Opus 2.

`hmb.cards.closure` is the shared canonicalization contract (rule set R, exclusion table E, MGS).
Contract entry points (docs/CONTRACTS.md) are re-exported here.
"""
from __future__ import annotations

from .build import build_card, check_builder_roster, check_primary_disjoint
from .consensus import consensus, consensus_rule_hash
from .gates import apply_book_cap, item_gates
from .oracle import oracle_card

__all__ = ["build_card", "check_builder_roster", "check_primary_disjoint", "consensus", "consensus_rule_hash",
           "item_gates", "apply_book_cap", "oracle_card"]
