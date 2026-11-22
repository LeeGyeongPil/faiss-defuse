from __future__ import annotations

import random
from pathlib import Path

from helpers import variant_reduce_os_system

from faiss_defuse import Limits
from faiss_defuse._stub_decode import decode_checked
from faiss_defuse.errors import DefuseError

LIMITS = Limits(max_pickle_bytes=1_000_000, max_opcodes=100_000)


def _mutate(rng: random.Random, data: bytes) -> bytes:
    raw = bytearray(data)
    for _ in range(rng.randint(1, 4)):
        kind = rng.choice(["flip", "set", "insert", "delete", "truncate", "dup"])
        pos = rng.randrange(len(raw)) if raw else 0
        if kind == "flip" and raw:
            raw[pos] ^= 1 << rng.randrange(8)
        elif kind == "set" and raw:
            raw[pos] = rng.randrange(256)
        elif kind == "insert":
            raw.insert(pos, rng.randrange(256))
        elif kind == "delete" and raw:
            del raw[pos]
        elif kind == "truncate" and raw:
            del raw[pos:]
        elif kind == "dup" and raw:
            raw[pos:pos] = raw[pos : pos + rng.randint(1, 16)]
    return bytes(raw)


def test_mutations_never_execute(clean_store: Path, sentinel: Path) -> None:
    clean = (clean_store / "index.pkl").read_bytes()
    evil, _ = variant_reduce_os_system(sentinel)
    rng = random.Random(20261122)
    outcomes = {"decoded": 0, "refused": 0}
    for i in range(500):
        base = clean if i % 2 == 0 else evil
        mutant = _mutate(rng, base)
        try:
            decode_checked(mutant, limits=LIMITS)
        except DefuseError:
            outcomes["refused"] += 1
        else:
            outcomes["decoded"] += 1
        assert not sentinel.exists(), f"mutation {i} executed the payload"
    assert outcomes["refused"] > 100, outcomes  # the fuzzer really produced hostile or broken input
