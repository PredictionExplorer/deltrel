from collections.abc import Mapping
from copy import deepcopy
import hashlib
from types import SimpleNamespace

import pytest

from scripts import migrate_continuous_profile as migration


def original_hash_oracle(config):
    """The previous eager algorithm, retained as an independent output oracle."""
    materialized = config.as_dict()
    variants = list(migration.compatible_config_epoch_payloads(materialized))
    for path, default in migration._ADDITIVE_DEFAULT_FIELDS:
        current = materialized
        for key in path:
            current = current.get(key) if isinstance(current, Mapping) else None
        if type(current) is not type(default) or current != default:
            continue
        variants.extend(
            stripped
            for variant in list(variants)
            if (stripped := migration._without_field(variant, path)) is not None
        )
    return {
        hashlib.sha256(migration._canonical_config_bytes(variant)).hexdigest()
        for variant in variants
    }


def payload():
    result = {"unrelated": {"preserved": [False, 0, 1, 1.0]}}
    for path, default in migration._ADDITIVE_DEFAULT_FIELDS:
        parent = result
        for name in path[:-1]:
            parent = parent.setdefault(name, {})
        parent[path[-1]] = default
    return result


@pytest.mark.parametrize("kind", ["defaults", "enabled", "type_lookalikes", "missing"])
def test_hash_dedup_preserves_exact_oracle_and_inputs(monkeypatch, kind):
    materialized = payload()
    for path, default in migration._ADDITIVE_DEFAULT_FIELDS:
        parent = materialized
        for name in path[:-1]:
            parent = parent[name]
        if kind == "enabled":
            parent[path[-1]] = True if type(default) is bool else 0.5
        elif kind == "type_lookalikes":
            parent[path[-1]] = int(default)
        elif kind == "missing":
            del parent[path[-1]]
    original = deepcopy(materialized)
    epochs = [deepcopy(materialized) for _ in range(4)]
    # Canonical bytes, not Python dict equality, distinguish numeric lookalikes.
    epochs.append(deepcopy(materialized) | {"distinct": False})
    epochs.append(deepcopy(materialized) | {"distinct": 0})
    prior = migration._without_field(
        epochs[0], migration._ADDITIVE_DEFAULT_FIELDS[0][0]
    )
    if prior is not None:
        epochs.extend([prior, deepcopy(prior)])
    frozen_epochs = deepcopy(epochs)
    monkeypatch.setattr(
        migration, "compatible_config_epoch_payloads", lambda _: tuple(epochs)
    )
    config = SimpleNamespace(as_dict=lambda: materialized)
    assert migration._compatible_source_config_sha256s(config) == original_hash_oracle(
        config
    )
    assert materialized == original and epochs == frozen_epochs


def test_duplicate_epochs_are_collapsed_before_expansion_and_each_hash_runs_once(
    monkeypatch,
):
    materialized = payload()
    monkeypatch.setattr(
        migration,
        "compatible_config_epoch_payloads",
        lambda _: tuple(deepcopy(materialized) for _ in range(16)),
    )
    config = SimpleNamespace(as_dict=lambda: materialized)
    expected = original_hash_oracle(config)
    assert len(expected) == 32
    original_strip = migration._without_field
    original_hash = migration.hashlib.sha256
    strip_calls = 0
    hashed_bytes = []

    def strip(*args):
        nonlocal strip_calls
        strip_calls += 1
        return original_strip(*args)

    def sha(encoded):
        hashed_bytes.append(encoded)
        return original_hash(encoded)

    monkeypatch.setattr(migration, "_without_field", strip)
    monkeypatch.setattr(migration.hashlib, "sha256", sha)
    assert migration._compatible_source_config_sha256s(config) == expected
    assert len(hashed_bytes) == len(set(hashed_bytes)) == len(expected)
    assert strip_calls == 31  # 1 + 2 + 4 + 8 + 16 unique parent representations.
