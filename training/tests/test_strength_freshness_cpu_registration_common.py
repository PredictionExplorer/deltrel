"""Pure registration extraction; historical admission remains mandatory."""

import copy
from typing import cast

import pytest
from scripts import strength_freshness_cpu_collect_identity as identity
from tests.test_strength_freshness_cpu_collect_identity import identity_fixture


def common(reg):
    return {key: copy.deepcopy(reg[key]) for key in identity.COMMON_FIELDS}


def test_common_checks_and_existing_constructor_use_same_contract():
    reg, io, _ = identity_fixture()
    before = copy.deepcopy(reg)
    assert identity.validate_common_registration(common(reg), io.scope) is None
    collector = identity.IdentityCollector(reg, cast(identity.readonly.ReadOnlyIO, io))
    assert collector.reg == before == reg
    assert not io.calls


@pytest.mark.parametrize(
    "field", ["format", "schema_version", "birth_reference", "skip_birth", "contract"]
)
def test_common_projection_never_accepts_admission_fields(field):
    reg, io, _ = identity_fixture()
    value = common(reg)
    value[field] = True
    with pytest.raises(identity.CollectionRefusal, match="common-registration-fields"):
        identity.validate_common_registration(value, io.scope)
    assert not io.calls


@pytest.mark.parametrize(
    "kind", ["null", "missing", "empty", "bounds", "format", "mixed", "common_only"]
)
def test_old_header_and_birth_refuse_before_any_io(kind):
    reg, io, _ = identity_fixture()
    if kind == "null":
        reg["birth_reference"]["qualification_sha256"] = None
    elif kind == "missing":
        del reg["birth_reference"]
    elif kind == "empty":
        reg["birth_reference"] = {}
    elif kind == "bounds":
        reg["birth_reference"]["bounds"]["learner"] += 1
    elif kind == "format":
        reg["format"] = "strength-preservation-observed-window-registration-v1"
    elif kind == "mixed":
        reg["contract"] = "learner-append-observed-window-v1"
    else:
        reg = common(reg)
    with pytest.raises(identity.CollectionRefusal):
        identity.IdentityCollector(reg, cast(identity.readonly.ReadOnlyIO, io))
    assert io.calls == []


def test_even_replaced_common_validator_cannot_bypass_old_birth(monkeypatch):
    reg, io, _ = identity_fixture()
    reg["birth_reference"]["qualification_sha256"] = None
    called = []
    monkeypatch.setattr(
        identity, "validate_common_registration", lambda *args: called.append(args)
    )
    with pytest.raises(identity.CollectionRefusal, match="birth-reference-binding"):
        identity.IdentityCollector(reg, cast(identity.readonly.ReadOnlyIO, io))
    assert called == io.calls == []


def mutate(reg, kind):
    if kind == "encoding":
        reg["encoding_contract_sha256"] = "f" * 64
    elif kind == "timer":
        reg["timer_environment_addendum_sha256"] = "f" * 64
    elif kind == "runtime":
        reg["policy"]["static"]["source_commit"] = "f" * 40
    elif kind == "roles":
        reg["heartbeats"].pop("learner")
    elif kind == "origin":
        reg["origins"]["learner"]["restart_counter_scope"] = "systemd-unit:NRestarts"
    elif kind == "source":
        reg["source_pins"]["source"]["bytes"] = 2**20 + 1
    elif kind == "cache":
        next(iter(reg["cached_references"].values()))["qualification_sha256"] = None
    elif kind == "env":
        reg["origins"]["learner"]["environment"] = {"BAD-NAME": {}}
    elif kind == "boot":
        reg["boot"]["edges"] = [
            {"from": "foreign.target", "to": "foreign.target", "relation": "Wants"}
        ]
    elif kind == "unit":
        reg["units"][identity.RUNTIME]["environment_files"] = [
            {"key": "missing", "ignore_missing": False}
        ]
    elif kind == "counter":
        reg["counter_scopes"]["workers"] = "owning-unit-NRestarts"
    else:
        reg["origins"]["learner"] = None


@pytest.mark.parametrize(
    "kind",
    [
        "encoding",
        "timer",
        "runtime",
        "roles",
        "origin",
        "source",
        "cache",
        "env",
        "boot",
        "unit",
        "counter",
        "malformed",
    ],
)
def test_common_and_legacy_reject_same_invalid_invariant(kind):
    reg, io, _ = identity_fixture()
    mutate(reg, kind)
    with pytest.raises(identity.CollectionRefusal) as direct:
        identity.validate_common_registration(common(reg), io.scope)
    with pytest.raises(identity.CollectionRefusal) as legacy:
        identity.IdentityCollector(reg, cast(identity.readonly.ReadOnlyIO, io))
    assert str(direct.value) == str(legacy.value)
    assert io.calls == []
