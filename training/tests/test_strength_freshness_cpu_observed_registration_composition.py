"""Credential-reader/schema composition with explicit synthetic admission.

The private proc files and registration premises are fixtures, not a qualified
production identity or a successful preservation capture.
"""

from copy import deepcopy

import pytest

from scripts import strength_freshness_cpu_observed_registration as registration
from tests.test_strength_freshness_cpu_credentials import ADMISSION, setup
from tests.test_strength_freshness_cpu_observed_registration import registration_fixture


@pytest.mark.parametrize(
    ("values", "accepted"),
    [((1000, 1000, 1000, 1000), True), ((1000, 1001, 1002, 1003), False)],
)
def test_actual_private_status_vector_is_not_coerced_into_registration(
    tmp_path, monkeypatch, values, accepted
):
    document, scope, writer, unused_io, _ = registration_fixture()
    status = (
        b"Name:\tPRIVATE_CREDENTIAL_NAME\nUid:\t"
        + b"\t".join(str(value).encode() for value in values)
        + b"\n"
    )
    _, _, io, _ = setup(tmp_path, monkeypatch, status=status)
    observed = io.process_credentials(ADMISSION)
    vector = deepcopy(observed.value["uids"])
    assert tuple(vector.values()) == values
    document["kernel_context"]["credentials"]["learner"] = vector
    raw = registration.encoded(document)

    def parse():
        return registration.parse_observed_registration(
            raw,
            approved_sha256=registration.sha(raw),
            scope=scope,
            writer_evidence=writer,
        )

    if accepted:
        parsed = parse()
        summary = parsed.safe_summary()
        assert summary["schema_validated"] is True
        for field in (
            "writer_qualified",
            "runtime_qualified",
            "preservation_passed",
            "execution_authorized",
        ):
            assert summary[field] is False
        assert "PRIVATE_CREDENTIAL_NAME" not in repr(parsed) + str(summary)
    else:
        with pytest.raises(registration.ObservedRegistrationRefusal, match="uid"):
            parse()
    assert observed.value["uids"] == vector
    assert "PRIVATE_CREDENTIAL_NAME" not in repr(observed) + str(observed.audit)
    assert unused_io.calls == []
