from __future__ import annotations

import copy
import hashlib
import json

import pytest

from scripts import recovery_label_compatibility as labels
from scripts.run_frozen_replay_optimizer_calibration import RECOVERY_ARMS


def _runner(arms: tuple[str, ...]) -> bytes:
    return (
        "# Frozen computation\nRECOVERY_ARMS = (\n"
        + "".join(f"    {arm!r},\n" for arm in arms)
        + ")\n\ndef numerical_update(value):\n    return value * 0.5\n"
    ).encode()


def _inputs(tmp_path):
    old_root, new_root = tmp_path / "old/training", tmp_path / "new/training"
    old_runner, new_runner = old_root / labels.RUNNER_KEY, new_root / labels.RUNNER_KEY
    for path in (old_runner, new_runner):
        path.parent.mkdir(parents=True)
    old_runner.write_bytes(_runner(labels.ORIGINAL_ARMS))
    new_runner.write_bytes(_runner((*labels.ORIGINAL_ARMS, *labels.RATIO_ARMS)))
    for root in (old_root, new_root):
        (root / "startrain").mkdir()
        (root / "startrain/model.py").write_text("model = 1\n")
    old_native, new_native = tmp_path / "old/native.so", tmp_path / "new/native.so"
    old_native.write_bytes(b"identical-native-binary")
    new_native.write_bytes(old_native.read_bytes())
    implementation = {
        "torch_version": "test",
        "python_version": "test",
        "source_sha256": {
            relative: hashlib.sha256((old_root / relative).read_bytes()).hexdigest()
            for relative in (labels.RUNNER_KEY, "startrain/model.py")
        },
    }
    control = tmp_path / "control.json"
    control.write_text(json.dumps({"recovery": {"implementation": implementation}}))
    receipt = tmp_path / "control-launch.json"
    receipt.write_text(
        json.dumps(
            {"native_sha256": hashlib.sha256(old_native.read_bytes()).hexdigest()}
        )
    )
    return dict(
        old_runner=old_runner,
        new_runner=new_runner,
        control_result=control,
        control_launch_receipt=receipt,
        old_native=old_native,
        new_native=new_native,
        output_dir=tmp_path / "bridge",
    )


def test_ratio_labels_are_exact_explicit_parser_choices():
    assert RECOVERY_ARMS == (*labels.ORIGINAL_ARMS, *labels.RATIO_ARMS)
    assert all(len(arm) < 64 for arm in RECOVERY_ARMS)


def test_bridge_archives_proof_without_mutating_control(tmp_path):
    kwargs = _inputs(tmp_path)
    before = kwargs["control_result"].read_bytes()
    bridge = labels.create_bridge(**kwargs)
    assert kwargs["control_result"].read_bytes() == before
    assert labels.verify_bridge(kwargs["output_dir"] / "bridge.json") == bridge
    assert (kwargs["output_dir"] / "runner-labels.diff").read_text().count(
        "+    'recovery-effective-"
    ) == 2
    assert labels.normalize_implementation(
        bridge["document"]["old_implementation"], bridge
    ) == labels.normalize_implementation(
        bridge["document"]["new_implementation"], bridge
    )
    different = copy.deepcopy(bridge["document"]["new_implementation"])
    different["source_sha256"]["startrain/model.py"] = "0" * 64
    with pytest.raises(ValueError, match="two proven"):
        labels.normalize_implementation(different, bridge)
    kwargs["new_runner"].write_bytes(kwargs["new_runner"].read_bytes() + b"# changed\n")
    with pytest.raises(ValueError, match="live implementation"):
        labels.verify_bridge(kwargs["output_dir"] / "bridge.json")


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data.replace(b"value * 0.5", b"value * 2.0"),
        lambda data: data.replace(b")\n\ndef", b"); numerical_update = None\n\ndef"),
        lambda data: data + b"# Even comment-only unrelated edits are refused\n",
    ],
)
def test_bridge_rejects_every_other_runner_byte_change(change):
    old = _runner(labels.ORIGINAL_ARMS)
    new = _runner((*labels.ORIGINAL_ARMS, *labels.RATIO_ARMS))
    with pytest.raises(ValueError, match="outside"):
        labels.verify_label_only_change(old, change(new))


def test_bridge_rejects_arbitrary_extra_labels():
    with pytest.raises(ValueError, match="exact two"):
        labels.verify_label_only_change(
            _runner(labels.ORIGINAL_ARMS),
            _runner((*labels.ORIGINAL_ARMS, "recovery-effective-anything")),
        )


@pytest.mark.parametrize("kind", ["model", "native"])
def test_bridge_rejects_unrelated_implementation_or_binary_changes(tmp_path, kind):
    kwargs = _inputs(tmp_path)
    if kind == "model":
        (kwargs["new_runner"].parents[1] / "startrain/model.py").write_text(
            "model = 2\n"
        )
    else:
        kwargs["new_native"].write_bytes(b"different-native-binary")
    with pytest.raises(ValueError, match="changed|differs"):
        labels.create_bridge(**kwargs)


def test_bridge_rejects_modified_original_control(tmp_path):
    kwargs = _inputs(tmp_path)
    labels.create_bridge(**kwargs)
    kwargs["control_result"].write_text("{}")
    with pytest.raises(ValueError, match="pin changed"):
        labels.verify_bridge(kwargs["output_dir"] / "bridge.json")
