"""Prove one exact arm-label extension without changing calibration computation.

This is deliberately not a general implementation-compatibility mechanism.
Only the original three arm literals may become the specified five literals;
every other runner byte and every other recorded implementation hash must match.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

FORMAT = "startrain.recovery-arm-label-only-bridge"
RUNNER_KEY = "scripts/run_frozen_replay_optimizer_calibration.py"
ORIGINAL_ARMS = (
    "recovery-effective-control",
    "recovery-effective-moderate",
    "recovery-effective-high",
)
RATIO_ARMS = ("recovery-effective-muon-half", "recovery-effective-adam-double")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"compatibility input must be a regular file: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _hash(path.read_bytes()),
    }


def _read(pin: dict) -> bytes:
    path = Path(pin["path"])
    if _pin(path) != pin:
        raise ValueError("label-only compatibility artifact pin changed")
    return path.read_bytes()


def _runner_parts(data: bytes) -> tuple[bytes, tuple[str, ...]]:
    text = data.decode("utf-8")
    nodes = [
        node
        for node in ast.parse(text).body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "RECOVERY_ARMS"
            for target in node.targets
        )
    ]
    if (
        len(nodes) != 1
        or len(nodes[0].targets) != 1
        or not isinstance(nodes[0].value, ast.Tuple)
    ):
        raise ValueError("runner requires one literal RECOVERY_ARMS tuple")
    node = nodes[0]
    assert isinstance(node.value, ast.Tuple)
    if any(
        not isinstance(item, ast.Constant) or not isinstance(item.value, str)
        for item in node.value.elts
    ):
        raise ValueError("arm identifiers must be literal strings")
    arms = tuple(
        item.value
        for item in node.value.elts
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    )
    lines = data.splitlines(keepends=True)
    # AST columns are UTF-8 byte offsets. Keep all bytes outside the assignment,
    # including a statement after its closing parenthesis on the same line.
    assert node.end_lineno is not None and node.end_col_offset is not None
    start = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
    remainder = data[:start] + data[end:]
    return remainder, arms


def verify_label_only_change(old: bytes, new: bytes) -> bytes:
    old_rest, old_arms = _runner_parts(old)
    new_rest, new_arms = _runner_parts(new)
    if old_arms != ORIGINAL_ARMS or new_arms != (*ORIGINAL_ARMS, *RATIO_ARMS):
        raise ValueError(
            "only the exact two approved ratio arm identifiers may be added"
        )
    if old_rest != new_rest:
        raise ValueError("runner changed outside the RECOVERY_ARMS literal assignment")
    return "".join(
        difflib.unified_diff(
            old.decode().splitlines(keepends=True),
            new.decode().splitlines(keepends=True),
            fromfile="old-runner.py",
            tofile="new-runner.py",
        )
    ).encode()


def _publish(path: Path, contents: bytes) -> dict:
    if path.exists():
        if path.is_symlink() or path.read_bytes() != contents:
            raise ValueError("existing label-only evidence differs")
    else:
        with path.open("xb") as stream:
            stream.write(contents)
        path.chmod(0o444)
    return _pin(path)


def create_bridge(
    *,
    old_runner: Path,
    new_runner: Path,
    control_result: Path,
    control_launch_receipt: Path,
    old_native: Path,
    new_native: Path,
    output_dir: Path,
) -> dict:
    old, new = old_runner.read_bytes(), new_runner.read_bytes()
    diff = verify_label_only_change(old, new)
    control_pin, receipt_pin = _pin(control_result), _pin(control_launch_receipt)
    control = json.loads(_read(control_pin))
    receipt = json.loads(_read(receipt_pin))
    old_impl = control["recovery"]["implementation"]
    old_hashes = old_impl["source_sha256"]
    if old_hashes.get(RUNNER_KEY) != _hash(old):
        raise ValueError("control result was not produced by the pinned old runner")
    old_root, new_root = (
        old_runner.resolve().parents[1],
        new_runner.resolve().parents[1],
    )
    new_hashes = {}
    for relative, expected in old_hashes.items():
        old_path, new_path = old_root / relative, new_root / relative
        if _pin(old_path)["sha256"] != expected:
            raise ValueError(f"old implementation pin differs: {relative}")
        new_hashes[relative] = _pin(new_path)["sha256"]
        if relative != RUNNER_KEY and new_hashes[relative] != expected:
            raise ValueError(f"non-runner implementation changed: {relative}")
    old_native_pin, new_native_pin = _pin(old_native), _pin(new_native)
    if (
        old_native_pin["sha256"] != new_native_pin["sha256"]
        or receipt["native_sha256"] != old_native_pin["sha256"]
    ):
        raise ValueError("native binary differs from the control launch evidence")
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "format": FORMAT,
        "schema_version": 1,
        "scope": "exact-two-arm-label-extension-only",
        "control_result": control_pin,
        "control_launch_receipt": receipt_pin,
        "old_runner": _publish(output_dir / "old-runner.py", old),
        "new_runner": _publish(output_dir / "new-runner.py", new),
        "runner_diff": _publish(output_dir / "runner-labels.diff", diff),
        "old_native": old_native_pin,
        "new_native": new_native_pin,
        "old_implementation": old_impl,
        "new_implementation": {**old_impl, "source_sha256": new_hashes},
        "old_training_root": str(old_root),
        "new_training_root": str(new_root),
    }
    bridge_path = output_dir / "bridge.json"
    _publish(
        bridge_path, (json.dumps(document, sort_keys=True, indent=2) + "\n").encode()
    )
    return verify_bridge(bridge_path)


def verify_bridge(path: Path) -> dict:
    pin = _pin(path)
    document = json.loads(_read(pin))
    if (
        document.get("format") != FORMAT
        or document.get("schema_version") != 1
        or document.get("scope") != "exact-two-arm-label-extension-only"
    ):
        raise ValueError("unsupported label-only implementation bridge")
    old, new = _read(document["old_runner"]), _read(document["new_runner"])
    if verify_label_only_change(old, new) != _read(document["runner_diff"]):
        raise ValueError("archived label-only runner diff does not match")
    old_native, new_native = (
        _read(document["old_native"]),
        _read(document["new_native"]),
    )
    if old_native != new_native:
        raise ValueError("native binary differs between label-only implementations")
    control = json.loads(_read(document["control_result"]))
    receipt = json.loads(_read(document["control_launch_receipt"]))
    if receipt["native_sha256"] != _hash(old_native):
        raise ValueError("control native launch pin differs")
    if control["recovery"]["implementation"] != document["old_implementation"]:
        raise ValueError("control implementation differs from bridge")
    before, after = document["old_implementation"], document["new_implementation"]
    if {k: v for k, v in before.items() if k != "source_sha256"} != {
        k: v for k, v in after.items() if k != "source_sha256"
    }:
        raise ValueError("runtime version changed outside arm labels")
    if set(before["source_sha256"]) != set(after["source_sha256"]):
        raise ValueError("implementation file set changed outside arm labels")
    for relative, expected in before["source_sha256"].items():
        actual_new = after["source_sha256"][relative]
        if relative == RUNNER_KEY:
            if expected != _hash(old) or actual_new != _hash(new):
                raise ValueError("runner hashes differ from archived source")
        elif expected != actual_new:
            raise ValueError("non-runner implementation changed outside arm labels")
        for root_key, digest in (
            ("old_training_root", expected),
            ("new_training_root", actual_new),
        ):
            if _pin(Path(document[root_key]) / relative)["sha256"] != digest:
                raise ValueError("live implementation bytes differ from bridge pins")
    return {"pin": pin, "document": document}


def normalize_implementation(implementation: dict, bridge: dict) -> dict:
    document = bridge["document"]
    if implementation not in (
        document["old_implementation"],
        document["new_implementation"],
    ):
        raise ValueError(
            "implementation is not one of the two proven label-only versions"
        )
    return {
        **implementation,
        "source_sha256": {
            **implementation["source_sha256"],
            RUNNER_KEY: "label-only-equivalent:" + bridge["pin"]["sha256"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "old-runner",
        "new-runner",
        "control-result",
        "control-launch-receipt",
        "old-native",
        "new-native",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create_bridge(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
