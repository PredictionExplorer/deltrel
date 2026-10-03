from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from deltreltrain import publish
from deltreltrain.browser_export import export_browser_champion
from deltreltrain.browser_runtime import qualified_runtime
from deltreltrain.checkpoint import sha256_file
from deltreltrain.publish import publish_browser_artifacts
from test_browser_export import champion_fixture


@pytest.fixture(scope="module")
def release(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    root = tmp_path_factory.mktemp("qualified-browser")
    export = export_browser_champion(
        champion_fixture(root, auxiliary=False), root / "export"
    )
    runtime = qualified_runtime()
    wasm = Path(__file__).parents[2] / "public/models/deltrel" / runtime["directory"]
    return Path(export["manifest"]), wasm


def test_qualified_publication_preserves_old_channel_and_commits_pointer_last(
    release: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, wasm = release
    runtime = qualified_runtime()
    target = tmp_path / "public"
    target.mkdir()
    legacy = target / "manifest.json"
    legacy.write_bytes(b"unchanged old page channel")
    old_runtime = target / runtime["legacy"]["directory"]
    old_runtime.mkdir()
    (old_runtime / "deltrel_wasm.js").write_bytes(b"old runtime")
    replacements = []
    replace = os.replace

    def observe(source: str, destination: str) -> None:
        replacements.append(str(destination))
        replace(source, destination)

    monkeypatch.setattr("deltreltrain.publish.os.replace", observe)
    published = publish_browser_artifacts(
        manifest, target, wasm_source_directory=wasm, channel="qualified"
    )
    assert replacements == [str(target / runtime["manifest"])]
    assert legacy.read_bytes() == b"unchanged old page channel"
    assert (old_runtime / "deltrel_wasm.js").read_bytes() == b"old runtime"
    assert Path(str(published["manifest"])).read_bytes() == manifest.read_bytes()
    for kind in ("module", "binary"):
        assert (
            sha256_file(target / runtime["directory"] / runtime[kind]["path"])
            == runtime[kind]["sha256"]
        )
    # A verified repeat is safe; no differing immutable bytes are overwritten.
    publish_browser_artifacts(
        manifest, target, wasm_source_directory=wasm, channel="qualified"
    )


@pytest.mark.parametrize("kind", ["module", "binary"])
def test_qualified_runtime_tampering_leaves_channel_untouched(
    release: tuple[Path, Path],
    tmp_path: Path,
    kind: str,
) -> None:
    manifest, wasm = release
    copied = tmp_path / "source"
    shutil.copytree(wasm, copied)
    entry = qualified_runtime()[kind]
    path = copied / entry["path"]
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    target = tmp_path / "target"
    with pytest.raises(ValueError, match="SHA-256"):
        publish_browser_artifacts(
            manifest, target, wasm_source_directory=copied, channel="qualified"
        )
    assert not (target / qualified_runtime()["manifest"]).exists()


def test_qualified_existing_conflict_and_interrupted_commit_keep_previous_pointer(
    release: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, wasm = release
    runtime = qualified_runtime()
    target = tmp_path / "target"
    target.mkdir()
    pointer = target / runtime["manifest"]
    pointer.write_bytes(b"previous complete channel")
    artifact = target / runtime["directory"] / runtime["binary"]["path"]
    artifact.parent.mkdir()
    artifact.write_bytes(b"conflicting immutable bytes")
    with pytest.raises(ValueError):
        publish_browser_artifacts(
            manifest, target, wasm_source_directory=wasm, channel="qualified"
        )
    assert artifact.read_bytes() == b"conflicting immutable bytes"
    assert pointer.read_bytes() == b"previous complete channel"
    artifact.unlink()
    replace = os.replace

    def interrupt(source: str, destination: str) -> None:
        if Path(destination) == pointer:
            raise OSError("interrupted before channel commit")
        replace(source, destination)

    monkeypatch.setattr("deltreltrain.publish.os.replace", interrupt)
    with pytest.raises(OSError, match="interrupted"):
        publish_browser_artifacts(
            manifest, target, wasm_source_directory=wasm, channel="qualified"
        )
    assert pointer.read_bytes() == b"previous complete channel"
    assert sha256_file(artifact) == runtime["binary"]["sha256"]


def test_closed_channel_rejects_arbitrary_targets_and_runtime_rebuilds(
    release: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    manifest, wasm = release
    with pytest.raises(ValueError, match="channel"):
        publish_browser_artifacts(
            manifest, tmp_path / "unknown", channel="https://other.test"
        )
    with pytest.raises(ValueError, match="prebuilt"):
        publish_browser_artifacts(
            manifest,
            tmp_path / "build",
            channel="qualified",
            wasm_build_command=("not-executed",),
        )
    payload = json.loads(manifest.read_text())
    payload["features"]["hash"] = "wrong"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incompatible"):
        publish_browser_artifacts(
            invalid, tmp_path / "wrong", wasm_source_directory=wasm, channel="qualified"
        )


@pytest.mark.parametrize("channel", ["legacy", "qualified"])
def test_manifest_replacement_after_validation_preserves_previous_channel(
    release: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
) -> None:
    manifest, wasm = release
    source = tmp_path / "source"
    shutil.copytree(manifest.parent, source)
    copied_manifest = source / manifest.name
    original_bytes = copied_manifest.read_bytes()
    target = tmp_path / "target"
    published = publish_browser_artifacts(
        copied_manifest, target, wasm_source_directory=wasm, channel=channel
    )
    pointer = Path(str(published["manifest"]))
    previous = pointer.read_bytes()
    replacement = json.loads(original_bytes)
    replacement["model_version"] = "x" + replacement["model_version"][1:]
    onnx = replacement["artifacts"]["onnx"]
    onnx["file"] = "x" + onnx["file"][1:]
    onnx["sha256"] = "0" * 64
    replacement_bytes = (
        json.dumps(replacement, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert len(replacement_bytes) == len(original_bytes)
    assert replacement_bytes != original_bytes
    assert not (source / onnx["file"]).exists()
    original_stage_copy = publish._stage_copy
    replaced = False

    def replace_before_staging(source_path: Path, destination: Path) -> Path:
        nonlocal replaced
        if source_path == copied_manifest:
            # Validation has consumed the original metadata. Atomically replace
            # it with a same-size manifest naming an unpublished model.
            replacement_path = source / "replacement.json"
            replacement_path.write_bytes(replacement_bytes)
            replacement_path.replace(copied_manifest)
            replaced = True
        return original_stage_copy(source_path, destination)

    monkeypatch.setattr(publish, "_stage_copy", replace_before_staging)
    with pytest.raises(ValueError, match="SHA-256"):
        publish_browser_artifacts(
            copied_manifest, target, wasm_source_directory=wasm, channel=channel
        )
    assert replaced
    assert pointer.read_bytes() == previous == original_bytes
    assert not (target / onnx["file"]).exists()
    assert not list(target.rglob("*.tmp"))
