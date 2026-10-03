"""Verified atomic publication of trained browser inference artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from .checkpoint import sha256_file, verify_file
from .contracts import RULES_HASH_HEX
from .browser_runtime import qualified_runtime
from .distill import (
    BROWSER_MANIFEST_FORMAT,
    BROWSER_MANIFEST_SCHEMA_VERSION,
    validate_browser_onnx,
)

WASM_ASSET_DIRECTORY = f"wasm-{RULES_HASH_HEX}-champion-v1"


def publish_browser_artifacts(
    manifest_path: str | Path,
    target_directory: str | Path,
    *,
    wasm_build_command: Sequence[str] = (),
    wasm_working_directory: str | Path | None = None,
    wasm_source_directory: str | Path | None = None,
    channel: str = "legacy",
) -> dict[str, object]:
    if channel not in ("legacy", "qualified"):
        raise ValueError("browser channel must be legacy or qualified")
    runtime = qualified_runtime() if channel == "qualified" else None
    if runtime is not None and wasm_build_command:
        raise ValueError(
            "qualified runtime must be prebuilt; publication cannot rebuild it"
        )
    source_manifest = Path(manifest_path).resolve()
    # Pin the exact bytes that supplied the validated artifact metadata. A later
    # atomic replacement must not substitute an unvalidated release pointer.
    source_manifest_bytes = source_manifest.read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    payload = json.loads(source_manifest_bytes)
    if (
        not isinstance(payload, dict)
        or payload.get("format") != BROWSER_MANIFEST_FORMAT
        or payload.get("schema_version") != BROWSER_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError("unsupported browser model manifest")
    if runtime is not None and (
        not isinstance(payload.get("rules"), Mapping)
        or not isinstance(payload.get("features"), Mapping)
        or payload["rules"].get("hash") != runtime["rules_hash"]
        or payload["features"].get("hash") != runtime["feature_schema_hash"]
        or payload.get("weights") != "ema"
    ):
        raise ValueError("model is incompatible with the qualified browser runtime")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("browser manifest artifacts are missing")
    resolved: dict[str, Path] = {}
    for name in ("onnx", "checkpoint"):
        entry = artifacts.get(name)
        if not isinstance(entry, Mapping):
            raise ValueError(f"browser manifest {name} artifact is missing")
        filename = entry.get("file")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
        ):
            raise ValueError(f"browser manifest {name} filename is unsafe")
        source = source_manifest.parent / filename
        verify_file(
            source,
            expected_sha256=_sha256(entry.get("sha256")),
            expected_bytes=_positive_int("bytes", entry.get("bytes")),
        )
        resolved[name] = source

    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or not isinstance(
        tensors.get("outputs"), Mapping
    ):
        raise ValueError("browser model output contract is missing")
    precision = payload.get("precision")
    if precision not in ("float16", "float32"):
        raise ValueError("browser precision must be float16 or float32")
    validate_browser_onnx(
        resolved["onnx"],
        include_auxiliary=len(tensors["outputs"]) == 11,
        precision=precision,
    )

    if wasm_build_command:
        subprocess.run(
            list(wasm_build_command),
            cwd=(
                str(Path(wasm_working_directory).resolve())
                if wasm_working_directory is not None
                else None
            ),
            check=True,
        )
    target = Path(target_directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    wasm_directory = (
        str(runtime["directory"]) if runtime is not None else WASM_ASSET_DIRECTORY
    )
    wasm_source = (
        Path(wasm_source_directory).resolve()
        if wasm_source_directory is not None
        else target / wasm_directory
    )
    wasm_sources = {
        "module": wasm_source / "deltrel_wasm.js",
        "binary": wasm_source / "deltrel_wasm_bg.wasm",
    }
    _verify_wasm(wasm_sources["module"], wasm_sources["binary"])
    if runtime is not None:
        for name, source in wasm_sources.items():
            verify_file(
                source,
                expected_sha256=runtime[name]["sha256"],
                expected_bytes=runtime[name]["bytes"],
            )

    wasm_target = target / wasm_directory
    if runtime is not None and wasm_target.resolve() != wasm_target:
        raise ValueError("qualified runtime directory cannot contain symlinks")
    wasm_target.mkdir(parents=True, exist_ok=True)
    destination_onnx = target / resolved["onnx"].name
    destination_manifest = target / (
        str(runtime["manifest"]) if runtime is not None else "manifest.json"
    )
    destinations = {
        "module": wasm_target / "deltrel_wasm.js",
        "binary": wasm_target / "deltrel_wasm_bg.wasm",
        "onnx": destination_onnx,
        "manifest": destination_manifest,
    }
    staged: dict[str, Path] = {}
    try:
        staged["module"] = _stage_copy(wasm_sources["module"], destinations["module"])
        staged["binary"] = _stage_copy(wasm_sources["binary"], destinations["binary"])
        staged["onnx"] = _stage_copy(resolved["onnx"], destination_onnx)
        staged["manifest"] = _stage_copy(source_manifest, destination_manifest)
        _verify_wasm(staged["module"], staged["binary"])
        if runtime is not None:
            for name in ("module", "binary"):
                verify_file(
                    staged[name],
                    expected_sha256=runtime[name]["sha256"],
                    expected_bytes=runtime[name]["bytes"],
                )
        onnx_entry = artifacts["onnx"]
        verify_file(
            staged["onnx"],
            expected_sha256=str(onnx_entry["sha256"]),
            expected_bytes=int(onnx_entry["bytes"]),
        )
        verify_file(
            staged["manifest"],
            expected_sha256=source_manifest_sha256,
            expected_bytes=len(source_manifest_bytes),
        )
        # A qualified channel never overwrites immutable runtime/model bytes,
        # including on a racing writer. The legacy default retains its API.
        if runtime is not None:
            for name in ("module", "binary", "onnx"):
                source = staged[name]
                try:
                    os.link(source, destinations[name])
                except FileExistsError:
                    if destinations[name].is_symlink():
                        raise ValueError(
                            "immutable browser artifact cannot be a symlink"
                        )
                    verify_file(
                        destinations[name],
                        expected_sha256=sha256_file(source),
                        expected_bytes=source.stat().st_size,
                    )
                source.unlink()
                staged.pop(name)
            _fsync_directory(wasm_target)
            _fsync_directory(target)
        # Only the selected mutable channel manifest is replaced, strictly last.
        for name in ("module", "binary", "onnx", "manifest"):
            if name not in staged:
                continue
            os.replace(staged[name], destinations[name])
            staged.pop(name)
        _fsync_directory(target)
        _fsync_directory(wasm_target)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return {
        "manifest": str(destination_manifest),
        "onnx": str(destination_onnx),
        "model_version": payload.get("model_version"),
        **({"channel": channel} if runtime is not None else {}),
        "wasm_build_invoked": bool(wasm_build_command),
        "wasm_module": str(destinations["module"]),
        "wasm_binary": str(destinations["binary"]),
        "wasm_module_sha256": sha256_file(destinations["module"]),
        "wasm_binary_sha256": sha256_file(destinations["binary"]),
    }


def publish_browser_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Verify and atomically publish trained browser inference artifacts"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--wasm-cwd")
    parser.add_argument("--wasm-source")
    parser.add_argument("--channel", choices=("legacy", "qualified"), default="legacy")
    parser.add_argument("--wasm-build", nargs=argparse.REMAINDER, default=())
    arguments = parser.parse_args(argv)
    result = publish_browser_artifacts(
        arguments.manifest,
        arguments.target,
        wasm_build_command=arguments.wasm_build,
        wasm_working_directory=arguments.wasm_cwd,
        wasm_source_directory=arguments.wasm_source,
        channel=arguments.channel,
    )
    print(json.dumps(result, sort_keys=True))


def _stage_copy(source: Path, destination: Path) -> Path:
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            with source.open("rb") as stream:
                shutil.copyfileobj(stream, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        staged = Path(temporary_name)
        staged.chmod(0o644)
        temporary_name = None
        return staged
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _verify_wasm(module: Path, binary: Path) -> None:
    if not module.is_file() or module.stat().st_size <= 0:
        raise ValueError("WASM JavaScript module is missing or empty")
    try:
        javascript = module.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("WASM JavaScript module is not UTF-8") from exc
    if not any(token in javascript for token in ("export", "WebAssembly")):
        raise ValueError("WASM JavaScript module lacks an export/runtime")
    if not binary.is_file() or binary.stat().st_size < 8:
        raise ValueError("WASM binary is missing or too small")
    with binary.open("rb") as stream:
        if stream.read(4) != b"\x00asm":
            raise ValueError("WASM binary magic is invalid")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("artifact SHA-256 is invalid")
    return value


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"artifact {name} must be a positive integer")
    return value
