"""Closed, shared browser runtime channel; model manifests cannot supply URLs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import FEATURE_SCHEMA_HASH, RULES_HASH_WIRE, RULES_SCHEMA_ID
from .native import SEARCH_ALGORITHM_ID


def qualified_runtime() -> dict[str, Any]:
    descriptor = json.loads(
        Path(__file__).with_name("browser_runtime.json").read_text()
    )
    if (
        descriptor.get("schema_version") != 1
        or descriptor.get("rules_hash") != RULES_HASH_WIRE
        or descriptor.get("rules_schema") != RULES_SCHEMA_ID
        or descriptor.get("feature_schema_hash") != f"{FEATURE_SCHEMA_HASH:016x}"
        or descriptor.get("search_algorithm") != SEARCH_ALGORITHM_ID
    ):
        raise ValueError("qualified browser runtime contracts are incompatible")
    files = []
    for key, filename in (
        ("module", "deltrel_wasm.js"),
        ("binary", "deltrel_wasm_bg.wasm"),
    ):
        artifact = descriptor[key]
        digest = artifact.get("sha256")
        if (
            set(artifact) != {"path", "sha256", "bytes"}
            or artifact["path"] != filename
            or type(artifact["bytes"]) is not int
            or not 0 < artifact["bytes"] <= 8 * 1024 * 1024
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("qualified browser runtime artifact is invalid")
        files.append({"path": filename, "sha256": digest, "bytes": artifact["bytes"]})
    identity = hashlib.sha256(
        json.dumps(files, separators=(",", ":")).encode()
    ).hexdigest()
    if identity != descriptor.get("runtime_content_sha256"):
        raise ValueError("qualified browser runtime content identity is invalid")
    return {
        **descriptor,
        "directory": f"wasm-{RULES_HASH_WIRE.split(':')[1]}-{identity}",
        "manifest": f"manifest-runtime-{identity}.json",
    }
