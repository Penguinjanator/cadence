#!/usr/bin/env python3
"""Build the curated proofs and audit every theorem's axioms with Lean itself.

Only an already provisioned dependency cache is used. The manifest pins the
dependencies, sources.json pins the imported subset, and verification.json
binds this checker and the current package sources to the result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_only(text: str) -> str:
    """Remove nested Lean comments, preserving newlines and string literals."""
    out, depth, quoted, index = [], 0, False, 0
    while index < len(text):
        char, pair = text[index], text[index:index + 2]
        if depth:
            if pair == "/-":
                depth += 1
                out.extend("  ")
                index += 2
            elif pair == "-/":
                depth -= 1
                out.extend("  ")
                index += 2
            else:
                out.append("\n" if char == "\n" else " ")
                index += 1
        elif quoted:
            out.append(char)
            index += 1
            if char == "\\" and index < len(text):
                out.append(text[index])
                index += 1
            elif char == '"':
                quoted = False
        elif pair == "/-":
            depth = 1
            out.extend("  ")
            index += 2
        elif pair == "--":
            stop = text.find("\n", index)
            index = len(text) if stop < 0 else stop
        else:
            out.append(char)
            quoted = char == '"'
            index += 1
    if depth or quoted:
        raise ValueError("unterminated Lean comment or string")
    return "".join(out)


def theorem_names(text: str) -> list[str]:
    """Inventory the explicit, named theorem/lemma declarations in this subset."""
    clean = code_only(text)
    if re.search(r"\b(?:sorry|admit|axiom)\b", clean):
        raise ValueError("admission or project-defined axiom in proof source")
    scopes: list[tuple[str, str]] = []
    names = []
    for line in clean.splitlines():
        line = line.strip()
        match = re.match(r"namespace\s+(\S+)", line)
        if match:
            scopes.append(("namespace", match.group(1)))
        elif re.match(r"(?:noncomputable\s+)?section(?:\s|$)", line):
            scopes.append(("section", ""))
        elif re.match(r"end(?:\s|$)", line):
            if not scopes:
                raise ValueError("unmatched scope end")
            scopes.pop()
        elif match := re.match(r"(?:theorem|lemma)\s+(\S+)", line):
            name = match.group(1).rstrip(":")
            prefix = ".".join(name for kind, name in scopes if kind == "namespace")
            names.append(prefix + "." + name if prefix else name)
    if scopes:
        raise ValueError("unclosed namespace or section")
    if len(re.findall(r"\b(?:theorem|lemma)\b", clean)) != len(names):
        raise ValueError("unsupported theorem declaration syntax; extend the reviewed inventory")
    return names


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                            env={**os.environ, "LEAN_NUM_THREADS": "2"})
    if result.returncode:
        raise RuntimeError(f"{' '.join(command)}\n{result.stdout}\n{result.stderr}")
    return result


def dependencies() -> dict[str, str]:
    manifest = json.loads((ROOT / "lake-manifest.json").read_text())
    result = {}
    for package in manifest["packages"]:
        directory = ROOT / ".lake/packages" / package["name"]
        if not (directory / package["configFile"]).is_file():
            raise ValueError("Dependency cache missing; provision the pinned packages first: "
                             + package["name"])
        head = run(["git", "-C", str(directory), "rev-parse", "HEAD"]).stdout.strip()
        if head != package["rev"]:
            raise ValueError("Dependency revision differs from lockfile: " + package["name"])
        changes = run(["git", "-C", str(directory), "status", "--porcelain",
                       "--untracked-files=no"]).stdout
        if changes:
            raise ValueError("Dependency has tracked modifications: " + package["name"])
        result[package["name"]] = head
    return result


def parse_axioms(output: str, expected: list[str]) -> dict[str, list[str]]:
    matches = re.findall(
        r"'([^']+)' (?:depends on axioms:\s*\[([^\]]*)\]|does not depend on any axioms)",
        output, re.S,
    )
    if len(matches) != len(expected) or {name for name, _ in matches} != set(expected):
        raise ValueError("Lean axiom output does not cover every inventoried theorem exactly once")
    result = {}
    for name, text in matches:
        axioms = [item.strip() for item in text.split(",") if item.strip()]
        if set(axioms) - ALLOWED_AXIOMS:
            raise ValueError(f"unexpected axiom in {name}: {axioms}")
        result[name] = axioms
    return result


def check_toolchain_version(output: str, pinned: str) -> None:
    expected = re.fullmatch(r"leanprover/lean4:v([^\s]+)", pinned)
    actual = re.match(r"Lean \(version ([^,\s)]+)", output)
    if expected is None or actual is None or actual.group(1) != expected.group(1):
        raise ValueError(f"Invoked Lean version differs from pinned toolchain: {output!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / "verification.json")
    args = parser.parse_args()
    started = time.perf_counter()
    provenance = json.loads((ROOT / "sources.json").read_text())
    names, counts = [], {}
    for source in provenance["sources"]:
        path = ROOT / source["path"]
        if sha256(path) != source["sha256"]:
            raise ValueError("Proof source differs from its reviewed manifest: " + source["path"])
        declared = theorem_names(path.read_text())
        if not declared:
            raise ValueError("Proof module has no inventoried theorem: " + source["path"])
        names.extend(declared)
        counts[source["scope"]] = counts.get(source["scope"], 0) + len(declared)
    if len(names) != len(set(names)):
        raise ValueError("Duplicate theorem name")
    # Also inspect aggregate modules: an extra source must be reviewed and inventoried.
    known = {source["path"] for source in provenance["sources"]}
    roots = {"Cadence.lean", "CadenceFlagship.lean", "CadenceMission.lean",
             "CadenceProofs.lean", "CadenceRecords.lean"}
    actual = {str(path.relative_to(ROOT)) for path in ROOT.rglob("*.lean")
              if ".lake" not in path.relative_to(ROOT).parts}
    if actual != known | roots:
        raise ValueError("Uninventoried or missing Lean source: " + str(actual ^ (known | roots)))
    for root in roots:
        if theorem_names((ROOT / root).read_text()):
            raise ValueError("Aggregate modules must contain imports, not unlisted theorems")
    package_paths = sorted(ROOT / path for path in actual) + [
        ROOT / name for name in ("check.py", "test_check.py", "README.md", "sources.json",
                                "lakefile.toml", "lake-manifest.json", "lean-toolchain")]
    source_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in package_paths}
    dependency_revisions = dependencies()
    toolchain = (ROOT / "lean-toolchain").read_text().strip()
    actual_lean_version = run(["lake", "--no-cache", "env", "lean", "--version"]).stdout.strip()
    check_toolchain_version(actual_lean_version, toolchain)
    build = run(["lake", "--no-cache", "build", "CadenceProofs", "CadenceRecords"])
    audit = ROOT / ".lake/AxiomAudit.lean"
    audit.write_text("import CadenceProofs\nimport CadenceRecords\n\n"
                     + "\n".join("#print axioms " + name for name in names) + "\n")
    checked = run(["lake", "--no-cache", "env", "lean", str(audit)])
    axioms = parse_axioms(checked.stdout, names)
    if any(sha256(ROOT / path) != value for path, value in source_hashes.items()):
        raise ValueError("Package sources changed during verification")
    receipt = {
        "schema": "cadence.lean-verification/v1",
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "toolchain": toolchain, "actual_lean_version": actual_lean_version,
        "sources": source_hashes, "dependency_revisions": dependency_revisions,
        "theorems": len(names), "theorems_by_scope": counts, "axioms": axioms,
        "allowed_foundational_axioms": sorted(ALLOWED_AXIOMS),
        "version_command": ["lake", "--no-cache", "env", "lean", "--version"],
        "build_command": ["lake", "--no-cache", "build", "CadenceProofs", "CadenceRecords"],
        "audit_command": ["lake", "--no-cache", "env", "lean", ".lake/AxiomAudit.lean"],
        "build_stdout": build.stdout, "build_stderr": build.stderr,
        "seconds": time.perf_counter() - started,
        "limits": "Conditional mathematical results; no formal refinement of Python, hardware, "
                  "learned semantic relevance, useful retention, or the general "
                  "equilibrium-propagation gradient theorem is asserted. "
                  "Optional records are not part of PatchNet.",
    }
    receipt["receipt_sha256"] = hashlib.sha256(json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"theorems": len(names), "scopes": counts,
                      "receipt": str(args.receipt), "seconds": receipt["seconds"]}))


if __name__ == "__main__":
    main()
