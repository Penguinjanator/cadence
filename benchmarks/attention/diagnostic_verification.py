"""Check the retained numerical boundary without rerunning a solver.

The original observed summary is immutable evidence. Arithmetic is recomputed
from retained row residuals where available. The short damping attempt retained
only a maximum and observed qualification flags: its aggregate failure is
checkable, but its individual row flags are custody checks, not recomputed errors.
Local binary hashes do not establish independent public equation reproduction.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL_SHA256 = "88cd3ccaff61b3e091eba1849a0bc4e5cee1d6a2b2542afb61a0aefb7d87814e"
ORIGINAL_DIGEST = "7ffafa9ae8216a966f12d20da066552ee74fdec4bd24d334edff636421c09088"
REFERENCE = "benchmarks/attention/results/heads-capacity-corrected.json"
ENRICHMENTS = {
    (): {"summary_custody"},
    ("same_boundary_numerical_diagnostics",): {"actions_issued", "learning_updates"},
    ("same_boundary_numerical_diagnostics", "continuation"): {"row_residuals", "qualified"},
    ("same_boundary_numerical_diagnostics", "damping3_budget8192"):
        {"row_residuals", "qualified"},
    ("same_boundary_numerical_diagnostics", "damping3_budget1024"):
        {"row_residuals_retained", "row_qualification_custody"},
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _finite(value: Any, minimum: float = 0.0) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


def _retained(original: Any, current: Any, path: tuple[str, ...] = ()) -> bool:
    """Every original observation survives, including its JSON type."""
    if isinstance(original, dict):
        return (isinstance(current, dict)
                and set(current) == set(original) | ENRICHMENTS.get(path, set()) and all(
            _retained(value, current[key], (*path, key))
            for key, value in original.items()))
    if isinstance(original, list):
        return (isinstance(current, list) and len(original) == len(current)
                and all(_retained(a, b, path) for a, b in zip(original, current, strict=True)))
    return type(original) is type(current) and original == current


def _rows(reading: dict[str, Any], batch: int, tolerance: float) -> None:
    residuals, flags = reading["row_residuals"], reading["qualified"]
    _require(isinstance(residuals, list) and len(residuals) == batch
             and all(_finite(x) for x in residuals), "invalid row residuals")
    _require(isinstance(flags, list) and len(flags) == batch
             and all(type(x) is bool for x in flags), "invalid qualification flags")
    expected = [x <= tolerance for x in residuals]
    _require(flags == expected, "row qualification differs from residuals")
    _require(reading["maximum_residual"] == max(residuals), "residual maximum differs")
    _require(type(reading["all_rows_qualified"]) is bool
             and reading["all_rows_qualified"] == all(expected),
             "aggregate qualification differs")


def check(body: dict[str, Any]) -> str | None:
    """Verify custody, refusals, numerical summaries and charges; never admit behavior."""
    try:
        custody = body["summary_custody"]
        _require(set(custody) == {"original_bytes_sha256", "original_receipt_digest",
                                  "snapshot", "enrichment"}, "invalid summary custody fields")
        relative = f"benchmarks/attention/results/sources/{ORIGINAL_SHA256}.txt"
        _require(custody["original_bytes_sha256"] == ORIGINAL_SHA256
                 and custody["original_receipt_digest"] == ORIGINAL_DIGEST
                 and custody["snapshot"] == relative, "original summary reference differs")
        path = ROOT / relative
        _require(hashlib.sha256(path.read_bytes()).hexdigest() == ORIGINAL_SHA256,
                 "original summary bytes differ")
        _require(Receipt.verify(path)[0], "original summary digest differs")
        original = Receipt.read(path)
        _require(original.digest == ORIGINAL_DIGEST and _retained(original.body, body),
                 "original numerical observations differ")
        _require(set(body) == set(original.body) | {"summary_custody"},
                 "unexpected diagnostic fields or behavior claims")

        _require(body["reference_receipt"]["path"] == REFERENCE, "unknown reference receipt")
        reference = Receipt.read(ROOT / REFERENCE)
        _require(Receipt.verify(ROOT / REFERENCE)[0]
                 and body["reference_receipt"]["digest"] == reference.digest,
                 "reference receipt digest differs")
        prior = reference.body
        _require(body["source_revision"] == prior["source_revision"], "source revision differs")
        _require(body["arm"] == "supplied_reads" and type(body["founder"]) is int
                 and body["founder"] == 0 and body["confirmation_seeds_used"] == [],
                 "unexpected arm, founder or confirmation work")
        run = next(r for r in prior["runs"] if r["arm"] == body["arm"]
                   and r["seed"] == body["founder"])
        protocol, training = prior["protocol"], run["training"]
        batch = protocol["task"]["batch"]
        completed = body["completed_rounds_before_refusal"]
        _require(_integer(completed) and completed == training["rounds"]
                 and _integer(body["round_1_based"], 1)
                 and body["round_1_based"] == completed + 1
                 and type(body["prefix_index_0_based"]) is int
                 and body["prefix_index_0_based"] == 0
                 and training["completed"] is False and training["stop"] == "refused_prefix",
                 "refusal boundary differs")

        replay = body["replay"]
        _require(type(replay["training_replays"]) is int and replay["training_replays"] == 1
                 and all(replay[key] is True for key in (
                     "exact_initial_evaluation_match", "exact_training_decisions_and_work_match",
                     "exact_final_parameter_digest_match")), "exact replay checks fail")
        _require(replay["parameter_digest"] == run["final_parameters"]
                 and replay["schedule_sha256"] == run["schedule_sha256"]
                 and replay["training_work"] == training["work"]
                 and replay["initial_evaluation_work"] == run["initial"]["work"],
                 "replay parameters, schedule or work differ")
        work = replay["training_work"]
        _require(work["action_refusals"] == 1 and work["lesson_refusals"] == 0
                 and work["actions"] == 4 * completed + 1
                 and work["action_rows"] == batch * work["actions"]
                 and work["presentations"] == sum(training["wrong_counts"])
                 and work["lessons"] == sum(n > 0 for n in training["wrong_counts"]),
                 "refused action or teaching work differs")
        for key in ("action_sweeps", "action_residual_checks"):
            row_key = key.replace("action_", "action_row_")
            _require(work[row_key] == batch * work[key], "row work differs")

        boundary = body["boundary"]
        _require(boundary["warm_state"] is None and boundary["held_bank_records"] == 0
                 and boundary["memory_amplitudes"] == [0.0, 0.0]
                 and boundary["maximum_abs_memory_drive"] == 0.0
                 and boundary["maximum_abs_read_source_drive"] == 0.0,
                 "boundary is not empty and unforced from rest")
        tolerance = boundary["tolerance"]
        _require(_finite(tolerance) and tolerance == protocol["brain"]["learning_tolerance"]
                 and boundary["dt"] == protocol["dt"]
                 and boundary["neuron_model"] == run["neuron_model"]
                 and boundary["original_sweeps"] == protocol["brain"]["learning_free_steps"],
                 "equations, tolerance or original budget differ")
        residuals = boundary["original_row_residuals"]
        _require(isinstance(residuals, list) and len(residuals) == batch
                 and all(_finite(x) for x in residuals)
                 and boundary["original_maximum_residual"] == max(residuals) > tolerance
                 and work["max_action_residual"] == max(residuals),
                 "original refusal residual differs")
        _require(boundary["largest_residual_row"] == residuals.index(max(residuals)),
                 "largest residual row differs")

        diagnostics = body["same_boundary_numerical_diagnostics"]
        _require(type(diagnostics["actions_issued"]) is int
                 and diagnostics["actions_issued"] == 0
                 and type(diagnostics["learning_updates"]) is int
                 and diagnostics["learning_updates"] == 0,
                 "a numerical diagnostic issued actions or learned")
        continuation, longer = diagnostics["continuation"], diagnostics["damping3_budget8192"]
        for reading in (continuation, longer):
            _rows(reading, batch, tolerance)
            _require(reading["all_rows_qualified"] is True, "longer boundary does not qualify")
        _require(_integer(continuation["additional_sweeps"], 1)
                 and continuation["total_sweeps"] == boundary["original_sweeps"]
                 + continuation["additional_sweeps"] == longer["steps"] <= 8192
                 and longer["halvings_used"] == 0
                 and continuation["row_residuals"] == longer["row_residuals"]
                 and continuation["additional_residual_checks"]
                 == math.ceil(continuation["additional_sweeps"] / 32) + 1
                 and longer["residual_checks"] == math.ceil(longer["steps"] / 32) + 1,
                 "continuation/damping sweep accounting differs")

        short = diagnostics["damping3_budget1024"]
        _require(short["row_residuals_retained"] is False and "row_residuals" not in short,
                 "short damping row-residual custody differs")
        flags = short["qualified"]
        _require(isinstance(flags, list) and len(flags) == batch
                 and all(type(x) is bool for x in flags) and not all(flags)
                 and _finite(short["maximum_original_residual"])
                 and short["maximum_original_residual"] > tolerance,
                 "short damping aggregate failure differs")
        attempts = short["attempts"]
        _require(len(attempts) == 4 and short["steps"] == boundary["original_sweeps"]
                 and short["steps"] == sum(attempt["steps"] for attempt in attempts),
                 "damping budget differs")
        for h, attempt in enumerate(attempts):
            _require(type(attempt["halving"]) is int and attempt["halving"] == h
                     and attempt["dt"] == boundary["dt"] / 2 ** h
                     and _integer(attempt["steps"], 1)
                     and attempt["steps"] == short["steps"] // 4
                     and _finite(attempt["maximum_original_residual"])
                     and attempt["maximum_original_residual"] > tolerance,
                     "damping attempt differs")
        _require(short["maximum_original_residual"] == attempts[-1]["maximum_original_residual"]
                 and short["residual_checks"] == sum(
                     math.ceil(attempt["steps"] / 32) + 1 for attempt in attempts) + 3
                 and _integer(short["stagnation_checks"])
                 and short["stagnation_checks"] <= short["residual_checks"],
                 "damping checks or final residual differ")

        for transports in diagnostics["measured_reference_transport_work"]:
            _require(all(_integer(value) for value in transports.values()),
                     "invalid transport work")
            calls = transports["memory_drive_evaluations"]
            _require(calls % 2 == 0 and transports["memory_scan_macs"] == 0
                     and transports["softmax_elements"] == batch * calls
                     and transports["cortical_synapse_transports"]
                     == batch * run["synapses"] * calls // 2, "transport work differs")

        snapshot = boundary["binary_snapshot"]
        _require(_digest(snapshot["sha256"]) and _integer(snapshot["bytes"], 1)
                 and _digest(diagnostics["state_snapshot"]["sha256"]),
                 "invalid boundary/state data hashes")
        for fingerprint in snapshot["selected_array_fingerprints"].values():
            _require(_digest(fingerprint["sha256_c_order_bytes"])
                     and all(_integer(size) for size in fingerprint["shape"]),
                     "invalid boundary array fingerprint")

        charges = body["host_work"]
        numeric = {key: value for key, value in charges.items() if key.endswith("seconds")}
        _require(all(_finite(value) for value in numeric.values()), "invalid host charge")
        _require(charges["prior_receipt_elapsed_seconds"] == prior["elapsed_host_seconds"]
                 and charges["prior_receipt_previous_seconds"] == prior["previous_host_seconds"]
                 and charges["prior_cumulative_seconds"] == charges["prior_receipt_elapsed_seconds"]
                 + charges["prior_receipt_previous_seconds"]
                 and charges["first_attempt_conservative_seconds"] == 30.0
                 and charges["diagnostic_charged_seconds"]
                 == charges["first_attempt_conservative_seconds"]
                 + charges["second_diagnostic_measured_seconds"]
                 + charges["third_diagnostic_measured_seconds"]
                 and charges["cumulative_charged_seconds"] == charges["prior_cumulative_seconds"]
                 + charges["diagnostic_charged_seconds"]
                 and charges["remaining_of_shared600_seconds"]
                 == 600 - charges["cumulative_charged_seconds"], "host charge arithmetic differs")
        return None
    except (KeyError, TypeError, ValueError, OSError, StopIteration, OverflowError):
        return "invalid numerical diagnostic: custody, residuals, work or charges differ"
