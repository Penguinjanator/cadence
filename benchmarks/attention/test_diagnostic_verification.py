"""Reject optimistic numerical summaries without repeating behavioral acquisition."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from cadence.receipts import Receipt

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "attention_diagnostic_verifier", HERE / "diagnostic_verification.py")
assert SPEC is not None and SPEC.loader is not None
verification = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verification)


def body():
    return Receipt.read(HERE / "results" / "heads-capacity-diagnostic.json").body


def test_retained_numerical_record_checks_without_solving():
    assert verification.check(body()) is None


@pytest.mark.parametrize("change", [
    "original_maximum", "original_residual", "warm", "memory", "read_drive",
    "tolerance", "source", "founder", "round", "confirmation", "replay_count",
    "replay_boolean", "parameter", "training_refusal", "training_action", "initial_work",
    "continuation_residual", "continuation_flag", "continuation_aggregate",
    "continuation_nan", "continuation_negative", "continuation_bool_residual",
    "continuation_missing_row", "continuation_steps", "continuation_checks",
    "longer_residual", "longer_halving", "longer_steps", "short_all_qualified",
    "short_row_custody", "short_invented_residuals", "short_dt", "short_checks",
    "actions", "learning", "transport", "hash", "charge", "drop_conservative_charge",
    "negative_time", "infinite_time", "summary_reference", "behavior_pass",
    "nested_behavior_pass", "missing",
])
def test_verifier_rejects_corrupted_residuals_custody_work_and_optimistic_claims(change):
    b = body()
    boundary = b["boundary"]
    replay = b["replay"]
    diagnostics = b["same_boundary_numerical_diagnostics"]
    continuation = diagnostics["continuation"]
    longer = diagnostics["damping3_budget8192"]
    short = diagnostics["damping3_budget1024"]
    charges = b["host_work"]
    if change == "original_maximum":
        boundary["original_maximum_residual"] = 0.0
    elif change == "original_residual":
        boundary["original_row_residuals"][0] = 0.0
    elif change == "warm":
        boundary["warm_state"] = {}
    elif change == "memory":
        boundary["held_bank_records"] = 1
    elif change == "read_drive":
        boundary["maximum_abs_read_source_drive"] = 3.0
    elif change == "tolerance":
        boundary["tolerance"] = 1.0
    elif change == "source":
        b["source_revision"] = "0" * 40
    elif change == "founder":
        b["founder"] = 1
    elif change == "round":
        b["round_1_based"] += 1
    elif change == "confirmation":
        b["confirmation_seeds_used"] = [200]
    elif change == "replay_count":
        replay["training_replays"] = 2
    elif change == "replay_boolean":
        replay["exact_training_decisions_and_work_match"] = 1
    elif change == "parameter":
        replay["parameter_digest"] = "0" * 64
    elif change == "training_refusal":
        replay["training_work"]["action_refusals"] = 0
    elif change == "training_action":
        replay["training_work"]["actions"] += 1
    elif change == "initial_work":
        replay["initial_evaluation_work"]["action_sweeps"] = 0
    elif change == "continuation_residual":
        continuation["row_residuals"][0] = 0.1
    elif change == "continuation_flag":
        continuation["qualified"][0] = False
    elif change == "continuation_aggregate":
        continuation["all_rows_qualified"] = 1
    elif change == "continuation_nan":
        continuation["row_residuals"][0] = float("nan")
    elif change == "continuation_negative":
        continuation["row_residuals"][0] = -0.1
    elif change == "continuation_bool_residual":
        continuation["row_residuals"][0] = False
    elif change == "continuation_missing_row":
        continuation["row_residuals"].pop()
    elif change == "continuation_steps":
        continuation["additional_sweeps"] += 32
    elif change == "continuation_checks":
        continuation["additional_residual_checks"] = 0
    elif change == "longer_residual":
        longer["row_residuals"][1] = 0.1
    elif change == "longer_halving":
        longer["halvings_used"] = 1
    elif change == "longer_steps":
        longer["steps"] += 32
    elif change == "short_all_qualified":
        short["qualified"] = [True] * 16
    elif change == "short_row_custody":
        short["row_residuals_retained"] = True
    elif change == "short_invented_residuals":
        short["row_residuals"] = [0.0] * 16
    elif change == "short_dt":
        short["attempts"][1]["dt"] = 0.1
    elif change == "short_checks":
        short["residual_checks"] = 0
    elif change == "actions":
        diagnostics["actions_issued"] = 1
    elif change == "learning":
        diagnostics["learning_updates"] = 1
    elif change == "transport":
        diagnostics["measured_reference_transport_work"][0]["memory_scan_macs"] = 1
    elif change == "hash":
        boundary["binary_snapshot"]["sha256"] = "not-a-digest"
    elif change == "charge":
        charges["diagnostic_charged_seconds"] = 0.0
    elif change == "drop_conservative_charge":
        charges["first_attempt_conservative_seconds"] = 0.0
        charges["diagnostic_charged_seconds"] -= 30.0
        charges["cumulative_charged_seconds"] -= 30.0
        charges["remaining_of_shared600_seconds"] += 30.0
    elif change == "negative_time":
        charges["second_diagnostic_measured_seconds"] = -1.0
    elif change == "infinite_time":
        charges["third_diagnostic_measured_seconds"] = float("inf")
    elif change == "summary_reference":
        b["summary_custody"]["snapshot"] = "../../unexpected"
    elif change == "behavior_pass":
        b["capacity_admitted"] = True
    elif change == "nested_behavior_pass":
        diagnostics["capacity_admitted"] = True
    else:
        del continuation["row_residuals"]
    assert verification.check(b) is not None


@pytest.mark.parametrize("value", [None, [], {}, {"summary_custody": []}])
def test_incomplete_or_malformed_records_fail_closed(value):
    assert verification.check(value) is not None
